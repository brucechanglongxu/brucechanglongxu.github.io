---
title: "Real world ASI II: Infrastructure"
date: 2025-09-05T14:17:07-07:00
draft: false
math: true
---

<!-- MathJax site-local init -->
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$','$$'], ['\\[','\\]']]
  },
  options: {
    skipHtmlTags: ['script','noscript','style','textarea','pre','code']
  },
  svg: { fontCache: 'global' }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" id="MathJax-script" async></script>


GPUs and custom AI hardware are the engines driving progress in AI research and applications. But raw hardware alone doesn’t explain why some teams ship reliable systems at scale while others struggle with bottlenecks and cost explosions. The difference comes down to how you think about optimization.

A useful way to frame it is as **three nested loops** wrapped around every model. Each loop sits at a different layer of abstraction, and each has its own gauges, and levers. The innermost loop is about the raw efficiency of a single device: is the GPU actually doing useful math, or is it starved by memory bandwidth or kernel overhead? The next loop out is the training loop: how do we scale beyond one GPU and make hundreds or thousands of them behave like a single machine? And finally, there is the product loop: once the model is trained, can we serve it reliably under real-world traffic, meet our latency and safety SLOs, and do so at a cost per token that won't sink the business? 

What makes this framing powerful is that most "mysterious" performance or reliability problems stop being mysterious once you identify which loop you are really in. If a profiler shows long gaps on the GPU timeline, that is an inner loop problem. If throughput collapses when yous cale from one rack to three, that is the training loop. If P95 latency is spiky even though single-GPU kernels are fast, you are in the product loop. The moment you can name the loop, you know where to look and which tools to bring to bear. 

## The Inner Loop: making a single GPU sing

At the very center of AI infrastructure is the question: **is your GPU doing useful math, or is it wasting cycles?** Every model, no matter how large, eventually boils down to kernels running on a single device. If those kernels are inefficient, no amount of distributed wizardry will save you - because inefficiency just multiplies as you scale. The innermost loop is about runtime and kernel performance: understanding where the cycles are going, and how to keep the accelerator fed with a steady diet of compute. 

The mental model here is the _roofline_. Imagine a chart with two axes: arithmetic intensity (FLOPs per byte of memory moved) and achievable performance as a fraction of the device's peak. Your kernels live somewhere on this curve. If they are stuck in the low-intensity, memory-bound region, HBM-bandwidth - not Tensor Cores - is your bottleneck. If they are compute bound, the question becomes whether you are actually saturating those Tensor Cores. And if they are nowhere near the roofline at all, overheads are eating you alive - kernel launch latency, Python-GIL stalls [^1], or CPU-GPU syncs that keep the device idle. 

**Profiling is your compass:** Tools like Nsight Systems or PyTorch's built-in profilers don't just tell you how long each op takes - they show whether your GPU is mostly waiting on memory, struggling with occupancy, or simply not being kept busy. A typical training run might reveal long gaps between kernels, pointing to launch overhead. Or you might see HBM pegged at 90% while tensor utilization hovers at 20%, a classic sign that you are memory bound. 

This isn't just hypothetical. Microbenchmark studies on Volta and Turing GPUs have shown just how sharp the drop-offs can be. On Volta, for example, the L1 cache hit latency was measured at only 28 cycles, compared to 84 cycles on Pascal - nearly a 3 times improvement, but only if your working set fits. On Turing's T4, researchers found that vectorized 128-bit global memory instructions doubled the throughput of a simple SAXPY kernel compared to NVIDIA's own cuBLAS implementation that used 64-bit loads. The point is that the line between compute- and memory-bound is not theoretical: it shows up directly in latency histograms and throughput counters, and it shifts with every architectural generation. 

When _memory is the problem_, the fix is often algorithmic. Attention kernels are a perfect example. Naive implementations read and write enormouse intermediate matrices, swamping HBM. FlashAttention-style kernels, by contrast, restructure the computation to minimize memory traffic, moving closer to the compute-bound region of the roofline. Similarly, fusing QKV projections or using block-sparse attention can trade redundant memory ops for denser math. 

When _overhead is the problem_, you need to reduce the number of trips between host and device. Every kernel launch carries a cost, and if your training loop is littered with tiny ops, that cost adds up. The solution is fusion: combining bias, activation, and dropout in a single kernel, or capturing whole decode loops with CUDA Graphs so they run as a single replayable unit. The mantra is **"fewer, fatter kernels."**

And whent he GPU seems underfed, the problem often lies outside the device. Dataloaders stuck behind the Python GIL, disk I/O that cannot keep up with training, or underpinned host memory can all cause the device to starve. In those cases, the fix is not in CUDA at all but how you structure the pipeline feeding the accelerator: double-buffering, pagelocked transfers, and async prefetch all matter here. 

Finally, there are subtle low-level effects taht the microbenchmark papers document (Jia et al.) and that few developers think about. Volta and Turing both use two 64-bit wide register banks, which means that certain three-operand instructions like FFMA can still suffer from bank conflicts if all sources happen to map to the same bank. Researchers showed that remapping registers to avoid these conflicts yielded 15% throughput improvements. Similarly, the T4's "uniform datapath" for scalar integer ops was introduced precisely to prevent loop counters and address arithmetic from polluting the main floating-point pipelines. These are the kinds of details that explain why a kernel doesn't hit its theoretical throughput even when you've done everything else "right". 

> **In short**, _the inner loop is about **discipline at the kernel level**._ 

It is the layer where hardware meets math. Get this wrong, and your scaling experiments or serving infrastructure will always be operating at a handicap. Get it right, and you establish the foundation on which the outer loops - distributed training and serving - can actually pay off. 

### Memory hierarchy (L1/L2/TLB): why kernels stall

Modern accelerators live and die by their memory systems. The same kernel can look _compute-bound_ on one device and _memory-bound_ on another simply because L1/L2/DRAM behavior shifts the roofline under your feet. We will build a mental model of the GPU memory stack, and what to do when profiles show that our math engines are waiting. 

### Case study: 2-simplicial attention meets the roofline (TLX on H100)

As a concrete example of inner-loop optimization paying real dividends, the PyTorch team recentlyd etailed a fused TLX (Triton Low-level Extensions) kernel for **2-Simplicial Attention** - an attention variant that models trilinear interactions among token triples. 

## The Training Loop: making many GPUs act like one

Once we have squeezed everything we can out of a single GPU, the next challenge is scale. Modern foundation models don't fit on a single device's memory, and even if they did, training them in a reasonable wall-clock time requires spreading the work across dozens, hundreds, or thousands of accelerators. That shift takes us from the **inner loop** (are we doing useful math on one GPU?) to the **training loop:** how do we coordinate many GPUs so they behave like one coherent machine? 

The essence of the training loop is deceptively simple. Every step of the gradient descent has two phases: forward/backward computation, and parameter updates. On a single device, this happens locally. At scale, the computation must be partitioned, and the updates must be communicated. Suddely, your "roofline" is no longer defined just by arithmetic intensity versus memory bandwidth - it is defined by the balance between compute, memory, and communication across the whole cluster. 

This introduces a new kind of bottleneck: the interconnect. Within a single node, GPUs may talk over NVLink or NVSwitch, with hundreds of GB/s bandwidth and low latency. Across racks, communication shfits to InfiniBand and Ethernet fabrics, where latency is higher and oversubscription looms. What looks compute-bound in isolation can become communication-bound at scale. For example, an attention layer that runs smoothly on eight GPUs might grind to a halt on 256 if gradient all-reduce saturates the fabric or if pipeline bubbles dominate. 

The training loop is also where parallelism strategies become design choices. Should you shard your data and simply average gradients (data parallelism)? Split your matmuls across devices (tensor parallelism)? Cut the model into layer chunks and run a conveyer belt (pipeline parallelism)? Or introduce sparsity with mixtures of experts (MoE)? Each strategy trades compute, memory, and communication differently, and the optimal recipe depends on both the model architecture and the topology of your hardware. 

And finally, the training loop is where **failure modes multiply**. On a single device, a kernel crash just means restarting the process. On a thousand-GPU job, one flaky node can stall everything unless you've built checkpointing, elasticity, and fault tolerance into the loop. The reliability of the whole training stack is set here. 

> **Put simply:** _the training loop is the art of making many GPUs act like one._ 

It's where distributed systems thinking meets numerical optimization. And it's the layer where your wall-clock training time is determined - not just by how fast a GPU runs its kernels, but by how efficiency you can keep a fleet of them working together without stepping on each other's toes. 

### The Mechanics of Parallelism

When you scale beyond a single GPU, you have four main levers to pull: **data parallelism, tensor parallelism, pipeline parallelism, and mixture of experts**. Each one slices the problem differently. Think of them as four ways of making the same model fit into distributed hardware, each with its own costs and quirks. 

**Data parallelism** is the most straightforward. Every GPU (or groups of GPUs) gets a copy of the model, processes a different shard of the training data, and at the end of each step all the gradients are averaged. It is conceptually simple and works well for medium-sized models, but it comes with a hidden tax: every device must hold a fully copy of the model's parameters and optimizer state. For tody's multi-hundred-billion-parameter models, that memory overhead is unsustainable unless you shard optimizer state with systems like ZeRO or Fully Sharded Data Parallel (FSDP). Data parallelism scales throughput linearly with more devices, but only if the all-reduce [^2] bandwidth is there to keep up. 

**Tensor parallelism** digs deeper. Instead of copying the whole model, you split the math inside a single layer. Imagine a huge matrix multiply: one GPU handles the first block of columns, another the next block, and so on. After each multiply, results are exchanged and stitched back together. This approach lets you fit extremely large layers across multiple GPUs, but it ties you to fast interconnects. Within a single server with NVLink or NVSwitch, tensor parallelism is efficient; stretch it across racks with slower fabrics, and the communication cost can swamp the benefits.

**Pipeline parallelism** cuts along a different axis: the depth of the model. Rather than splitting layers within a block, you assign whole layers to different GPUs, and let minibatches flow like cars on a factory conveyor belt. Stage one computes the first few layers, passes the activations forward; stage two continues, and so on. Once the pipeline is full, utilization looks high, but there’s always some idle time—bubbles at the start and end of each pass. Managing these bubbles is a game of microbatching: too few, and your pipeline starves; too many, and memory pressure spikes.

And then there are **mixtures of experts (MoE)**, which change the rules altogether. Instead of every input flowing through every parameter, only a sparse subset of “experts” are activated per token. This makes it possible to train trillion-parameter models without paying a trillion-parameter compute bill on every forward pass. The price you pay is a more complex communication pattern: tokens must be routed to the right experts, which requires all-to-all traffic across devices. You also inherit a new optimization problem: keeping the load balanced across experts so that some don’t become bottlenecks while others sit idle.

In practice, large-scale training is almost always a hybrid. A cluster might use tensor parallelism inside a node to exploit NVLink bandwidth, pipeline parallelism across nodes to manage memory, and data parallelism on top to shard the dataset. MoE can be layered on top of that, introducing sparsity to extend capacity further. The art is in choosing the recipe that matches your topology. A 512-GPU DGX pod wired with NVSwitch will want a different mix than a cloud cluster built on Ethernet. 

This is the heart of the training loop: balancing compute, memory, and communication by deciding how to partition the model. Each form of parallelism is a different answer to the same question: _how do we make this model fit across the hardware we actually have, without leaving performance on the floor?_ 

### A Practical Throughput Model

Once you've chosen a parallelism strategy, the next question is: _what throughput will I actually get?_ Raw FLOP counts don't tell the story at scale. The real unit of progress in training is **tokens per second per GPU**, and wall-clock convergence depends on how well you keep that number high. A useful way to think about step time is with a simple model:

$$T_{step} = \textbf{max}(T_{compute}, T_{comm}) + T_{bubble} + T_{overhead}$$

The compute time is how long the math itself takes - the sum of forward and backward passes [^3] once you've squeezed the kernels with roofline discipline. $T_{comm}$ is the communication cost: gradient all-reduces, activation exchanges, expert dispatches. If these don't overlap well with compute, they set the floor. $T_{bubble}$ captures pipeline idle time. When you split a model into stages, some GPUs sit idle while the pipeline fills or drains. Microbatching controls this, but it never fully disappears. $T_{overhead}$ is everything else: framework bookkeeping, launch latency, data loading, scheduling hiccups. 

### Failure Modes at Scale

Scaling training across racks of GPUs isn’t just about bandwidth and FLOPs. It’s also about fragility. The bigger your job, the more ways it can fall apart. Every team that has trained a large model has lived through the same failures: sudden throughput collapse, out-of-memory explosions, inexplicable bubbles in the pipeline, or MoE experts that refuse to balance. Understanding these failure modes—and how to tame them—is the difference between a run that converges in weeks and one that limps along for months.

**OOM isn't just about parameters:** At small scale, “out of memory” means your batch doesn’t fit. At large scale, it gets subtler. The optimizer state balloons with data parallelism. Activations pile up in pipeline stages. Gradient shards and activation rematerialization interact in non-obvious ways. The mitigation playbook is now well established: shard optimizer states with ZeRO or FSDP; checkpoint or recompute activations to trade memory for extra FLOPs; drop precision (FP16, BF16, FP8) where numerically safe. But the real discipline is in modeling memory budgets ahead of time. Teams that don’t do this end up firefighting OOM crashes mid-run.

**Throughput collapse at scale:** Every scaling curve looks linear—until it doesn’t. The telltale sign is tokens/sec per GPU dropping sharply as you cross some cluster boundary (say, 8 → 64 → 512 GPUs). The root cause is almost always communication outpacing compute. Maybe gradient all-reduce buckets are too small and serialize; maybe tensor-parallel collectives stretch across a fabric that can’t hide latency; maybe your comm streams aren’t overlapping with math. The mitigation is topology-aware design: keep tensor parallelism inside NVLink islands, push pipeline parallelism across racks, size communication buckets to amortize latency but not starve overlap. It sounds obvious, but most jobs die here first.

**Pipeline bubbles (hidden idle time)** Pipeline parallelism looks elegant on paper - different stages working in lockstep. In practice, the edges fray. When microbatch count is too low, early stages sit idle while later ones drain; when it's too high, memory usage balloons. 

## Closing Remarks

AI infrastructure is diagnosis first, optimization second. In the **inner loop**, we should decide if we're compute, memory, or overhead bound and act accordingly. In the **training loop**, we choose the right parallelism/memory/communication mix and make failure routine.

### References
1. Zhe Jia, Marco Maggioni, Benjamin Staiger, Daniele P. Scarpazza. Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking. Technical Report, Citadel Enterprise Americas, LLC. arXiv:1804.06826 [cs.DC], 2018. [Online]. Available: https://arxiv.org/abs/1804.06826
2. Zhe Jia, Marco Maggioni, Jeffrey Smith, Daniele P. Scarpazza. Dissecting the NVIDIA Turing T4 GPU via Microbenchmarking. Technical Report, Citadel Enterprise Americas, LLC. arXiv:1903.07486 [cs.DC], 2019. [Online]. Available: https://arxiv.org/abs/1903.07486
3. Cristiano Malossi, et al. Characterizing the Performance and Scalability of Graphcore’s IPU Architecture via Microbenchmarking. arXiv:2104.07346 [cs.DC], 2021. [Online]. Available: https://arxiv.org/abs/2104.07346
4. Harris, M. (2012, July 2). Six ways to SAXPY. NVIDIA Technical Blog. https://developer.nvidia.com/blog/six-ways-saxpy/
5. Chen, S., Chou, T., Roy, A., Yu, H., Fang, Y., Wang, X., Yu, J., Liu, T. C. W., Zhuge, C., Fromm, J., Zhang, Y., Anil, R., & Mathews, A. (2025, Sept 5). Fast 2-Simplicial Attention: Hardware-Efficient Kernels in TLX. PyTorch Blog. _(Discusses TLX refactor of 2-Simplicial Attention with fused kernels, asymmetric windows, head-packing, and Hopper-specific scheduling, achieving up to 588 BF16 TFLOPs on H100.)_

[^1]: This is a performance bottleneck that occurs in multi-threaded programs when a CPU-intensive thread holds the Global Interpreter Lock (GIL) for an extended period, preventing other threads - including I/O-bound ones - from running. This effectively serializes execution and can cause application-wide delays, leading to unresponsiveness. 
[^2]: All-reduce is a fundamental collective communication operation used in distributed AI training, to efficiently synchronize and aggregate data (most commonly gradients), across multiple devices. The first phase **reduction phase** each device calculates its local contribution to the overall computation (e.g. gradients from a subset of data), and these are then combined "reduced" across all devices (via summation, averaging or another operation) to incorporate into a global result. The second phase is the **broadcast phase** where the aggregated global result is then distributed back to all participating devices, so every device has access to the same, synchronized information for subsequent steps. 
[^3]: The forward pass in AI is like an orchestra performing a symphony: each musician (neuron) plays their part from the sheet music (input data), and together they produce the music heard by the audience (the output prediction). The conductor compares what was played to the intended score (the ground truth), and the difference is the loss. In the backward pass, the conductor walks back through the sections, giving targeted feedback—“violins, soften here,” “trumpets, enter later”—so each player adjusts. These corrections propagate backward from the overall performance to individual musicians, refining the ensemble so that next time the music aligns more closely with the composer’s vision.
