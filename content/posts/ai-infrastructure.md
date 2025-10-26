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

<!-- Enable blockquotes -->
<style>
blockquote {
  font-style: italic;
  color: #444;
  border-left: 4px solid #aaa;
  margin: 1em 0;
  padding: 0.5em 1em;
  background: #f9f9f9;
}
</style>

GPUs and custom AI hardware are the engines driving progress in AI research and applications. This of a CPU like a sports car that often has better single performance statistics than a GPU, which is more like a bus (there are a lot more seats, but it takes a longer time to get from A to B). Indeed, it often takes sometimes orders of magnitude less time to do an L2 access in a CPU than a GPU, or an HBM DRAM access; the key however is that GPUs can take advantage of _enormous_ parallelism and throughput despite their lower single-use latency. This parallelism arises from the many "SM" cores that consist of multiple arithmetic logical units (INT32, FP32, FP64 ALUs), special function units, and memory (registers, shared memory) - but the true star here in the GPU is the "tensor core" which is heavily optimized for GEMM accelerations, giving us incredible FLOP performance for deep learning matrix multiplications.

![Alt text](/image-8.png)

> An SM executes threads in groups called **warps**. A warp is a group of typically 32 threads, and they execute the same instructions in the same order in a _SIMT (single instruction multiple thread)_ fashion, on different data - similar to the SIMD execution model. At any given time, the SM may execute one wart, switch between multiple warps or stall if there are dependencies or resource constraints. A single SM can manage multiple warps concurrently depending on their resource capacity. 

To understand the relative latencies of memory operations in the SM (global memory access, L1/2 cache access, thread register accesses) compared with compute operations such as FMA or tensor core operations (orders of magnitude faster), the EleutherAI has an extraordinarily educational exposition on this topic. 

![Alt text](/image-9.png)

But raw hardware alone doesn’t explain why some teams ship reliable systems at scale while others struggle with bottlenecks and cost explosions. The difference comes down to how you think about optimization.

A useful way to frame it is as **three nested loops** wrapped around every model. Each loop sits at a different layer of abstraction, and each has its own gauges, and levers. The innermost loop is about the raw efficiency of a single device: is the GPU actually doing useful math, or is it starved by memory bandwidth or kernel overhead? The next loop out is the training loop: how do we scale beyond one GPU and make hundreds or thousands of them behave like a single machine? And finally, there is the product loop: once the model is trained, can we serve it reliably under real-world traffic, meet our latency and safety SLOs, and do so at a cost per token that won't sink the business? 

What makes this framing powerful is that most "mysterious" performance or reliability problems stop being mysterious once you identify which loop you are really in. If a profiler shows long gaps on the GPU timeline, that is an inner loop problem. If throughput collapses when yous cale from one rack to three, that is the training loop. If P95 latency is spiky even though single-GPU kernels are fast, you are in the product loop. The moment you can name the loop, you know where to look and which tools to bring to bear. 

We will begin the exposition on the three loops first with an overview of the highly valuable _Amdahl's Law_. 

> **Amdahl's Law:** This gives us a theoretical limit of speedup from parallelization or acceleration. $$S_{max} = \frac{1}{(1 - P) + \frac{P}{N}}$$ where $P$ is the fraction of work that _can_ be parallielized, $(1 - P)$ is the fraction that is inherently serial, and $N$ is the speedup factor (or number of parallel units) present in our processor. As $N$ tends to infinity, the best speedup that we can ever get is $\frac{1}{1-P}$, which means that even if 99 percent of our workload parallelizes perfectly, the remaining 1 percent caps the total speedup to $100x$, no matter how many GPUs that we throw at it. 

When we train or serve a large model, every step involves a _graph_ of kernels: matrix multiplies, layernorms, activation functions, attention, all-reduce ops etc. Each kernel has different scaling characteristics:

| Kernel type                  | Nature                                    | Example                  | Parallel fraction (P)            |
| ---------------------------- | ----------------------------------------- | ------------------------ | -------------------------------- |
| Dense GEMM / Tensor Core ops | Highly parallel                           | `matmul`, convolution    | ~0.999                           |
| Reduction / softmax          | Partly parallel, serial across dimensions | `reduce_sum`, `softmax`  | ~0.9                             |
| Control & memory kernels     | Poorly parallel                           | indexing, scatter/gather | ~0.5                             |
| Communication                | Serially coupled                          | all-reduce, all-gather   | ~0.7–0.95 depending on bandwidth |

When we optimize or scale a system, we are not accelerating the whole workload, but only parts of it e.g. speeding up GEMMs with Tensor Cores or fusing several small ops into one. With Amdahl's law, we are reminded that the _non-accelerated and parallelized_ portions dominate once the fast parts are already fast. Indeed suppose that our LLM training consists of $80$ percent GEMM (which runs on Tensor Cores), $10$ percent communication (all-reduce), and 10 percent everything else (elementwise, indexing, overhead). Now say that we double Tensor Core throughput (2x faster GEMMs), then our total speedup, via Amdahl's law, is actually on 1.67. The takeaway is that even enormouse hardware speedups yield modest end-to-end gains if the remaining kernels or communication cannot keep up.

Within a GPU (micro-scale) at the kernel-level, fusing bias and activation and dropout eliminates kernel launch overhead and redundant memory reads, improving the _serial fraction_ between kernels. Over time, this matters more than another 10 percen tin GEMM throughput. Across GPUs (meso-scale) for distribuied training or inference, compute scales well (GEMM), but communication (all-reduce) adds serial or latency-bound segments. No matter how many GPUs we add, the network-bound fraction sets the Amdahl ceiling. 

Across infrastructure (macro-scale) for the full-pipeline, preprocessing, I/O, scheduling, logging will dominate the performance hits - the "outer loop" serial components, even once we have beautiful orchestration over beastly processors. Modern compiler and infrastructure stacks (Triton, CUTLASS, XLA, PyTorch 2.0) fight Amdahl's ceiling in three key ways:

1. **Fusion:** Converting multiple small kernels into one large kernel to reduce launch overhead (reducing the serial fraction).
2. **Asynchronous execution:** Overlapping compute and data movement (hides the serial part under parallel compute)
3. **Rerouting:** Moving serial or low-intensity tasks onto faster subsystems (e.g. FP8 Tensor Cores, NVLink all-reduce engines)
4. **Pipeline parallelism:** Converting formerly sequential layer execution into overlapping stages (which reduces the effective serial region)

All of these strategies reduces or hides the non-parallel/serial part of the workload, which is the primary way we can combat Amdahl's ceiling. 

## The Inner Loop: making a single GPU sing

At the very center of AI infrastructure is the question: **is your GPU doing useful math, or is it wasting cycles?** Every model, no matter how large, eventually boils down to kernels running on a single device. If those kernels are inefficient, no amount of distributed wizardry will save you - because inefficiency just multiplies as you scale. The innermost loop is about runtime and kernel performance: understanding where the cycles are going, and how to keep the accelerator fed with a steady diet of compute. 

The mental model here is the _roofline_. Imagine a chart with two axes: arithmetic intensity (FLOPs per byte of memory moved) and achievable performance as a fraction of the device's peak. Your kernels live somewhere on this curve. If they are stuck in the low-intensity, memory-bound region, HBM-bandwidth - not Tensor Cores - is your bottleneck. If they are compute bound, the question becomes whether you are actually saturating those Tensor Cores. And if they are nowhere near the roofline at all, overheads are eating you alive - kernel launch latency, Python-GIL stalls [^1], or CPU-GPU syncs that keep the device idle. 

**Profiling is your compass:** Tools like Nsight Systems or PyTorch's built-in profilers don't just tell you how long each op takes - they show whether your GPU is mostly waiting on memory, struggling with occupancy, or simply not being kept busy. A typical training run might reveal long gaps between kernels, pointing to launch overhead. Or you might see HBM pegged at 90% while tensor utilization hovers at 20%, a classic sign that you are memory bound. 

This isn't just hypothetical. Microbenchmark studies on Volta and Turing GPUs have shown just how sharp the drop-offs can be. On Volta, for example, the L1 cache hit latency was measured at only 28 cycles, compared to 84 cycles on Pascal - nearly a 3 times improvement, but only if your working set fits. On Turing's T4, researchers found that vectorized 128-bit global memory instructions doubled the throughput of a simple SAXPY kernel compared to NVIDIA's own cuBLAS implementation that used 64-bit loads. The point is that the line between compute- and memory-bound is not theoretical: it shows up directly in latency histograms and throughput counters, and it shifts with every architectural generation. 

When _memory is the problem_, the fix is often algorithmic. Attention kernels are a perfect example. Naive implementations read and write enormouse intermediate matrices, swamping HBM. FlashAttention-style kernels, by contrast, restructure the computation to minimize memory traffic, moving closer to the compute-bound region of the roofline. Similarly, fusing QKV projections or using block-sparse attention can trade redundant memory ops for denser math. 

When _overhead is the problem_, you need to reduce the number of trips between host and device. Every kernel launch carries a cost, and if your training loop is littered with tiny ops, that cost adds up. The solution is fusion: combining bias, activation, and dropout in a single kernel, or capturing whole decode loops with CUDA Graphs so they run as a single replayable unit. The mantra is **"fewer, fatter kernels."**

And when the GPU seems underfed, the problem often lies outside the device. Dataloaders stuck behind the Python GIL, disk I/O that cannot keep up with training, or underpinned host memory can all cause the device to starve. In those cases, the fix is not in CUDA at all but how you structure the pipeline feeding the accelerator: double-buffering, pagelocked transfers, and async prefetch all matter here. 

Finally, there are subtle low-level effects taht the microbenchmark papers document (Jia et al.) and that few developers think about. Volta and Turing both use two 64-bit wide register banks, which means that certain three-operand instructions like FFMA can still suffer from bank conflicts if all sources happen to map to the same bank. Researchers showed that remapping registers to avoid these conflicts yielded 15% throughput improvements. Similarly, the T4's "uniform datapath" for scalar integer ops was introduced precisely to prevent loop counters and address arithmetic from polluting the main floating-point pipelines. These are the kinds of details that explain why a kernel doesn't hit its theoretical throughput even when you've done everything else "right". 

> **In short**, _the inner loop is about **discipline at the kernel level**._ 

It is the layer where hardware meets math. Get this wrong, and your scaling experiments or serving infrastructure will always be operating at a handicap. Get it right, and you establish the foundation on which the outer loops - distributed training and serving - can actually pay off. 

### Memory hierarchy (L1/L2/TLB): why kernels stall

Modern accelerators live and die by their memory systems. The same kernel can look _compute-bound_ on one device and _memory-bound_ on another simply because L1/L2/DRAM behavior shifts the roofline under your feet. We will build a mental model of the GPU memory stack, and what to do when profiles show that our math engines are waiting. 

### Case study: 2-simplicial attention meets the roofline (TLX on H100)

As a concrete example of inner-loop optimization paying real dividends, the PyTorch team recentlyd etailed a fused TLX (Triton Low-level Extensions) kernel for **2-Simplicial Attention** - an attention variant that models trilinear interactions among token triples. 

### Case study: From roofline theory to real tensor-core kernels

Every GPU (and indeed AI processor) kernel lives between two limits:  _how fast it can perform math_ and _how fast it can move data_. The first is set by the peak floating-point throughput of the hardware -- usually measured in FLOP/s (denoted $\tau$). The second is set by the bandwidth of the memory system i.e. the number of bytes per second that can reach the compute units (denoted $\beta$); these two parameters are key components to consider in the **roofline model** of processor performance.

> The roofline model is a performance model seeking to give the limitations of a specific hardware component in terms of algorithm performance. It is employed visually as a log-log plot of _Arithmetic Intensity_ versus _Flops/s_

Recall that FLOPS/s is a simple measurement that indicates the number of mathematical operations that the computer does per second on a given algorithm; usually dependent on the hardware, algorithm, and implementation. The independent variable in the roofline model is _arithmetic intensity_, which is the measure of how many operations are done per bytes loaded or stored from memory. 

$$\textbf{AI} = \frac{\textbf{Flops/s}}{\textbf{Memory Traffic}} = \frac{\textbf{flops/second}}{\textbf{bytes/second}} = \frac{\textbf{flops}}{\textbf{bytes}}$$

Because memory movement operations are extraordinarily slow in all modern-day hardware (compared to <1 clock cycle latency of math operations, which can be reduced by fusing operations into for example FMAC), it is in our best interest to maximize arithmetic intensity - hence for every memory movement operation, we have more math being performed, giving us more "bang for our buck". Indeed, the deeper our memory accesses need to go (e.g. registers, L1 cache, L2 cache, L3 cache, main memory), the slower and longer these take. 

> **cuBLAS** (CUDA basic linear algebra subprograms) is NVIDIA's high-performance library that implements that standard BLAS routines for GPUs. It is a GPU-accelerated equivalent of the long-established CPU BLAS libraries. BLAS defines a set of common, reusable linear-algebra building blocks. _Level 1_ are vector-vector operations (e.g. AXPY `y = ax + y`, DOT, SCAL), _Level 2_ are matrix-vector operations (e.g. GEMV `y = Ax + y`) and _Level 3_ are matrix-matrix operations (e.g. GEMM `C = A B + C`). cuBLAS provides GPU implementations of all these routines in single-, double-, and complex-precision flavours. 

Under the hood, cuBLAS launches highly optimized CUDA kernels written and tuned by NVIDIA kernel engineers. It chooses kernels dynamically depending on the GPU architecture (e.g. Volta, Turing, Ampere, Hopper, Blackwell), and uses **tensor cores**, shared-memory tiling, double buffering, and vectorized loads/stores for maximum throughput. It handles details such as memory coalescing, alignment, and fused multiply-adds so that the code achieves near-peak FLOP/s without manual PTX work. 

Since linear-algebra operations dominate most of AI, graphics, and scientific workloads, by off-loading these primitives to cuBLAS, developers can 1. Get close to hardware peak performance out of the box 2. Avoid reinventing and tuning kernslf for each GPU generation 3. Rely on a consistent API that mirrors CPU BLAS, easing porting from existing code. For example, we can use the following function call for a single-precision [^4] saxpy: 

```cpp
// Single-precision saxpy: y = alpha * x + y
float alpha = 2.0f;
cublasHandle_t handle;
cublasCreate(&handle);
cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1);
```

Here `d_x` and `d_y` are pointers to vectors in GPU memory. cuBLAS handles the kernel launch, memory scheduling, and synchronization. In the papers dissecting the [volta](https://arxiv.org/pdf/1804.06826) and [turing](https://arxiv.org/pdf/1903.07486) GPUs via benchmarking, the authors demonstrate that even though cuBLAS is NVIDIA's GPU-optimized, production-grade linear-algebra library that most frameworks (PyTorch, TensorFlow etc.) use under the hood for all GEMMs and related ops, we can still hand optimize lets say an single-precision AXPY (e.g. with 128-bit vectorized loads/stores [^5]) with knowledge of the underlying architecture to push even a mature library like cuBLAS closer to the theoretical bandwith roofline.  

### A short primer on Tensor Cores

> **Tensor cores** perform mixed-precision arithmetic. They are hardware blocks that accelerate matrix multiplies (and MMA), exploiting spatial/temporal reuse out of the data that allow us to bridge the memory wall, exploiting nice algorithmic properties about MMA and thereby increasing arithmetic intensity. 

| **Aspect** | **GEMM (General Matrix Multiplication)** | **MMA (Matrix Multiply-Accumulate)** |
|-------------|------------------------------------------|--------------------------------------|
| **Definition** | A full matrix multiplication and accumulation operation defined as 𝐶 ← α𝐴𝐵 + β𝐶, where **A**, **B**, and **C** are matrices and α, β are scalars. | A specialized, low-level hardware instruction that performs a small matrix multiplication and accumulation (𝐷 = 𝐴 × 𝐵 + 𝐶). |
| **Execution Level** | A high-level software algorithm, often implemented by libraries like NVIDIA’s **cuBLAS**, that coordinates the entire matrix operation. | A primitive hardware instruction executed by specialized units such as NVIDIA’s **Tensor Cores**. |
| **Hierarchy** | A complete GEMM is broken down into many smaller MMA operations executed in parallel across the GPU. | A single MMA is one small component (“tile” or “fragment”) of the full GEMM. |
| **Parallelism** | The GEMM operation is distributed across all Streaming Multiprocessors (SMs). Each SM runs multiple thread blocks, each performing many MMAs. | One MMA instruction is executed by a single warp of 32 threads within an SM. |
| **Abstraction** | Abstracted from hardware: a user calls a GEMM routine without managing how the GPU performs the multiplication. | A low-level, explicit interface for specialized hardware. Developers can issue MMA instructions directly for fine-grained control. |
| **Performance** | Measured in total throughput (TFLOPs) for the entire operation. Often bottlenecked by memory bandwidth on small matrices. | Extremely high throughput for small, low-precision matrix tiles. Designed to be compute-bound rather than memory-bound. |
| **Hardware** | Runs on general-purpose units (CUDA Cores), but is dramatically faster when offloaded to Tensor Cores. | Executes only on specialized hardware such as NVIDIA Tensor Cores or Google TPUs. |

As mentioned thematically, all of artifiical intelligence and high performance computing workloads essentially boil down to linear algebra, and particularly matrix-multiple-accumulate (GEMM) [^6]; every GPU instruction should be computing $$C = A \times B + C$$ but on huge matrices, billions of times per second. Before Tensor Cores existed, GPUs used **CUDA cores** (which are 'scalar' or 'vector' ALUs) to do these operations, though rather inefficiently. We eventually realizes that the whilst the math throughput grew rapidly (due to more CUDA cores), the memory bandwidth could no longer keep up, so GPUs spent cycles feeding operands instead of doing math; hence a dedicated hardware unit that performed small matrix multiplication directly, with less data motion was introduced i.e. the Tensor Core. 

| Feature                   | **CUDA Core**                 | **Tensor Core**                                        |
| ------------------------- | ----------------------------- | ------------------------------------------------------ |
| **Type of unit**          | Scalar/vector ALU             | Matrix math accelerator                                |
| **Operation**             | One multiply or add per cycle | Many (hundreds) of multiply-accumulate per cycle       |
| **Data granularity**      | Operates on individual floats | Operates on small matrices (e.g., 4×4, 16×8×8)         |
| **Precision focus**       | FP32, FP64, Int32             | FP16, BF16, TF32, INT8, FP8                            |
| **Programming interface** | Implicit via CUDA C/C++       | Explicit via WMMA / cuBLAS / CUTLASS                   |
| **Goal**                  | General-purpose compute       | Dense linear algebra (esp. GEMM)                       |
| **Performance**           | High flexibility              | High throughput (10–100× higher TFLOPs for matrix ops) |

Ultimately, Tensor Cores are _specialized, fixed-function units_ built inside the GPU's Streaming Multiprocessor (SMs), where each one executes a _matrix fused multiply-add_ (MMA) on a small tile of operands - on Volta/Turing this is defined as $16 * 8 * 8$. This tile size is the shape of the micro-matrix each Tensor Core computes per instruction i.e. we a multiply a $16 * 8$ tile of matrix A by an $8 * 8$ tile of matrix B, and accumulate the result into a $16 * 8$ tile of C. Each instruction `mma.sync.m16n8k8` performs 128 multiply-add operations per thread group, or 256 FLOPs. When thousands of threads across warps execute these in parallel, the throughput explodes. 

By fusing 128+ multiply-add operations into one instruction and keepign operands local in registers/shared memory, Tensor Cores massively raise FLOPs without increasing memory traffic. This is why every AI workload (transformers, CNNs, diffusion models) now run primarily on Tensor Cores. 

| Operation         | FLOPs per memory access | Approx throughput   |
| ----------------- | ----------------------- | ------------------- |
| FP32 CUDA cores   | 1 FLOP / 4 bytes        | ~15 TFLOPs (A100)   |
| FP16 Tensor Cores | 256 FLOPs / 16 bytes    | ~300+ TFLOPs (A100) |

In practice, we do not typically program Tensor Cores directly in assembly (PTX, SASS) and instead we use the WMMA API `nvcuda::wmma` which is a CUDA C++ intrinsics that map to Tensor Core instructions. 

```cpp
wmma::fragment<matrix_a, 16, 16, 16, half> a_frag;
wmma::fragment<matrix_b, 16, 16, 16, half> b_frag;
wmma::fragment<accumulator, 16, 16, 16, float> c_frag;
wmma::load_matrix_sync(a_frag, A, lda);
wmma::load_matrix_sync(b_frag, B, ldb);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(C, c_frag, ldc);
```

Each SM contains dozens of CUDA cores for general math and several Tensor Cores for block matrix operations. Tensor Cores handle the dense GEMM work, whereas CUDA cores handle everything else - vector ops, address arithmetic, activation functions etc. Indeed, most of the GPU's advertised "AI perforamnce" comes almost entirely from their Tensor Cores. CUTLASS is the bridge that allows us a way to explicitly program Tensor Cores for custom kernels, without writing raw PTX. 

### Accelerating GEMMs with sparsity, interleaved DSRs and SIMD-32

![Alt text](/image-4.png)

> `FMACH dst = src0 + src1 * src2` is the canonical instruction underpinning GEMMS.

To increase arithmetic intensity

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

#### ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

In vanilla data parallel, every GPU keeps full copies of parameters, gradients, and optimizer states (e.g. Adam's FP32 master weights and moments). This wastes memory N times over N GPUs. For mixed-precision training with Addam, this amounts to 16 bytes per parameter (2B weights, 2B gradients, 12B optimizer states), so 1 trillion parameters is 16 TB just for model states (not even taking into account CUDA kernel overhead etc.), which is far beyond a single GPU. 

> The key idea behond **ZeRO (Zero Redundancy)** is instead of replicating, we shard the model states (optimizer states, gradients, parameters) across data-parallel ranks and assemble them just-in-time with efficient collectives, _exploiting the fundamentally temporal nature of training models_. We keep only our slice/layer of parameters most of the time, and briefly all-gather when we need the rest, then discard. 

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

**Pipeline bubbles (hidden idle time)** Pipeline parallelism looks elegant on paper - different stages working in lockstep. In practice, the edges fray. When microbatch count is too low, early stages sit idle while later ones drain; when it's too high, memory usage balloons. The fix is a careful balance: interleaving virtual stages to shrink bubbles, tuning microbatch counts so bubbles amortize without tipping memory over the edge, and splitting "whale" layers that hog an entire stage. Teams often discover that pipeline tuning is less about abstract math and more about artisanal craft, guided by per-stage FLOP profiles. 

**MoE instability and tail experts:** Sparse models promise trillion-parameter capacity, but they come with their own demons. In early training, the router may overload a handful of experts while others sit idle, leading to both instability and wasted compute. Tokens pile up at _"tail" experts, spilling capacity and spiking latency. The usual antidotes are auxiliary losses that encourage balance, capacity factors that cap per-expert load, and token drop strategies when overload can't be avoided. But the deeper truth is that MoE is hypersensitive to parallelism topology: if your all-to-all dispatch spans slow links, imbalance gets amplified. Co-locating expert groups within fast interconnect domains is as important as the algorithmic tweaks. 

## The Product Loop: serving and inference engines

> How do we turn a trained model into a fast, scalable service?

This section breaks down a modern LLM inference engine through the lens of vLLM - paged attention, continuous batching, prefix caching, speculative decoding, and how these pieces scale from a single GPU to multi-node serving with load balancing and autoscaling. It closes with the metrics that matter (TTFT/TIL/throughput) and how to tune for them. 

The inference engine core consists of three main components - a scheduler, a KV-cache manager, and a model executor. Requests arrive as tokenized prompts, and are packed into a continuous batch. The scheduler decies whether each step will handle prefill (processing the full prompt) or decode (processing only the newest token), and mixes them when possible. 

The KV-cache manager assigns fixed-size memory blocks that store key/value vectors for attention, which lets the system grow sequences without right-padding and reuse memory efficiently. Continuous batching allows new requests to join at every step, rather than waiting for the next global batch, and prefix caching avoids recomputing shared tokens across requests with the same prefix. Together, these components ensure high GPU utilization while keeping per-request latency bounded, forming the foundation that later features and scaling layers build on. 

Every inference request alternates between two workloads - prefill and decode. 

1. **Prefill** is the initial forward pass over all prompt tokens. It is compute-bound i.e. each layer must process the entire prompt, and the cost grows linearly with prompt length. At the end of prefill, the engine produces logits for the final position and samples the first output token. 
2. **Decode** is the steady-state loop that follows. Each step only process the most recent token, since all previous key/value vectors are already in the cache. This makes decode much lighter in FLOP,s but it is memory-bandwidth-bound: the model weights and cached KVs still need to be ready every step just to produce one new token.

Serving systems must juggle both at once. Long prefills are bursty and can monopolize GPU time if not chunked, while decodes are latency-sensitive and dominate user experience. Modern schedulers therefore prioritize decodes (to minimize time-to-next-token), and use chunked prefill or disaggregated prefill/decode execution to keep throughput high without letting single long prompts stall the system. 

## Closing Remarks

AI infrastructure is diagnosis first, optimization second. In the **inner loop**, we should decide if we're compute, memory, or overhead bound and act accordingly. In the **training loop**, we choose the right parallelism/memory/communication mix and make failure routine.

### References
- Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. ZeRO: Memory Optimizations Towards Training Trillion Parameter Models. 2020
- 
- Zhe Jia, Marco Maggioni, Benjamin Staiger, Daniele P. Scarpazza. Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking. Technical Report, Citadel Enterprise Americas, LLC. arXiv:1804.06826 [cs.DC], 2018. [Online]. Available: https://arxiv.org/abs/1804.06826
- Zhe Jia, Marco Maggioni, Jeffrey Smith, Daniele P. Scarpazza. Dissecting the NVIDIA Turing T4 GPU via Microbenchmarking. Technical Report, Citadel Enterprise Americas, LLC. arXiv:1903.07486 [cs.DC], 2019. [Online]. Available: https://arxiv.org/abs/1903.07486
- Cristiano Malossi, et al. Characterizing the Performance and Scalability of Graphcore’s IPU Architecture via Microbenchmarking. arXiv:2104.07346 [cs.DC], 2021. [Online]. Available: https://arxiv.org/abs/2104.07346
- Harris, M. (2012, July 2). Six ways to SAXPY. NVIDIA Technical Blog. https://developer.nvidia.com/blog/six-ways-saxpy/
- Chen, S., Chou, T., Roy, A., Yu, H., Fang, Y., Wang, X., Yu, J., Liu, T. C. W., Zhuge, C., Fromm, J., Zhang, Y., Anil, R., & Mathews, A. (2025, Sept 5). Fast 2-Simplicial Attention: Hardware-Efficient Kernels in TLX. PyTorch Blog. _(Discusses TLX refactor of 2-Simplicial Attention with fused kernels, asymmetric windows, head-packing, and Hopper-specific scheduling, achieving up to 588 BF16 TFLOPs on H100.)_

[^1]: This is a performance bottleneck that occurs in multi-threaded programs when a CPU-intensive thread holds the Global Interpreter Lock (GIL) for an extended period, preventing other threads - including I/O-bound ones - from running. This effectively serializes execution and can cause application-wide delays, leading to unresponsiveness. 
[^2]: All-reduce is a fundamental collective communication operation used in distributed AI training, to efficiently synchronize and aggregate data (most commonly gradients), across multiple devices. The first phase **reduction phase** each device calculates its local contribution to the overall computation (e.g. gradients from a subset of data), and these are then combined "reduced" across all devices (via summation, averaging or another operation) to incorporate into a global result. The second phase is the **broadcast phase** where the aggregated global result is then distributed back to all participating devices, so every device has access to the same, synchronized information for subsequent steps. 
[^3]: The forward pass in AI is like an orchestra performing a symphony: each musician (neuron) plays their part from the sheet music (input data), and together they produce the music heard by the audience (the output prediction). The conductor compares what was played to the intended score (the ground truth), and the difference is the loss. In the backward pass, the conductor walks back through the sections, giving targeted feedback—“violins, soften here,” “trumpets, enter later”—so each player adjusts. These corrections propagate backward from the overall performance to individual musicians, refining the ensemble so that next time the music aligns more closely with the composer’s vision.
[^4]: Single-precision refers to FP32 floating-point format, where we use one-bit for the sign, eight bits for the exponent and 23 bits for the fraction (standard format for many applications, offers a balance of speed and accuracy and is often considered the full precision). Half-precision refers to FP16 floating-point format where we use 10 bits for the mantissa and 5 bits for the exponent, and 1 bit for the sign (uses less memory and allows for faster computation but less accuracy). Double-precision refers to FP64 floating-point, where we use 52 bits for the mantissa and 11 bits for the exponent (provides the highest level of accuracy and range, used for applications where rounding errors cannot be tolerated). Note that higher precision, whilst increasing memory usage (limiting for large models and batch sizes), can lead to greater numerical stability by reducing the risk of overflow and underflow. 
[^5]: GPUs spend a lot of time moving data between global memory (HBM) and compute units. Every time a thread executes a load `ld.global` or store `st.global` instruction, it requests some bytes from memory. Each memory request has a fixed latency (e.g. 400-800 ns to reach DRAM and come back) and a fixed overhead in the memory controller to schedule and acknowledge it. So if each thread requests tiny pieces of data at a time (e.g. 4 bytes at a time for one `float`), the GPU wastes time handling a flood of small requests. The latency is the same whether we ask for 4 bytes or 16 bytes, so we want each request to carry as much useful data as possible (hence vectorized memory accesses). 
[^6]: It is important to clarify the distinction between GEMM and MMA, general matrix multiplication and matrix multiply-accumulate. GEMM is a high-level, _software_ defined operation, while MMA Is a low-level, specialized hardware instruction. MMA instructions are the building blocks that enable Tensor Cores to dramatically accelerate GEMM operations. An efficient GEMM operation on a modern GPU is a pipeline that works by 1. **Data loading** (input matrices A and B are divided into tiles and loaded from global memory into the faster shared memory) 2. **MMA execution** (each thread block takes a tile from shared memory, and within the thread block, a warp of threads performs a series of MMA instructions on sub-tiles/fragments of the data) 3. **Accumulation** (the MMA instructions compute the product and accumulate the result in registers. This is much faster than repeatedly reading from and writing to memory) 4. **Epilogue** (after all the MMA operations are complete, the final, combined result is written from the registers back to global memory). By pipelining these different parts of the GEMM life-cycle/MMA building block components, we can achieve fast and accurate GEMM. 
