---
title: "Real world ASI V: Applications"
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

## History of Transformers

The study of attention and Transformers is a journey from intuitive analogies (e.g. focusing on parts of the input like human attention) to sophisticated mathematical tools (e.g. softmax of dot products) to large-scale engineering (e.g. distributed training across GPUs). Mastering this will require an ability to fluidly move between these different levels of abstractionn, which is a challenging endeavour. However this will form a powerful foundation to deeply understand the backbone of almost every state-of-the-art AI system today, and the ability to push these frontiers even further - designing new model variants, and building infrastructure that serves a 100B parameter model to millions of users. 

The complete Transformer architecture consists of the following components:

1. **Positional Encoding:** Since self-attention on its on has no sense of token order, we must inject information about position. The original Transformer introduced _positional encodings_ which are fixed vectors that are added to the token embeddings at the bottom of the model to give each position a unique signal; Originally _sinusoidal positional embeddings_ were used.
2. **Encoder Layer:** 
    - _Multi-head Self-Attention (MHA):_ This takes the outputs from the previous layer (or the embedding/positional-encoding outputs for the first layer) and performs multi-head self-attention as described. Because it is an encoder, this attention is now over the entire sequence (no future masking), and the output is a set of context-mixed representations for each token. 
3. **Decoder Layer:** The decoder layer is similar to an encoder layer with one crucial addition, there are in total three sub-layers in the decoder layer of the transformer. 
4. **Layer Normalization and Residuals:**

## Self-attention in Transformers

> Self-attention is a mechanism that enables a model to weigh the importance of different elements within a single input sequence to better understand the context and relationships between them, regardless of their distance in the sequence.

The input to a self-atttention layer is a sequence of N items (e.g. words or tokens), each represented by a vector of dimension d (the embedding size, or model dimension). Because the inputs and outputs of a self-attention layer have the exact same shape, the layers can be stacked on top of each other (e.g. in a Transformer Encoder). The Tensor Shape that is being inputted is `(Sequence Length, Embedding Dimension)` or more commonly in the batch setting `(Batch Size, Sequence Length, Embedding Dimension)`. 

The output of the self-attention layer is a new sequence of N vectors, where each output vector is a _contextualized representation_ of the corresponding input item i.e. a list of new, context-aware word embeddings. It retains the exact same shape as the input. 

In practice, we represent data as triplets of query (Q), key (K) and value (V) vectors. The _query_ represents what content we are looking for, each _key_ represents what content a particular value contains, and each _value_ is the content to retrieve. The model learns projections to produce Q, K, V from inputs (e.g. the same input sequence for self-attention), and uses a scoring function between Q and K to decide how much of each value to include in the output. This mechanism was first popularized to help sequence models _attend_ to relevant parts of an input. 

Let us analyze the computational complexity at every step of this self-attention mechanism. 

1. Computing Q, K, V matrices - $O(N \cdot d^2)$
    - This involves multiplying the input matrix $X (N \times d)$ by the learned weight matrices $W^Q, W^K, W^V$ which are each of dimension $d \times d$. This gives us the Q, K, V matrices e.g. $Q = X \cdot W^Q$, each of which is a $N \times d$ matrix multiplied by a $d \times d$ matrix. The computational complexity of this is $O(N \cdot d \cdot d) = O(N \cdot d^2)$. 
2. (Bottleneck) Computing the attention score matrix $(Q \cdot K^T)$ - $O(N^2 \cdot d)$
    - This is the most computationally intensive step, where we multiply the Query matrix $Q(N \times d)$ by the transpose of the key matrix $K^T (d \times N)$. This gives us the scores $Q \cdot K^T$, and a resulting matrix shape of $N \times N$. The complexity of this step is $O(N \cdot d \cdot N) = O(N^2 \cdot d)$. 
3. Softmax computation - $O(N^2)$
    - We apply the softmax function as a row-wise operation on the $N \times N$ score matrix, for each of the $N$ rows, the operations scale with the length of the row $N$. The complexity of this is $O(N^2)$, which is much smaller than $O(N^2 \cdot d)$ and is typically ignored in the final complexity value. 
4. Computing the output matrix (attention weights multiplied by $V$) - $O(N^2 \cdot d)$ 
    - Finally, the normalized $N \times N$ attention weight matrix is multiplied by the value matrix $V$ ($N \times d$). The overall time complexity of this operation is $O(N \cdot N \cdot d) = O(N^2 \cdot d)$. 

Hence we see that the total computational complexity is the sum of $O(N \cdot d^2 + N^2 \cdot d + N^2 \cdot d)$. Note that $O(N^2 \cdot d)$ is usually the dominant factor (since N can become very very large), and cause the "quadratic complexity problem" for long sequences. 

## Expanding and Contracting in FFNs

> The **"expand and contract"** mechanism of the FFN allows us to take a rich, detailed summary of the input data, place it in a larger embedding/workspace to analyze it more thoroughly and give us more "room to think" in complex/non-linear ways, before contracting and distilling down to a smaller dimension to bring back our conclusions to a precise format. 

More concretely, the input vector $x$ to the FFN layer (with a dimension d e.g. 512) is multiplied by a large weight matrix $W_1$, transforming it into a much larger vector (with dimension $4d$ e.g. 2048), allowing us to use an expanded embedding space to discover rich, non-linear patterns and store meaningful representations. Typically a non-linear activation function (usually GELU or ReLU) _immediately after expansion_, to prevent the FFN (essentially a 2-layer MLP) to collapse into a monolithic linear transform. 

The second linear layer then contracts this expanded, processed vector back to the original dimension, forcing the network to distill the most important and useful information it found during the expansion phase. It ensures that the output vector fits the expected dimensions of the rest of the Transformer block (specifically the residual conneciton and the next layer). 

**ReLU:** Note that there are a plethora of activation functions that we could use to introduce non-linearities into our MLP, we mentioned ReLU (Rectified Linear Unit) which outputs the input directly if it is positive, otherwise it outputs zero. It is extremely fast to compute, and also sets negative values to zero, which makes the network sparse (fewer active neurons), leading to efficient computation. The problem with this is that if a neuron consistently receives negative inputs, its output is always zero, and gradients can no longer flow through it during training, which makes the neuron permanently inactive. Furthermore, the sharp "kink" at zero can sometimes lead to instability during training.

**GeLU:** The other activation function that has largely replaced ReLU is GeLU, which weights the input by its probability according to a Gaussian distribution, which applies a "probabilistic" form of gating. We apply the transformation $f(x) = x \cdot P(X \le x)$ where $P(X \le x)$ is the cumulative distribution function of the normal distribution. Note that this is smooth and continuous, and attenuates negative inputs but doesn't strictly zero them out like ReLU, which allows small negative values to pas through. This facilitates more stable training, and better performance - albeit, much more computational intensive. 

> There are **four** sets of learnable parameters per FFN layer; the first $W_1$ is an expand weight matrix learned during training, the second is the corresponding bias for the first layer $b_1$. The third $W_2$ is a contract weight matrix learned during training, and the corresponding learned bias for that layer $b_2$. These parameters are shared _across all token positions_ within a single FFN layer.

In practice, in the forward pass (during both training and inference), a batch of contextualized vectors (from the MHA layer) enters the FFN. The exact same learned parameters $(W_1, b_1, W_2, b_2)$ are applied independently to each token's vector in parallel. A new batch of processed vectors which has now been expanded, processed in the MLP/FFN latent space, and contracted back to the original input dimension, is now passed to the next layer via. a residual connection and layer normalization. Now in the backward pass (which occurs during only training), the model's error (loss) is calculated at the output layer and then backpropagated through the network. The gradients are computed for the learnable parameters $(W_1, b_1, W_2, b_2)$ and an optimizer (such as Adam or AdamW) uses these gradients to slightly adjust the numerical values of the four learned parameters, teaching our FFN how to better process information next time. 

## Multi-head Attention (MHA) in Transformers

![Alt text](/image-5.png)

At its core, an attention mechanism lets a model focus on the most relevant pieces of information by computing a _weighted sum_ of values, where the weights reflect the relevance of each value to a given query. This can be viewed as a "soft" form of information retrieval, each query looks up all keys and softly retrieves their associated values according to a similarity score. Multi-Head Attention is an extension of the self-attention mechanism of Transformers that performs multiple attention operations in parallel, allowing our model to jointly attent to information from different representation subspaces at different positions. 

> Multi-head attention (MHA) lends transformer-based AI models the ability to _jointly attend to information from across a sequence_. By running multiple attention heads in parallel, we are able to learn orthogonal and complementary representations, and create a more robust overall representation of the input data than a single attention head could.

Indeed, each attention head learns to focus on different aspects of data, for example one might focus on positional relationships, another on syntactic dependencies, and a third on semantic meaning; each head projects the inputs queries, keys and values in different subspaces, allowing it to capture unique patterns. 

_Multihead Attention (MHA) with Mixture of Experts (MoE)_

![Alt text](/multi-head-attention.png)

MHA is a core component for _how_ a Transformer attends to information, while MoE is an architectural modification (usually to the FFN layer) primarily aimed at improving the _efficiency and scalability_ of the model's overall capacity. [^1]

_How MHA works_ 

The mechanics of MHA is similar to that of self-attention, however for every input vector to the MHA layer of dimension $d$ (for example 512 dimensions), we decompose this vector into smaller sub-dimensions that are fed into each MHA head. For example, if we had $8$ heads, a $512$-dimension vector might be split into $8$ chumnks of $64$ dimensions each, creating $8$ separate sets of ${Q, K, V}$ matrices, each with its own independent, learned weight matrices $(W^Q, W^K, W^V)$. 

Each of the $8$ heads independently computes the standard _scaled dot-product attention_ (described in the self-attention section), and each head produces its own contextualized output for every token, based on its specific focus. The outputs from all 8 heads are then concatenated (joined back together end-to-end), which results in a matrix that has the full original dimension (e.g. $8x \times 64 = 512$ dimensions). The concatenated output is passed through a final, dense linear layer with a new learned weight matrix ($W^O$) which mixes the information gathered by all the different heads and projects it into the final output space that has the correct dimensiosn to be passed to the next layer of the Transformer. 

$$\textbf{MultiHeadAttention}(Q,K,V) = \textbf{Concat}(H_1, \cdots, H_h) \cdot W^O$$

Indeed, MHA works by creating multiple "subspaces" where attention can operate differently - allowing our model to capture richer and more diverse set of relationships within the data simultaneously, providing a much more robust and powerful contextual understanding than a single attention mechanism could achieve alone. Furthermore, we can parallelize the operations of each of these 8 independent heads which makes wall-clock time much faster compared to running a single, large 512-dimension attention operation sequentially. 

_Additive vs. Dot-product Attention_ 

### GPU Optimizations: Fusing kernels, Tiling, memory Access and Precision

> Every new idea in this space, be it FlashAttention, GQA, ALiBI, RoPE or block-sparse patterns, contributes to one of two goals: 1. Making attention faster/leaner or 2. Enabling attention to capture needed information more efficiently. 

Implementing MHA efficiently is critical for both training and inference of large-scale AI models. The MHA mechanism is often a major contributor to the overall computation and memory usage in large Transformer networks; indeed training time can become very long due to the quadratic cost complexity as models scale up in size. Naively computing MHA would involve multiple large matrix operations and memory-heavy steps: computing the $QK^T$ attention score matrix for each head (which for a sequence length $n$ is a huge $nxn$ matrix i.e. $O(n^2)$ operation), storing these scores, applying softmax, and then multiplying by the $V$ matrix. This could lead to significant memory bandwidth demand and under-utilization of compute units if not optimized. For example, the softmax computation is sequential, element-wise that does not fully leverage our GPU's parallel FMA units; and we will also need to read and write large intermediate matrices from high bandwidth memory (HBM). 

- **Fused Attention Kernels:** One common yet effective strategy is to fuse the multiple sub-operations of attention ($Q \cdot K^T$, softmax, and applying the attention weights to $V$) into a single GPU kernel. By doing this, the intermediate results (e.g. the attentions cores) need not be written out to global memory and read back in; instead they are kept in on-chip registers or shared memory and immediately used for the next step, greatly reducing memory traffic and latency. 

- **Leveraging Tensor Cores and Hardware Optimizations:** Modern GPUs excel at dense matrix operations (via their Tensor Cores). Optimized MHA implementations tile their computations to make full use of these units. The $Q \cdot K^T$ and attention-weighted $V$ multiplies are cast into forms that utilize these tensor operations at maximum throughput. Meanwhile, certain transformations (like scaling by $\frac{1}{\sqrt{d_k}}$ and softmax) might be merged or overlapped with these matrix operations so that the GPU spends more time doing high-throughput math and less time on memory moves or scalar operations. 

- **Memory and Cache Optimizations:** Because attention can be memory-bandwidth bound for large $n$, layout and caching optimizations are important. This includes using shared memory to cache chunks of K or V that are reused across threads, coallescing memory accesses for reading/writing Q, K, V and organizing threads to reduce redundant loads. In multi-head attention, there is also opportunity to parallelize across heads (since each head's computation is independent until the final concatenation), GPUs can dispatch threads or warps for different heads concurrently, as long as the kernel is designed to handle that. 

Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) is a powerful innovation where instead of each head having its own distinct Key and Value matrices, multiple heads share a single K/V (or a smaller set of K/V groups). This reduces the memory footprint and overhead of the attention mechanism (especially beneficial for inference where K/V from all the past tokens are cached), and whilst we sacrifice some flexibility, we gain _drastically lower memory usage_. For instance, if 8 heads share one key/value set, the size of the cache (and the cost of computing attention) drops roughly by a factor of 8.

#### FlashAttention

As discussed, the standard attention mechanism has two dominant bottlenecks:

1. **Memory $O(N^2)$:** This is computing the attention matrix $A = \textbf{softmax}(\frac{QK^T}{\sqrt{d}})$ requires storing an N by N matrix in GPU memory, where N is the sequence length.
    - This is _prohibitively large_ for long sequences (e.g. N=16k would lead to 256 _million_ elements in our matrix)
    - Even if compute on the static matrix is feasible, writing intermediate results (e.g. $QK^T$ and the attention scores) to and from our HBM will completely dominate runtime.
2. Even though the GPU has plenty of FLOPs (e.g. from Tensor Cores), the time is dominated by moving data (I/O) between HBM and on-chip SRAM/registers.

![Alt text](/image-6.png)

Before FlashAttention, there were attempts such as Triton-based kernels and newer cuDNN fused ops that improved on base-line naive full attention by fusing some steps of the operation, however FlashAttention established a new paradigm where we fuse _all_ the attention steps in one kernel, and reorganize computations to exploit the GPU memory hierarchy, yielding dramatic speedups without approximations.

From a kernel engineering standpoint, the FlashAttention is a streamining GEMM, softmax, and reduction fusion kernel. The key innovation is I/O-aware scheduling, where we minimize the read/write between DRAM, shared memory and registers, and maximize reuse inside the warp tiles. 

FlashAttention v1 was introduced to achive $O(N)$ memory usage instead of quadratic cost, and significantly higher speed by minimizing expensive HBM traffic. It does so by tiling the computation and working in on-chip SRAM (shared memory) as much as possible. The subsequent versions FlashAttention-2 (2023) and FlashAttention-3 (2024) build on this foundation, introducing better parallelism and leveraging new GPU features, respectively. Collectively, these methods have become the industry standard for high-performance attention, widely adopted in PyTorch's `scaled_dot_product_attention` and have enabled long-context LLMs.

#### FlashAttention v1

Version 1 rearranges the attention computation to avoid materializing the entire $N \times N$ attention matrix on slow HBM. The core idea is to use _tiling_ in the sequence dimension and perform softmax reduction incrementally, so that only smaller sub-matrices are handled at any given time in fast on-chip memory.

> The fundamental motivation of FlashAttention v1 was to reduce memory I/O even at the cost of extra compute. This bet paid off because GPUs have far higher FLOP capability than HBM bandwidth. The key result was to bring _memory complexity_ down from $O(N^2)$ to $O(N)$ with a tiling technique and online softmax.

All the steps, $Q \cdot K^T$, softmax, dropout, and $P V$ are **fused in a single CUDA kernel**, eliminating redundant memory reads/writes between steps. We break down the large matrix operations with tiling, and on-the-fly softmax normalization to avoid ever writing the full attention matrix to HBM. Queries Q are processed in blocks, and for each block Q, the kernel iterates over blocks K (and V), loading them into SRAM and computing partial scores $S = QK^T$ for that tile. These partial scores are exponentiated and accumulated into the softmax sum. Output aprtials ($P V$) are accumulated as well - by the end, the correct attention output is produced without storing intermediate $N \times N$ matrices. 

The algorithm proceeds as follows (for one attention head at a time, though batch and heads are parallelized as usual):

- _Tiled Forward Pass:_ Instead of computing $S = QK^T$ for all queries and keys at once, FlashAttention partitions the sequence into blocks (for example, blocks of 128 queries by 128 keys). It loops over key/value blocks for a given block of queries, loading a block of $Q$ and a block of $K$ (and corresponding $V$) from HBM into shared memory, computing the partial attention scores for that tile, and immediately applying Softmax normalization _within that tile_. Crucially, it keeps track of partial Softmax results so that after iterating over all key blocks, the final Softmax is correct as if done in one pass. 

> This choice of tile size (e.g. how many queries per block and how many keys per sub-block) is crucial. Larger tiles mean more reuse and fewer iterations (which improves compute efficiency), but they consume more shared memory and registers. 

#### FlashAttention v2

#### FlashAttention v3

FlashAttention v1, v2 and v3 demonstrate how algorithm and kernel co-design can yield massive performance gains for a critical operation like attention, by being mindful of GPU memory hierarchies and parallel execution capabilities. The key takeaways are:

1. Fuse operations to avoid memory bottlenecks
2. Tile data to maximize on-chip reuse
3. Parallelize across all available dimensions
4. Overlap different computations to utilize all hardware units

By following these principles and adjusting to the specifics of the GPU architecture, one can achieve performance that approaches the physical limits of the device, enabling training and inference of long-context LLMs at feasible speeds. 

[^1]: Though there have been recent efforts to combine the two ideas, e.g. "Mixture-of-Head Attention" (MoH) where attention heads themselves are treated as experts and are sparsely activated.

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 5998–6008.
2. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. Advances in Neural Information Processing Systems (NeurIPS 2022)