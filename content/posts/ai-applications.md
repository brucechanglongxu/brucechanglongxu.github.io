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
4. **Layer Normalization and Residuals:** When we stack many layers of nonlinear functions (e.g. self-attention and feedforward blocks) the distribution of activations (means, variances) tend to drift as it passes through layers. This causes vanishing / exploding gradients (backward signals get too small or too large), and internal covariate shift (each layer sees changing input statistics during training, making learning unstable). We want to keep these activations in a "healthy range" i.e. roughly zero-centered and unit-scaled so gradients flow smoothly through hundreds of layers. 

## Self-attention in Transformers

> Self-attention is a mechanism that enables a model to weigh the importance of different elements within a single input sequence to better understand the context and relationships between them, regardless of their distance in the sequence.

The input to a self-atttention layer is a sequence of N items (e.g. words or tokens), each represented by a vector of dimension d (the embedding size, or model dimension). Because the inputs and outputs of a self-attention layer have the exact same shape, the layers can be stacked on top of each other (e.g. in a Transformer Encoder). The Tensor Shape that is being inputted is `(Sequence Length, Embedding Dimension)` or more commonly in the batch setting `(Batch Size, Sequence Length, Embedding Dimension)`. 

The output of the self-attention layer is a new sequence of N vectors, where each output vector is a _contextualized representation_ of the corresponding input item i.e. a list of new, context-aware word embeddings. It retains the exact same shape as the input. 

In practice, we represent data as triplets of query (Q), key (K) and value (V) vectors. The _query_ represents what content we are looking for, each _key_ represents what content a particular value contains, and each _value_ is the content to retrieve. The model learns projections to produce Q, K, V from inputs (e.g. the same input sequence for self-attention), and uses a scoring function between Q and K to decide how much of each value to include in the output. This mechanism was first popularized to help sequence models _attend_ to relevant parts of an input. 

Let us analyze the computational complexity at every step of this self-attention mechanism. 

1. Computing Q, K, V matrices - $O(N \cdot d^2)$
    - This involves multiplying the input matrix $X (N \times d)$ by the learned weight matrices $W^Q, W^K, W^V$ which are each of dimension $d \times d$. This gives us the Q, K, V matrices e.g. $Q = X \cdot W^Q$, each of which is a $N \times d$ matrix multiplied by a $d \times d$ matrix. The computational complexity of this is $O(N \cdot d \cdot d) = O(N \cdot d^2)$. 
2. (Bottleneck) Computing the attention score matrix $(Q \cdot K^T)$ - $O(N^2 \cdot d)$ [^8]
    - This is the most computationally intensive step, where we multiply the Query matrix $Q(N \times d)$ by the transpose of the key matrix $K^T (d \times N)$. This gives us the scores $Q \cdot K^T$, and a resulting matrix shape of $N \times N$. The complexity of this step is $O(N \cdot d \cdot N) = O(N^2 \cdot d)$. 
3. Softmax computation - $O(N^2)$
    - We apply the softmax function as a row-wise operation on the $N \times N$ score matrix, for each of the $N$ rows, the operations scale with the length of the row $N$. The complexity of this is $O(N^2)$, which is much smaller than $O(N^2 \cdot d)$ and is typically ignored in the final complexity value. 
4. Computing the output matrix (attention weights multiplied by $V$) - $O(N^2 \cdot d)$ 
    - Finally, the normalized $N \times N$ attention weight matrix is multiplied by the value matrix $V$ ($N \times d$). The overall time complexity of this operation is $O(N \cdot N \cdot d) = O(N^2 \cdot d)$. 

Hence we see that the total computational complexity is the sum of $O(N \cdot d^2 + N^2 \cdot d + N^2 \cdot d)$. Note that $O(N^2 \cdot d)$ is usually the dominant factor (since N can become very very large), and cause the "quadratic complexity problem" for long sequences. 

_Layer Normalization and Residuals_ 

As more layers of nonlinear functions are stacked on top of one another, the distribution of activations (means, variances) tends to drift as it passes through layers, causing vanishing/exploding gradients (backward signals get too small or too large) and internal covariate shift (each layer sees changing input statistics during training). 

**LayerNorm:** For each token's hidden vector $x \in \mathbb{R}^d$:
$$\mu = \frac{1}{d} \sum_i x_i, \sigma = \sqrt{\frac{1}{d} \sum_i (x_i - \mu)^2}$$
then we normalize:
$$\textbf{LN}(x)_i = \gamma \frac{x_i - \mu}{\sigma + \epsilon} + \beta$$

where $\gamma, \beta$ are the learned scale and shift (per feature), and $\epsilon$ avoids division by zero. It normalizes across the feature dimension for each token, ensuring each embedding vector has zero mean and unit variance. 

**RMSNorm:** This simplifies LayerNorm by dropping the mean subtraction, and only normalizes the _magnitude_.

$$\textbf{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_i x_i^2 + \epsilon}} \cdot \gamma$$

This is much cheaper (there is no mean computation), keeps the direction of the hidden vector intact, and works equally well empirically when we combine this with residuals.

> Intuitively, imagine each token's embedding as a point in a high-dimensional space. After each layer's transformations (attention, FFN), some embeddings blow up in length, whereas others shrink. LayerNorm/RMSNorm act to re-center and re-scale, so that all these embeddings are kept roughly on a sphere of radius 1, and no single dimension dominates, thereby making optimization smoother. Indeed RMSNOrm literally projects each token's vector onto a hypersphere of fixed radius scaled by $gamma$. 

Now, we typically combine RMSNorm with a _residual connection_, which means that instead of letting each layer completely overwrite its input, we let it add its transformation on top of it:

$$y = x + f(x)$$

This leads to easier gradient flow, and acts like an incremental update - we refine the representation rather than replace it entirely. We normalize the signal so the layer can learn a clean delta, then add the delta back to the running state. Think of this process as a scientist refining a hypothesis.

- The residual connections allow each layer to slowly accumulate and refine their representation without overwriting their former work.
- RMSNorm/LayerNorm and the normalization layer calibrates the embeddings so that each layer operates under stable numerical conditions. 

The combination of these two elemnets enables us to train extraodinarily deep transformer-based networks without stability issues.

#### Understanding the softmax - why it exists and what it does

At the heart of self-attention lies a deceptively simple question:

> Given a query token, how much should it pay attention to each other token in the sequence? 

When we compute the dot products $QK^T$, we obtain raw similarity scores -- large numbers if the query and key vectors point in similar directions, and small (or negative) numbers otherwise. But these raw scores by themselves are just unbounded real numbers; they do not yet have the semantics of "attention weights". We need a way to turn them into _a smooth probability distribution_ over all possible keys - one that says, in effect, _"out of all tokens, here is how much I will listen to each."_ 

This is where the softmax function comes in; for every row of scores $s_i = (s_{i1}, s_{i2}, \cdots, s_{iN})$, it computes, over all of the $j = 1, \cdots, N$ elements in this row $S(s_i)_j$:

$$ \frac{e^{s_{ij}}}{\sum_{k=1}^N e^{s_{ik}}}$$

where each score is exponentiated, and then normalized by the sum across all keys. This accomplishes two crucial goals:

1. _Exponentiation_ [^7] amplifies differences between scores, so slightly higher similarities become disproportionately more important, helping the model focus.
2. _Normalization_ ensures that all resulting weights are positive and sum to 1, turning the vector into a valid probability distribution over which tokens the query attends to. 

Hence, the softmax transforms a raw, unbounded vector of similarity scores into a set of interpretable attention weights, where larger values mean "listen more closely" and smaller values mean "pay less attention." 

> We can think of the softmax as a spotlight that distributes a fixed amount of "attention energy" across all tokens, tokens whose keys are strongly aligned with the query (large dot product) receive a larger share of light, whereas tokens that are unrelated get very little. The total light intensity is always 1, ensuring stable scaling across all layers. Hence there is a competitive mehcanism across tokens, where they all "bid" for the query's attention. 

In the attention equation, we always divide the dot product $QK^T$ here $d$ is the head dimension. Thisi s because the dot product of two random $d$-dimensional vectors grows with $d$, and without scaling, as dimensionality increases, the variance of the scores would blow up. The exponentials in the softmax would then saturate (most weights become near 0, and one dominates), which will hurt gradient flow. Dividng by $\sqrt{d}$ keeps the scores roughly unit-scaled regardless of dimension, keeping the softmax in its linear regime where gradients remain healthy. 

In self-attention, we compute this softmax for each query token independently - that is, row-wise over the score matrix $S = QK^T$, subtracting the maximum of each of the scores first $M = \max_j S_{ij}$:

$$A_{ij} = \frac{e^{S_{ij} - M}}{\sum_k e^{S_{ik} = M}}$$

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

- **Sparsification:** One of the most direct ways to tackle the quadratic complexity of attention is to _sparsify_ the attention matrix i.e. limit which queries can attend to which keys. **Block-sparse attention** means that we divide the $N \times N$ matrix into blocks (e.g. 32 by 32 or 64 by 64 submatrices) and zero-out (or skip) many of those blocks, computing only a subset that follows some pattern.

  This is more GPU-efficient than arbitrary elementwise sparsity because computations can still be vectorized on blocks. Many long-document transformer variants use patterns like _local attention_ (each token attends only to tokens within a window of size $w$), or _dilated patterns_ or a mix of local and global tokens. These can be represented as a block-sparse matrix (with blocks along the diagonal for local attention for instance). The net benefit is that complexity reduces to $O(n \cdot w)$ rather than $O(n^2)$ (if each query attends to at most $w$ keys). 

  ![Alt text](/image-7.png)

  Several groups have implemented _block-sparse FlashAtttention_, which applies the original FA's IO-aware approach but skips blocks that are not needed. For example, using a sliding window of radius $w$, we would compute blocks near the diagonal and not the far-off blocks, leading to much faster computation times than dense FlashAttention for large contexts, at the cost of missing some long-range connections (depending on the sparsity pattern). Researchers have found however that cleverly chosen patterns (with some global tokens or random attention) can approximate full attention well. NVIDIA's ampere GPU even introduce hardware support for sparse (2:4 structured sparsity [^2]) matrix multiply. 

> By sacrificing the ability of every token to attend to every other token, and instead structuring the pattern (often based on prior knowledge of locality in language or vision), one can bring the quadratic cost down substantially. When combined with FlashAttention-style tiling, block-sparse methods can become even more powerful. 

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

All the steps, $Q \cdot K^T$, softmax, dropout, and $P V$ are **fused in a single CUDA kernel**, eliminating redundant memory reads/writes between steps. We break down the large matrix operations with tiling, and on-the-fly softmax normalization to avoid ever writing the full attention matrix to HBM. Queries Q are processed in blocks, and for each block Q, the kernel iterates over blocks K (and V), loading them into SRAM and computing partial scores $S = QK^T$ for that tile. These partial scores are exponentiated and accumulated into the softmax sum. Output partials ($P V$) are accumulated as well - by the end, the correct attention output is produced without storing intermediate $N \times N$ matrices. 

To better appreciate the power of this fusion, let us first recap the naive pipeline from a vanilla framework from PyTorch or TensorFlow (from a memory movement perspective, which is the key factor that FlashAttention endeavours to optimize):

1. _Compute (attention) scores:_ $S = QK^T / \sqrt{d}$, and write $S$ (a matrix of size $N \times N$) to HBM.
2. _Apply mask and softmax:_ We ready $S$ from HBM and apply causal mask, exponentiate, and then normalize. Then write a normalized $A$ (=softmax$(S)$) which is again an $N \times N$ matrix back to HBM.
3. _Multiply by values:_ Read $A$ and $V$ from HBM, and compute $O = AV$, then write $O$ (a matrix of size $N \times d$) back to HBM.

Pay particular attention to the memory movement from HBM to on-chip - indeed each step forces hundreds of GB of data to move through HBM, especially as $N$ (our context length) becomes large. In particular both $S$ and $A$ are quadratic-size intermediates, and each $K, V$ block is reloaded from global memory every time it is needed for a different query; every kernel boundary forces a write-read pair to DRAM. 

FlashAttention optimizes the I/O on each of these steps - every global read (from HBM) is _reused many times_ before being evicted, and no intermediate tensors ever touch HBM - the only thing sthat leave the chip are the inputs $(Q, K, V)$ once and the final output $O$ once. Everything else, $QK^T$, softmax, exponentials, normalizations and partial $AV$ values will all stay and live out their usefulness in fast on-chip memory. We never need to move the huge $QK^T$ matrix because it never materializes, do not need to move the many softmax weights because they are never written to HBM, and we perform the $AV$ operations locally per tile, on the fly; and we fuse all of these operations into a single kernel to prevent unnecessary flushes to DRAM, so synchronization is done with very lightweight thread-block barriers rather than device-wide writes.

| Stage                             | Naïve attention I/O               | FlashAttention I/O                                               |
| --------------------------------- | --------------------------------- | ---------------------------------------------------------------- |
| **Scores (QK^\top)**              | Written to and read from HBM (2×) | Kept in shared memory only                                       |
| **Softmax intermediates**         | Written & read (2×)               | On-chip streaming, no writes                                     |
| **K,V reuse**                     | Reloaded for every query          | Reused within tile from shared memory                            |
| **Kernel boundaries**             | Separate kernels = flush to DRAM  | Fused single kernel                                              |
| **Global reads/writes per token** | (O(N^2)) loads/stores             | (O(N \times \text{tile size})) local loads, linear global access |


> FlashAttention's entire gain comes from removing all redundant global memory I/O - it never writes $QK^T$ or the softmax matrix to DRAM, fuses all substeps into one kernel, and reuses each $K/V$ tile multiple times in shared memory, making attention compute-bound, rather than memory-bound.

The algorithm proceeds as follows (for one attention head at a time, though batch and heads are parallelized as usual):

- _Tiled Forward Pass:_ Instead of computing $S = QK^T$ for all queries and keys at once, FlashAttention partitions the sequence into blocks (for example, blocks of 128 queries by 128 keys). It loops over key/value blocks for a given block of queries, loading a block of $Q$ and a block of $K$ (and corresponding $V$) from HBM into shared memory, computing the partial attention scores for that tile, and immediately applying Softmax normalization _within that tile_. Crucially, it keeps track of partial Softmax results so that after iterating over all key blocks, the final Softmax is correct as if done in one pass. This saves $2$ global memory passes (write and read) of an $N \times N$ matrix.

> This choice of tile size (e.g. how many queries per block and how many keys per sub-block) is crucial. Larger tiles mean more reuse and fewer iterations (which improves compute efficiency), but they consume more shared memory and registers. 

- _Online softmax and output accumulation:_ In naive attention, we would compute the softmax, write the normalized $A$ attention matrix, read the attention matrix then compute $AV$. In FlashATtention, for each tile once we compute the partial attention scores on our tile, we apply softmax incrementally (streaming), and immediately multiply by $V_{tile}$, accumulating the running outputs $O_i$. THis saves $2$ global memory passes of the $N \times N$ attention matrix $A$ i.e. the intermediate attention weights never leave on-chip memory.

#### FlashAttention v2

FlashAttention v1 noticed that naive attention materializes an $N \times N$ score matrix in DRAM and shuttles it back and forth through softmax and weighting $V$, inducing an enormous cost from I/O and DRAM movement. FlashAttention v1 asked _"What is the minimum information we truly need to keep off chip?"_, and indeed the answer is it is sufficient to only keep tiny, per-row softmax statistics and a running output rather than the whole attention score matrix. We therefore tile the sequence, load one $K, V$ tile at a time into on-chip memory, updates an online softmax for a block of queries, and immediately folds that into the output, without ever writing a giant score matrix into DRAM. 

_FlashAttention v2_ pushes this slightly further, by rethinking 1. the bookkeeping around softmax 2. the shape of parallelism 3. the division of work inside a thread block. 

1. **Softmax bookkeeping:** Online softmax needs two pieces of information per row - a running maximum $m$ for numerical stability, and a running exponential sum $l$. In v1, eaach time a new tile of keys/values arrives, we update $(m,l)$ and rescale the running output so it remains correctly normalized at every step. This rescale is a small, but frequent recurring scalar operation, often in higher precision, sitting on the critical path. 

  In v2, the insight is that _we do not keep the output permanently normalized_ and instead keep an unscaled output $\tilde{O}$ while we sweep over the tiles, and apply the single $\frac{1}{l}$ normalization at the very end. The softmax math is unchanged; we just defer the cheap but constant rescaling that used to happen after every tile. In backward, v2 stores only _one_ number per row now $$L = m + \log l$$ and recomputes local probabilities when needed. This leads to fewer tiny saled ops, fewer converstions, and more uninterrupted time on Tensor Cores. 

> The intuition here is that if we are filling a bucket from many faucets, it is wasteful to stop and remeasure the bucket's fraction full after each cup. We should instead just keep pouring, track the total, and then normalize once. 

2. **Paralli

#### FlashAttention v3

FlashAttention v1, v2 and v3 demonstrate how algorithm and kernel co-design can yield massive performance gains for a critical operation like attention, by being mindful of GPU memory hierarchies and parallel execution capabilities. The key takeaways are:

1. Fuse operations to avoid memory bottlenecks
2. Tile data to maximize on-chip reuse
3. Parallelize across all available dimensions
4. Overlap different computations to utilize all hardware units

By following these principles and adjusting to the specifics of the GPU architecture, one can achieve performance that approaches the physical limits of the device, enabling training and inference of long-context LLMs at feasible speeds. 

An example of a FlashAttention CUDA kernel is as follows:

```cpp
// Assumptions:
// - Q, K, V: [N, d], row-major
// - O: [N, d], row-major (output)
// - N = sequence length, d = head dimension
// - causal mask: ignore j > i when computing softmax(QK^T)
// - all matrices in FP16

#define BLOCK_SIZE 128
#define D_HEAD 64 // head size (e.g. 64 or 128)

__global__  void flash_attn_fwd_kernel(
    // This is all stored in global, main HBM GPU memory, access is quite slow (high latency), and the Q/O matrices are read/written once per thread. The K/V matrices are accessed repeatedly in tiles.
    const half* __restrict__ Q,
    const half* __restrict__ K, 
    const half* __restrict__ V,
    half* __restrict__ O,
    int N, int d) {
    
    // Shared memory for K and V tiles. This is fast on-chip SRAM cache, the BLOCK_SIZE (128) is the size of the tile being loaded by the thread block. D_HEAD is the dimension of the data. Threads within the same block read chunks of K/V from slow global memory into this fast shared memory to maximize computational throughput.
    __shared__ half tile_K[BLOCK_SIZE][D_HEAD];
    __shared__ half tile_V[BLOCK_SIZE][D_HEAD];

    // Load Q row assigned to this thread
    // Each CUDA thread in this kernel is assigned to compute the full output for a single query row (corresponding to a single token in the input sequence)
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= N) return;

    // Register for Q row and output accumulator. This is the fastest memory space, local to a single thread. q_row holds the query for the current token being processed, and the acc_row holds the accumulating output vector for the current token. Keeping data in registers avoids slow memory access during recomputation. 
    half q_row[D_HEAD];
    float acc_row[D_HEAD] = {0.0f};

    // Load q_row from global memory
    #pragma unroll
    for (int j = 0; j < D_HEAD; j++) {
      q_row[j] = Q[i * d + j];
    }

    // Softmax accumulator (for numerical stability)
    float max_score = -1e9f;
    float sum_exp = 0.0f;

    // Iterate over K/V tiles
    for (int t = 0; t < gridDim.x; ++t) {
      int j_base = t * BLOCK_SIZE;

      // Load K/V tiles into shared memory
      int j = j_base + threadIdx.x;
      if (j < N) {
        # pragma unroll
        for (int d_i = 0; d_i < D_HEAD; d_i++) {
            tile_K[threadIdx.x][d_i] = K[j * d + d_i];
            tile_V[threadIdx.x][d_i] = V[j * d + d_i];
        }
      }

      __syncthreads();

    // Compute QK^T for this tile
    #pragma unroll
    for (int j_tile = 0; j_tile < BLOCK_SIZE; ++j_tile) {
      int j_pos = j_base + j_tile;
      if (j_pos >= N || j_pos > i) continue;

      float dot = 0.0f;
      #pragma unroll
      for (int k = 0; k < D_HEAD; ++k) {
        dot += __half2float(q_row[k]) * __half2float(tile_K[j_tile][k]);
      }

      float score = dot / sqrtf((float)D_HEAD);
      float exp_score = __expf(score - max_score); // stable softmax 

      // Update softmax denominator and max
      max_score = fmaxf(max_score, score);
      sum_exp += exp_score;

      // Accumulate softmax * V
      #pragma unroll
      for (int k = 0; k < D_HEAD; ++k) {
        acc_row[k] += exp_score * __half2float(tile_V[j_tile][k]);
      }
    }
    __syncthreads(); // reload new K/V tile

  }

    // Normalize output
    float inv_sum_exp = 1.0f / (sum_exp + 1e-6f);
    #pragma unroll
    for (int k = 0; k < D_HEAD; ++k) {
      float val = acc_row[k] * inv_sum_exp;
      O[i * d + k] = __float2half(val);
    }
  }

```

Here `__restrict__` tells the compiler that this pointer is the only way to access the memory that it points to. In GPU programming, especially in memory-bound kernels like attention, it is critical for the compiler to know when two pointers cannot alias (i.e point to the same memory). When we write `const half* __restrict Q` then we are promising that `Q`, `K`, `V` and `O` all point to distinct, non-overlapping memory regions. This allows the compiler (e.g. nvcc) to safely cache values in registers, avoid unnecessary loads, and fuse memory instructions more aggressively.

Let us say that we have `O[i] = Q[i] + V[i]`. If Q, V and O are not marked as `__restrict__`, the compiler must assume O might alias Q or V. It then has to reload from memory (to be safe), or might not cache or reorder. But with `__restrict__`, the compiler knows that these do not overlap so it can cache and optimize freely. In practice, for memory-bound GPU kernels (like attention, matmuls, convs) this can give up to 10-30 percent speedup because it helps the compiler skip unnecessary loads. 

> `__restrict__` tells the compiler not to be paranoid, since the pointers are safely independent from each other; it can cache, reorder and optimize freely.

The `__shared__` memory space is a very important feature of the CUDA programming model taht tells the CUDA compiler to allocate a variable in the GPU's shared memory rather than in the slower global memory or the private registers of a single thread. 

Shared memory is a small, on-chip memory store (an SRAM cache) located very close to the GPU's SMs, and accessing this is up to 1000x faster than accessing global HBM. The memory allocated with `__shared__` is visible to all threads within the same thread block, but is not accessible to threads in different blocks, which makes it an ideal mechanism for cooperative data sharing among threads working on the same chunk of data (e.g. the tiling algorithms that we use in FlashAttention).

Because multiple threads within a block might read and write to the same shared memory location, a synchronization function `__syncthreads()` is often required, to ensure that all threads in the block have completed their memory accesses before proceeding, to _prevent race conditions_. 

Now let us take a look at how this kernel will be called and launched from the host (CPU) side. We will use `cudaMalloc` to allocate device (GPU) memory, copy data to and from the device with `cudaMemcpy`, launch the kernel with our grid/block configuration, and synchronize all the events with `cudaEventRecord`, `cudaDeviceSynchronize`, eventually freeing memory with `cudaFree`. 

```cpp
#include <cuda_runtime.h>
#include <cstdio>

void flash_attn_forward_launcher(
    const half* h_Q, const half* h_K, const half* h_V,  // host input
    half* h_O,                                          // host output
    int N, int d
) {
    const size_t size_QKV = N * d * sizeof(half);
    half *d_Q, *d_K, *d_V, *d_O;

    // Allocate on device
    cudaMalloc(&d_Q, size_QKV);
    cudaMalloc(&d_K, size_QKV);
    cudaMalloc(&d_V, size_QKV);
    cudaMalloc(&d_O, size_QKV);

    // Copy host → device
    cudaMemcpy(d_Q, h_Q, size_QKV, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size_QKV, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size_QKV, cudaMemcpyHostToDevice);

    // Kernel launch config
    const int BLOCK_SIZE = 128;
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Timing (optional but best practice)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    flash_attn_fwd_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_Q, d_K, d_V, d_O, N, d);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("FlashAttention forward took %.3f ms\n", ms);

    // Copy result back
    cudaMemcpy(h_O, d_O, size_QKV, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

```

### Deconstructing Frontier OSS models from first principles

To consolidate everything that we have discussed in this post, we will deconstruct two leading (as of 2025) open source models, Kimi K2 and Qwen2.5-32B, both are autoregressive decoder-only models that are intended to be served at scale, with low latency as text-generating intelligent assistants.

Here when we say that Qwen2.5-32B has 64 layers, we mean that there are 64 repetitions of:

`RMSNorm -> attention -> residual -> RMSNorm -> FFN -> residual
`

where each repetition deepens the reasoning chain and allows the model to integrate increasingly abstract contextual information. The attention sublayer is used for token-token communication ("talk to the other tokens"), the feedforward sublayer is used as a uniform nonlinear transformation per token (no mixing between the tokens, allowing the tokens to transform and "think for themselves") [^6], and then the residuals and norms are used to stay stable whilst stacking the layers deep. 

| **Specification**            | **Qwen2.5-32B**                                 | **Kimi K2 (MoE)**                                                                                       |
| ---------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Release / Type**           | Dense, decoder-only                             | Mixture-of-Experts, decoder-only                                                                        |
| **Parameters (total)**       | 32.5 B                                      | ≈ 1 T total; ≈ 32 B activated                                                                   |
| **Layers**                   | 64                                          | 61 (incl. 1 dense layer)                                                                            |
| **Hidden size (d_model)**    | 5120                                        | 7168 (attn); FFN per-expert ≈ 2048                                                                  |
| **Attention Heads (Q / KV)** | 40 / 8 (GQA)                                | 64                                                                                                  |
| **Positional Encoding**      | Rotary (RoPE)                                   | MLA-based attention (positional bias built in)                                                          |
| **Activation / FFN**         | SwiGLU, RMSNorm                             | SwiGLU                                                                                              |
| **Context Window**           | 131 072 tokens                              | 128 000 tokens                                                                                      |
| **Tokenizer / Vocab Size**   | ≈ 152 k tokens (BPE)                            | ≈ 160 k tokens (BPE)                                                                                    |
| **Architecture Notes**       | Long-context + GQA for efficient KV cache usage | 384 experts, top-8 routing (+ 1 shared expert); MLA attention; sparse activation for efficiency |

Note that both of these models are _decoder-only_ models instead of encoder-decoder models. Think of decoder-only models as next-token storytellers, that read a prefix, and keep predicting the next token (with a causal mask, so they cannot peek at the future), that are great for open-ended generation (chat, code, writing) and streaming outputs. Encoder-decoder models however read the whole input bidirectionally and compress it into rich features; and a decoder then generates the target sequence while attending back to those features, this is great when we have a clearly separated input to output task (e.g. translation, summarization, and question-answer over a passage). 

| **Area**                | **Decoder-only**                                                            | **Encoder–decoder** [^3]                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Latency & streaming** | Excellent. KV cache lets you append tokens with tiny per-token cost; streams naturally. | Upfront encode cost each turn; streaming possible but heavier (decoder depends on full encoder memory). |
| **Throughput & cost**   | Scales well with batching + KV cache reuse; single stack, fewer cross-attn passes.      | Two stacks + cross-attention; re-encoding (even with prefix caching) adds steady overhead.                  |
| **Long context**        | Mature: FlashAttention, GQA/MQA, 128 k+ windows, rolling caches.                        | Works, but you pay encode cost proportional to input length whenever it changes.                            |
| **RAG grounding**       | Works (prepend docs), but generation competes with understanding in one pass.           | Strong: encode many docs once, decoder focuses on synthesis; easier to constrain to sources.                |
| **Serving simplicity**  | One graph; mature kernels, simpler caching, easier autoscaling.                         | Two graphs; more knobs (encoder cache, cross-attn shapes), trickier batching.                               |
| **Fine-tuning & data**  | One objective (causal LM) across tasks; instruction/RLHF sits naturally.                | Pretrain often denoising/seq2seq; finetunes may need task-specific mixes.                                   |
| **Memory footprint**    | KV cache only for decoder; predictable growth with output length.                       | Encoder activations + decoder KV + cross-attn; larger per-request memory.                                   |

Note that if cost, scalability and speed were not a concern, encoder-decoder models would be strictly superior than a pure decoder-only model (given that the decoder portions of both models are the same size [^4]), because the encoder-decoder is a two-stage thinker that reads the entire input like a careful researcher, builds a compact, bidirectional representation, then decodes this into an output whilst looking back at the encoded representation (via. cross-attention). It reads, organizes its thoughts, and then writes. Note that FlashAttention could be applied to the encoder self-attention, the decoder self-attention (causal), and also cross-attention. 

The decoder-only model however is a think-aloud monologue, for example in a live presidential debate, where we have a growing conversation from left to right that speaks as it thinks - it is optimized for the "just keep going" mentality, appending tokens, reusing the KV cache, and emitting the next token with minimal overhead. Both paradigms have their strengths and weaknesses, the choice is more about where we want to spend out compute, and where the product must shine most. Generally, we would choose a decoder-only model if our product is conversation/streaming, needs very long outputs relative to inputs (KV cache dominates), and cost-sensitive; on the other hand we would choose an encoder-decoder if our product is grounded generation over long, fixed inputs (e.g. document question-answer), multi-modal fusion before text, and short outputs relative to large inputs (encode once, and decode small), and also requires strong faithfulness, citation and provenance constraints [^5].  

> Decoder-only models behave like seasoned improvisers: they think while they speak and are engineered for fast, continuous output. Encoder–decoders act like meticulous writers: they read everything first, organize a global understanding, then compose. If you can ignore latency and cost, the two-stage approach is at least as expressive and often more faithful on document-grounded tasks. In production, though, streaming UX, KV caching, and serving simplicity make decoder-only the default for chat at scale.

[^1]: Though there have been recent efforts to combine the two ideas, e.g. "Mixture-of-Head Attention" (MoH) where attention heads themselves are treated as experts and are sparsely activated.
[^2]: 2:4 sparsity is a specific instance of a more generla N:M structured sparsity where a block of M consecutive weights must contain at most N non-zero values. This achieves excellent acceleration on modern GPUs. Coarse-grained sparsity refers to pruning entire channels or blocks, with less accuracy at high sparsity levels than N:M but a simpler implementation.
[^3]: Some examples of encoder-decoder models are T5 (text-to-text transformer) and BART (bidirectional and auto-regressive transformer) as well as UL5. They are used mainly for speech recognition, image captioning, text summarization and language conversion.
[^4]: It would actually be an interesting study to analyze say for a fixed number of parameters $B$, if we have a $X:Y$ split amongst the encoder-decoder, compared to just a decoder with the same number of of parameters $B$, the comparative performance between the two. How does this change as $B$ gets large. 
[^5]: Suppose we want to accurately cite every aspect of a scientific article without hallucinations, an encoder-decoder architecture would be much better at this specific task than a decoder-only model. 
[^6]: In mixture of experts blocks like Kimi, the FFN step is replaced by multiple "expert" MLPs, and a small gating network picks a few experts (e.g. top-8 of 384) to process each token. The outputs are combined, and then passed to the next block. 
[^7]: We use exponentials for a variety of reasons (at a high level they provide smoothness and contrast). They ensure non-negativity (no negative probabilities), make the function differentiable everwhere (very important for moving over the loss curve and backpropagation) and they yield a sharp focus when needed - a few large scores dominate, while small ones fade exponentially; a "controllable soft selection". 
[^8]: Note that when we compute the attention score and do the weighted sum over the token values, we also include the token's own value $V_i$. Indeed, in most cases the resulting attention weights $A_{ij}$ have a non-trivial diagonal i.e. the weights that the tokens assign to _themselves_ $A_{ii}$. Early in training it tends to be relatively large (tokens attend strongly to themselves), and as the model learns it begins to redistribute attention across context tokens. We know from studies for example BERT-style encoder ablations ("Rethinking the importance of analysis in self-attention") that when masking the diagonal, performance was comparable and sometimes slightly better than baseline. Studies show that the diagonal is one of the least important positions; perhaps we don't need an explicit self-lookup in the attention sublayer because the _residual path alreeady preserves self-information_. 

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 5998–6008.
2. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. Advances in Neural Information Processing Systems (NeurIPS 2022)
3. Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., et al. (2022). PaLM: Scaling Language Modeling with Pathways. arXiv preprint arXiv:2204.02311.
4. "Accelerating sparse deep neural networks" is Mishra, A., Latorre, J. A., Pool, J., Stosic, D., Stosic, D., Venkatesh, G., Yu, C., & Micikevicius, P. (2021). Accelerating sparse deep neural networks (arXiv:2104.08378).
5. Qwen2.5 (7B / 32B and family)
Qwen Team. (2024/2025). Qwen2.5 Technical Report. arXiv:2412.15115 (v2, Jan 3, 2025).
6. Kimi K2 (Moonshot AI)
Kimi Team. (2025). Kimi K2: Open Agentic Intelligence (Technical Report). arXiv