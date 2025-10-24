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

In practice, we represent data as triplets of query (Q), key (K) and value (V) vectors. The _query_ represents what content we are looking for, each _key_ represents what content a particular value contains, and each _value_ is the content to retrieve. The model learns projections to produce Q, K, V from inputs (e.g. the same input sequence for self-attention), and uses a scoring function between Q and K to decide how much of each value to include in the output. This mechanism was first popularized to help sequence models _attend_ to relevant parts of an input. 

_Additive vs. Dot-product Attention_ 

### GPU Optimizations: Fusing kernels, Tiling, memory Access and Precision

> Every new idea in this space, be it FlashAttention, GQA, ALiBI, RoPE or block-sparse patterns, contributes to one of two goals: 1. Making attention faster/leaner or 2. Enabling attention to capture needed information more efficiently. 

Implementing MHA efficiently is critical for both training and inference of large-scale AI models. The MHA mechanism is often a major contributor to the overall computation and memory usage in large Transformer networks; indeed training time can become very long due to the quadratic cost complexity as models scale up in size. Naively computing MHA would involve multiple large matrix operations and memory-heavy steps: computing the $QK^T$ attention score matrix for each head (which for a sequence length $n$ is a huge $nxn$ matrix i.e. $O(n^2)$ operation), storing these scores, applying softmax, and then multiplying by the $V$ matrix. This could lead to significant memory bandwidth demand and under-utilization of compute units if not optimized. For example, the softmax computation is sequential, element-wise that does not fully leverage our GPU's parallel FMA units; and we will also need to read and write large intermediate matrices from high bandwidth memory (HBM). 

- **Fused Attention Kernels:**
- **Leveraging Tensor Cores and Hardware Optimizations:** 
- **Memory and Cache Optimizations:** 

[^1]: Though there have been recent efforts to combine the two ideas, e.g. "Mixture-of-Head Attention" (MoH) where attention heads themselves are treated as experts and are sparsely activated.

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 5998–6008.