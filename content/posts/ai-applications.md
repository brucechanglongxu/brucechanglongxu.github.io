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

In practice, in the forward pass (during both training and inference), a batch of contextualized vectors (from the MHA layer) enters the FFN. The exact same learned parameters $\mathcal{L}_{\mathbf{FFN}}(W_1, b_1, W_2, b_2)$ are applied independently to each token's vector in parallel. A new batch of processed vectors which has now been expanded, processed in the MLP/FFN latent space, and contracted back to the original input dimension, is now passed to the next layer via. a residual connection and layer normalization. Now in the backward pass (which occurs during only training), the model's error (loss) is calculated at the output layer and then backpropagated through the network. The gradients are computed for the learnable parameters $\mathcal{L}_{\mathbf{FFN}}(W_1, b_1, W_2, b_2)$ and an optimizer (such as Adam or AdamW) uses these gradients to slightly adjust the numerical values of the four learned parameters, teaching our FFN how to better process information next time. 

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


Applications are where capability meets consequence. A good application does not look like “a model with a UI”; it looks like a choreographed system that turns intent into outcomes under real constraints: latency, trust, safety, compliance, and cost. The model is the engine, but the car is everything around it—retrieval that keeps answers grounded and current, tools that execute real actions, guardrails that prevent harm, observability that explains behavior, and a feedback loop that steadily improves the whole stack. Done well, the experience feels simple to the user and operationally boring to the on-call engineer, even though it’s balancing tight budgets and shifting risk in the background.

Start with the serving path. Decoding is the hot loop, so you budget it like any other production system: shape requests into batches without blowing out tail latency, pin memory headroom for KV caches, and size concurrency to match your admission limits. Little’s law is a helpful sanity check for capacity planning—if requests arrive at rate $\lambda$ and the average time in system is $W$, then the average in-flight load is $L=\lambda W$; when $L$ rises, either slow the intake or shorten the work. Speculative decoding trims wall time, paged caches keep long contexts from thrashing, and autoscaling should be driven by both utilization and SLO proximity, not just raw QPS. Multi-tenant isolation and cost attribution matter as much as raw speed; an application that can’t keep tenants from starving each other, or can’t explain its cloud bill, is one outage or one executive review away from rollback.

Grounding is the second pillar. Retrieval-augmented generation works when the index is fresh, deduplicated, and attribution-friendly. Chunking, passage scoring, and reranking are product choices, not academic footnotes, because they determine whether a user sees verifiable evidence or a plausible guess. Applications should support “evidence-required” modes that refuse to answer when sources cannot be retrieved or verified; this is not just a safety posture, it is a quality posture. Treat prompt injection and data exfiltration as first-class threats: sanitize inputs, strip dangerous instructions from retrieved context, sandbox tool calls, and never let the model leak secrets simply because they appeared in a prompt. Grounding isn’t only about citations; it’s about making answers reproducible and auditable.

The third pillar is orchestration of tools and agents. Most practical value flows from models that know when to look things up, when to compute, when to call an API, and when to ask for help. Prefer shallow plans with robust interrupts, deadlines, and fallbacks over deep chains that only work in demos. Route to tools with small, supervised heads and reward grounded, minimal action sequences in post-training so the policy doesn’t learn to thrash. Every tool runs in a sandbox with strict resource and permission boundaries, and commits that have real-world impact are gated behind confirmations or evidence thresholds. When a tool fails—as it will—the application degrades gracefully with a truthful explanation instead of inventing a result.

Safety and compliance wrap the entire surface. Inputs pass through validation and policy checks before generation; sensitive domains can demand retrieval evidence or trigger refusal templates. During generation, decoding constraints and topic-aware modes keep the model inside the rails. After generation, lightweight classifiers scrub PII and screen for policy violations, and any action that touches user data is logged with purpose, provenance, and consent. Region-aware behavior, retention windows, and deletion guarantees are product features, not footnotes; users remember whether your application respected boundaries. Keep red-team corpora and adversarial prompts versioned and integrated into both evaluation and training so your defenses evolve with attacks rather than lag them.

Observability turns behavior into something you can reason about. Trace every request end-to-end with structured logs that include tool calls, retrieval contents, and timing at each hop; collect token-level telemetry where privacy allows, with redaction baked in rather than stapled on. Build an event taxonomy—answer served, safe refusal, escalation, policy block, tool failure—so dashboards tell a coherent story instead of a blur of counters. Online evaluation should be continuous and unobtrusive: small interleaved experiments for win-rate and satisfaction, automated checks for grounding precision, drift detectors on domain slices that matter to your users. When something shifts, you want to know whether the cause was a model change, an index update, a tool outage, or a traffic mix you didn’t anticipate.

Applications close the loop back to training. Every “needs-work” example, every unsafe refusal, every agentic failure becomes labeled fuel for the next SFT or preference pass—after consent and de-identification, and with clear policy on what production data may become training data. Active learning is most useful when it is specific: mine prompts that produced low-confidence answers, cluster similar failures, and generate targeted negatives that teach the model to prefer grounded, succinct, and correct behavior. This is how products get better week over week without gambling on giant retrains: small, clean deltas with measurable effects.

In production, quality and economics meet. It helps to look at a simple ratio: the cost per helpful answer is $C_{\text{help}} = C_{\text{req}} / p_{\text{help}}$, where $C_{\text{req}}$ is your average serving cost and $p_{\text{help}}$ is the probability a response actually helps the user under your definition. You can push $C_{\text{help}}$ down by making serving cheaper—distillation, quantization, batching—or by making answers more often helpful—better grounding, clearer prompts, tighter post-training. Either route is legitimate; the best teams do both, and they watch the tails as much as the averages, because users experience P99.

Shipping discipline keeps you out of the news. Releases ride canaries with automatic rollback on SLO breach or safety incident; model cards and changelogs turn behavior changes into something leaders and regulators can read. The same rigor applies to A/Bs: pre-register hypotheses, define stopping rules, and guard against peeking and novelty effects so you don’t ship mirages. When incidents happen—and they will—the application offers a clear path to remediation: reproduce, explain, patch, re-evaluate, and, if necessary, revert.

In the end, “Applications” is not a veneer on top of a clever model; it is the environment in which the model learns to be useful, safe, and affordable. The best products make that environment explicit: grounded by retrieval, empowered by tools, wrapped in guardrails, observable to a fault, and stitched into a feedback loop that steadily raises the floor while lowering the cost. That is how real-world ASI earns trust—one request at a time, under the bright light of production.

[^1]: Though there have been recent efforts to combine the two ideas, e.g. "Mixture-of-Head Attention" (MoH) where attention heads themselves are treated as experts and are sparsely activated.

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 5998–6008.