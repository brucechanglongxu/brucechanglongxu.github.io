---
title: "Real world ASI III: Pretraining"
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

---

Pretraining is where you turn a mountain of tokens into a general-purpose model that actually understands the world you care about. The target is simple to say and hard to execute: learn a policy that predicts the next symbol so well, across such a broad and balanced mixture, that useful capabilities emerge without hand-holding. Everything starts with intent. Write down the capabilities you expect to surface at the end of the run—language, code, math, tool-use priors, multimodal alignment, long-context recall—and let that plan drive what enters the corpus and how it is proportioned over time. A trillion indiscriminate tokens won’t beat a carefully balanced hundred billion that genuinely reflect the deployment world.

> Pre-training is the art and craft of compressing the entire internet, cultures, physical, auditory and visual world into dense matrics of weights and activations to deploy in the forward pass. 

Formally, pretraining minimizes the negative log-likelihood over a mixture of datasets. Let the training distribution be $M=\sum_{i}\lambda_i D_i$ with $\lambda_i\ge 0$ and $\sum_i \lambda_i=1$. The objective is the usual autoregressive loss

$$
L(\theta)=\mathbb{E}_{x_{1:n}\sim M}\Big[-\sum_{t=1}^{n}\log p_\theta(x_t\mid x_{<t})\Big],
$$

but the interesting work hides in the choice of the $\lambda_i$ and how they evolve. Mixture weights should not be static; they should anneal with a curriculum that starts with cleaner, simpler distributions and gradually increases difficulty—more code and math, more multilingual and domain-specific data, longer contexts, more tool-marked traces—while guarding eval integrity and license boundaries. Measure effective tokens, not raw tokens; de-duplicate aggressively, remove templated spam, quarantine known evals, and keep lineage so you can trace any sample from source to checkpoint.

Compute is the other half of the equation. At this scale, the budget is not abstract—it constrains everything from parameter count to token count to context length. A simple first-principles model is that total training FLOPs scale on the order of $C \approx k\,P\,T$, where $P$ is non-embedding parameters, $T$ is seen tokens, and $k$ is a constant that captures architectural details. Under a fixed $C$, the loss after training is well-approximated by a separable scaling law,

$$
L(P,T)\;\approx\;L_\infty + a\,P^{-\alpha} + b\,T^{-\beta},
$$

which makes the trade clear: too many parameters with too few tokens and you under-train; too many tokens with too few parameters and you underfit. The job is to pick $(P,T)$ (and a context schedule) that sit near the compute-optimal frontier for your mixture, then actually hit those targets in a world with failures, preemptions, and imperfect utilization. That demands diligence on both the math and the logistics: you plan the frontier, then you execute a run that stays on it.

Architecture and representation choices are quietly decisive. Tokenization determines compression and the shape of the softmax; vocabulary that’s too small bloats sequences and attention cost, too large creates optimization drag, so treat it as an architectural hyperparameter rather than an afterthought. For long context, use encodings and training schedules that don’t crumble as you scale from short to long sequences; ramp the window over the run instead of flipping a switch in the last 5%. Keep normalization simple and stable, and pick activations that play well with mixed precision. Dense models remain the simplest thing that works; mixture-of-experts can buy you quality-per-compute if your router receives enough varied training signal and your batch shapes keep every expert learning. However you slice it, remember that pretraining should teach *general* priors: reasoning patterns that later fine-tuning can shape rather than fight.

Optimization is where runs live or die. Warm up calmly, schedule the learning rate with a predictable decay (cosine or step both work when tuned), and keep a tight grip on stability: gradient clipping on by default, loss-scaling solid, update-fusion and activation checkpointing used for what they are—means to fit the job, not badges. Large batches buy you throughput but can flatten the loss landscape; compensate with schedule and regularization rather than assuming bigger is always better. Monitor the gradient-noise scale, keep an eye on optimizer state growth, and never ignore the first telltales of instability: rare NaNs, sudden validation spikes on specific domains, or weird oscillations on long-context probes. When those appear, treat them as smoke from a real fire, not a logging artifact.

Curriculum is not just for language difficulty; it applies to modalities and behaviors too. If you plan to expose the model to tool-augmented traces post-training, seed those patterns lightly during pretraining so the policy isn’t shocked by function-call tokens later. If code quality matters, make sure code isn’t a rounding error in the mixture and that formatting and docstring patterns are diverse enough to prevent brittle shortcuts. If multilingual competence is a requirement, stage languages with attention to script diversity, tokenization quirks, and realistic domain content rather than dumping a pile of parallel corpora in one go. Above all, make mixture updates incremental and measurable: change one dial, watch validation move, then commit.

Validation deserves paranoia. Hold out stratified sets that mirror your mixture, but also keep clean “canary” suites that never leak into training: math scratchpads, domain-specific QA, retrieval-grounded tasks where you can check citation precision and recall, and long-context narratives that punish shallow recency hacks. Perplexity is necessary but not sufficient; track capability proxies tied to your intent. When a domain improves on validation, confirm it didn’t come from contamination; when it degrades, look for mixture drift before you blame the optimizer. Keep the validation pipeline deterministic and auditable so regressions are explainable.

Finally, treat the run like a product with artifacts you could hand to a future you. Seed management, checkpoint cadence, and resumability are boring until they are the only things that let you survive a cluster hiccup without derailing compute-optimal plans. Tag checkpoints with mixture versions and context windows, not just step numbers, so downstream comparisons are apples-to-apples. Log the metrics that matter—tokens per second, realized FLOPs per second, memory headroom, attention flop share, validation curves by domain—so you can tell whether you are moving along the planned frontier or wandering off it. Pretraining doesn’t have the immediate dopamine hit of a post-training win-rate bump, but it sets the gradient for everything that follows. When done well, it makes the later stages feel like sculpting; when done poorly, it turns them into repair.

## Backpropagation

We feed an input $x$ into the network, and layer by layer it computes: 

$$a^{(0)} := x$$
$$z^{(l)} := W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} := \sigma(z^{(l)}) \textbf{(e.g. ReLU/GELU/Softmax)}$$

The tensors $z^{(l)}$ and $a^{(l)}$ are the layer's forward activations, the intermediate results produced on the way to the final output. In Transformer's, _"activations"_ includes hidden states per token, the Q/K/V projects, attention scores/probabilities, MLP intermediates, LayerNorm stats, etc. 

_Backpropagation_ is the chain rule applied backward to compute gradients of the loss $L$ with respect to every parameter. Starting from the loss gradient at the output, we propagate an _"upstream gradient"_ $\delta^{(l)} := \partial L / \partial z^{(l)}$ through each layer to get 

## Napkin Heuristics

> **Theorem Alpha:** $$C \simeq C_{\textbf{forward}} + C_{\textbf{backward}} \simeq \tau T = 6PD$$ 

This is the cost that is required to train a transformer, where we sum the cost (in FLOPs) of the forward and backward pass. $\tau$ denotes the cluster's throughput, and $T$ denotes its training time. 

Indeed suppose that we have a 70B parameter open source model (e.g. LLaMA), which was trained on 200B tokens using 2048 NVIDIA A100 GPUs. We can apply the above equation:

- The peak float16 FLOPs throughput of an A100 is $\tau = 312$ teraFLOPs $\simeq 3.12e14$ FLOPs. Since we have $2048$ A100 GPUs, the "collective" (of course in practice we probably would have less flops due to communication overhead), throughput is $2048 * 3.12e14$. 

- Based on the above  _Theorem Alpha_, we know that the total compute required is $7.9e10 * 2.0e11 * 6 = 9.48e22$

- This means that the training time for the model must take around $T = C / \tau = 9.48e22 / (2048 * 3.12e14) / 86400 \simeq 1.72$ days. 

The above napkin math is often an estimate on the optimistic end, because more often than not, as mentioned briefly, we will not achieve the peak throughput $\tau$ on our GPUs or processors with our current distributed training paradigms and communication overhead. This is the variable that is primarily responsible for our margin of error. 

_Checkpointing Caveat_

During training, backpropagation needs each layer's forward activations to compute gradients. If we store 

- Pytorch: `torch.utils.checkpoint.checkpoint(function, *inputs)`
- JAX: `jax.checkpoint` / `jax.remat`
- DeepSpeed/FSDP: _"activation checkpointing"_ per layer/block

> Checkpointing reduces activation memory by intentionally forgetting and later recomputing, so the FLOPs budget goes up (extra forward work during backward). 

_Derivation of Theorem Alpha_

> **Axiom FLOPS:** The FLOPs that truly matter in training are weight FLOPs, that are performed when intermediate states are multiplied by weight matrices. As we tend towards infinity, weight FLOPs dominate until non-weight FLOPs are negligible. 

Weight FLOPs for multiplying a matrix $W$ is equal to $6$ times the batch size times the size of $W$. 

1. Hoffmann et al. _Training Compute-Optimal Large Language Models_ Google Deepmind, 2022.
2. 