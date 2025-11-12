---
title: "Real world ASI IV: Posttraining"
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

Post-training is where a broadly capable base model $\pi_0$ becomes a dependable product $\pi$. The work breaks into three intertwined threads: teach the behaviors you actually want to see, align those behaviors with real user preferences, and harden the system so it’s safe, observable, and economical to run. Treated well, this isn’t a waterfall but a tight control loop: add data, run a short supervised pass, take a preference step, re-benchmark offline and online, and ship behind a small canary before widening exposure. Keep a true quarantine set of prompts and tasks that never touch training so you can detect contamination and overfitting; track style drift against $\pi_0$ so you don’t sand the model’s useful instincts down to a bland mean while chasing win-rates.

Teaching behavior starts with supervised fine-tuning on traces that look like production. That means clean instruction–response pairs, yes, but also multi-turn dialogs and, critically, tool use: retrieval calls with passages attached, code execution with inputs and outputs, calculators with intermediate results, structured API invocations with error handling. The goal is not to stuff the transcript with artificial scaffolding; include intermediate reasoning only when it improves user outcomes or helps downstream judges separate good from bad behavior. Refusal and escalation should be explicit, not emergent: when the model should ask for clarification, when it should call a tool, when it should defer or decline. In practice, SFT is most effective when the dataset is de-duplicated, stratified by domain and difficulty, and salted with a modest number of hard negatives so later preference steps have sharp signal. Anti-regression checks belong here too: grounding precision and recall on citation tasks, fidelity of references, and tool-routing accuracy under paraphrase and context shifts.

Alignment then layers user preferences on top. One well-tested route is reinforcement learning with a learned reward model and a KL stabilizer that tethers the policy to its reference. The generic objective looks like

$$
J(\pi) \;=\; \mathbb{E}_{x\sim D,\,y\sim \pi(\cdot|x)}[r(x,y)] \;-\; \beta\,\mathrm{KL}\!\left(\pi(\cdot|x)\,\|\,\pi_0(\cdot|x)\right),
$$

where $\beta$ trades off reward-seeking against style stability and guards against reward hacking. Another increasingly popular route dispenses with an explicit reward model and optimizes direct preferences from pairwise judgments. A standard form is

$$
\mathcal{L}_{\mathrm{DPO}}(\pi)
= - \mathbb{E}_{(x,y^+,y^-)}
\left[
  \log \sigma\!\left(
    \beta\!\left[\log \pi(y^+\!\mid x) - \log \pi(y^-\!\mid x)\right]
    -
    \left[\log \pi_{0}(y^+\!\mid x) - \log \pi_{0}(y^-\!\mid x)\right]
  \right)
\right],
$$

with $\sigma(t)=\frac{1}{1+e^{-t}}$. In both cases, signal quality dominates everything else. Diversify annotators, filter templated pairs, mix in deliberately tricky negatives, and watch for pathologies like verbosity inflation, hedging, and gratuitous citations. Short SFT refreshes interleaved with preference steps keep the policy anchored; periodically updating the reference or re-anchoring to $\pi_0$ prevents slow drift. Over-optimization shows up first in diversity and calibration, so monitor response variety, tokens-per-answer, grounding precision, and confidence calibration rather than a single aggregate score.

Safety and policy shaping belong in layers around generation, not as a single classifier stapled on at the end. Inputs should pass through basic validation, policy screening, and retrieval grounding where required, so the model sees the right context before it speaks. During generation, you can enforce citation-required modes for specific domains, constrain decoding in sensitive areas, and keep refusal patterns consistent. After generation, run lightweight PII scrubbers, domain-specific safety filters, and, for tools that can make irreversible changes, guard commit points behind evidence thresholds. Red-team corpora—adversarial prompts, jailbreak attempts, harmful content—should be versioned assets in the training and evaluation pipeline, not a side folder. Pair that with a human-readable policy document and a changelog so behavior shifts are explainable at each release.

Most real productivity gains come from orchestration: models that know when and how to use tools. Train simple routers—small classifiers or heads on the policy—to decide when to retrieve, when to call a calculator, when to draft code, and when to ask the user for more information. Retrieval-augmented generation is only as good as the hygiene of your index: freshness, de-duplication, passage attribution, and sensible chunking matter more than squeezing another 0.5% from an embedding recall benchmark. When you train with preferences, penalize ungrounded spans on tasks that require evidence so the model learns that citing isn’t optional theater but part of correctness. Be wary of deep agentic chains without tight supervision and timeouts; shallow plans with robust interrupts are usually more reliable in production.

To hit latency and cost targets without giving up too much quality, shrink and speed up. Distill teachers into students with logit matching and preference-consistent losses, then re-tune the student on tool traces so it doesn’t forget how to act. Quantize weights to INT8 or INT4 and consider FP8 for activations where kernels support it; validate not only average accuracy but also long-context behavior and tool-heavy tasks, which tend to regress first. On the serving path, speculative decoding can lop off wall-time at modest compute overhead, paged KV caches keep memory from exploding at long context, and admission control with sane batching keeps both P50 and P99 inside budget. Multi-tenant isolation, sandboxed tool execution, and rate limits round out the practicalities of sharing an inference fleet across products.

Evaluation is a system, not an event. Offline, you still need the familiar batteries—helpful, harmless, honest—alongside code, math, and domain-specific tasks, but add targeted probes for grounding: measure precision and recall on citation correctness and measure whether refusals are appropriate when evidence is missing. Treat calibration as a first-class metric; a simple expected calibration error makes drift visible:

$$
\mathrm{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{n}\,\bigl|\,\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\,\bigr|.
$$

Diversity metrics help catch collapse into a single verbose style. Online, run small, fast A/Bs to track win-rate, safe-refusal rate, escalation and deferral rates, latency SLO hit rate, and incident frequency, and wire every “needs-work” example back into the SFT and preference backlogs. The point is not to maximize a single number but to keep a multi-signal dashboard that tells you when to push, when to pause, and when to roll back.

Ship with the boring but essential runbook. Releases should pass quarantine evals and safety gates, carry a model card that explains capabilities and known risks, and include a rollback hash that on-call engineers can flip without debate. Observability should capture request and response traces including tool calls, token-level logs with sensitive data redacted, utilization and saturation metrics across the serving path, and real dollars per request. Define service-level objectives that reflect user experience—latency at P50/P95/P99, error rate, grounding precision, a safety incident budget—and add one business-flavored metric that keeps everyone honest: cost per helpful answer. If a guardrail trips or an SLO breaches, the canary rolls back automatically while you investigate; if behavior changes, the changelog explains what moved and why.

Post-training is never “done.” It’s a continuous control loop where behavior shaping, preference alignment, and hardening reinforce each other. Keep the loop tight, keep the signals clean, and keep the rollback switch close. That’s how a capable model turns into a product you can trust—day after day, release after release.

## PTQ: Post-training Quantization

At inference, LLMs are usually memory and bandwidth bound - weights and KV-cache traffic dominate latency and cost. PTQ is a method that trades precision for improved footprint and bandwidth. Recall that quantization shrinks numbers (weights/activations/KV-cache) from FP16/BF16 to fewer bits so that your model uses less memory and bandwidth at inference. For LLMs, bandwidth (moving weights/KV) if often the bottlenck, so fewer bits leads to fewer bytes moved, and therefore better latency and throughput without retraining. 

> The goal of PTQ is to cut inference cost by shrinking numbers into more compact bit representations, we turn FP16/BF16 weights and activations into low-bit integers _after_ we finish pre-training the model. 

1. Frantar, Elias, et al. "Gptq: Accurate post-training quantization for generative pre-trained transformers." arXiv preprint arXiv:2210.17323 (2022).
2. Wang, Hongyu, et al. "Bitnet: Scaling 1-bit transformers for large language models." arXiv preprint arXiv:2310.11453 (2023).
3. Ma, Shuming, et al. "The era of 1-bit llms: All large language models are in 1.58 bits." arXiv preprint arXiv:2402.17764 (2024).
4. Dettmers, Tim, et al. "Qlora: Efficient finetuning of quantized llms." Advances in Neural Information Processing Systems 36 (2024).