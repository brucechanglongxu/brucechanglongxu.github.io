---
title: "Bruce Changlong Xu"
description: "AI, healthcare, and the road to superintelligence."
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

I currently support frontier AI training and inference systems at _Cerebras_, where we work with the world's most pioneering enterprises, governments, and institutions such as GlaxoSmithKline, Novo Nordisk, Perplexity, Mistral, G42 and other leaders to build transformative artificial intelligence.  

I have the privilege to support my talented friends at [3Cap AGI Partners](https://3cagi.vc/) to invest in the next generation of companies defining AGI infrastructure. Alongside them, I was fortunate to personally invest in [Synchron's Series D](https://synchron.com/), which is pioneering endovascular brain-computer interface technology for patients worldwide. As a member of _Asia Society's Dragon and Eagle Rising Leaders Council_, I am part of a wonderful community that helps to promote crucial dialogue between the East and the West in areas such as AI Governance and Oncology Clinical Trial Harmonization. 

I have previously worked at _Apple_ and _NVIDIA_, and completed my undergraduate and graduate studies in Mathematics and Computer Science at _Stanford University_, working with phenomenal researchers at [SAIL](https://ai.stanford.edu/), [HAI](https://hai.stanford.edu/), [Stanford Cardiovascular Institute](https://www.ahajournals.org/doi/10.1161/CIRCRESAHA.124.325652), and [Stanford Neurosurgery](https://ieeexplore.ieee.org/document/11058715). Earlier in my life, I was deeply involved Olympiad Mathematics, earning Gold/Silver medals at the British and Asian Pacific Mathematical Olympiads, and representing Hong Kong at the International Mathematical Olympiad as HKG01.

Together with cofounders, I started _Meirona_ a stealth platform building a global pervasive fabric defining multimodal AI infrastructure for science and medicine. Outside of work, I stay grounded through endurance running, music, and time with my border collie Aegis, and of course family. 

---

**Towards ASI: _A Rigorous Definition_**

> **Superintelligence:** An artificial system—whether a single model, a network of models, or a tightly coordinated agentic collective—whose general problem‑solving and strategic abilities across virtually all domains are decisively and sustainably superior to those of the best human experts and human organizations, enabling it to create new knowledge, plan and act autonomously in open‑ended environments, and (often, though not necessarily) improve its own capabilities.

From a more rigorous, axiomatic perspective, I propose that superintelligence can be defined with the following framework.
Let:
- $\mathcal{E}$ be a set of environments (simulated or real), each modeled as a partially observable, stochastic control process (e.g., POMDPs with continuous state/action for physical tasks).
- $\mathcal{T}$ be a task distribution over $\mathcal{E}$ (covering science, engineering, manipulation, navigation, multi-agent coordination, etc.).
- $\mathcal{R}$ be resource budgets: time, sample/episode count, energy, compute, sensors/actuators, and allowed external tools.
- $\Pi$ be the set of admissible policies/agents (human teams, institutions, and AI systems), each bound to the same $\mathcal{R}$.
- For task $e \sim \mathcal{T}$, let $U(\pi,e)$ be the utility/performance (cumulative reward, task success, profit, safety-weighted return, etc.), with penalties for unsafe actions, violations, or collateral harm.

_Normalized, resource-fair performance (resource-normalized score)._ 

$$
S(\pi) = \mathbb{E}_{e \sim \mathcal{T}}\left[U(\pi,e) \middle| \mathcal{R} \right]
$$
with the following langrangian to price resource use:
$$
\[
U'(\pi,e) = U(\pi,e) - \lambda \cdot \mathrm{Resources}(\pi,e)
\]
$$

% Human/institutional reference set
\paragraph{Human/institutional reference set.}
Let $\Pi_{\mathrm{H}}$ be the reference frontier of the best human experts and institutions under the same $\mathcal{R}$ (operationalized via champion baselines, world records, or certified human-team policies measured in the same testbed).

% Decisive, general dominance

\paragraph{Decisive, general dominance.}
An agent $\pi^\*$ is \emph{superintelligent (ASI)} w.r.t.\ $(\mathcal{T},\mathcal{R},U)$ if it Pareto-dominates $\Pi_{\mathrm{H}}$ across domains, not just on average:

1. \textbf{Average dominance:}
\[
S(\pi^\*) \;\ge\; (1+\delta)\, \max_{\pi \in \Pi_{\mathrm{H}}} S(\pi)
\quad \text{for some margin } \delta>0.
\]

2. \textbf{Breadth/coverage:}
Let $\{\mathcal{T}_k\}_{k=1}^K$ be a partition of $\mathcal{T}$ into disjoint task families (e.g., robotics, design, negotiation, scientific discovery). For at least a fraction $\beta$ of these families,
\[
\mathbb{E}_{e \sim \mathcal{T}_k}\!\big[ U(\pi^\*,e) \big]
\;\ge\; (1+\delta)\, \max_{\pi \in \Pi_{\mathrm{H}}}
\mathbb{E}_{e \sim \mathcal{T}_k}\!\big[ U(\pi,e) \big].
\]

3. \textbf{Robustness:}
The inequalities above hold under bounded distribution shift
\[
\mathsf{Shift}(\mathcal{T}\!\to\!\mathcal{T}') \;\le\; \epsilon
\]
and under perturbations to sensing/actuation noise within specified envelopes.

4. \textbf{Safety/constraints:}
$U$ includes constraint violations (harm, legal breaches) with large negative terms; dominance must hold without increasing expected harm above human baselines.

This makes “decisive and sustained superiority” into \textbf{$(\delta,\beta,\epsilon)$-dominance}, measurable and auditable.

% Practical metrics (optional, if you want to include)
\paragraph{Practical metrics.}
Dominance Ratio:
\[
D \;=\; \frac{S(\pi^\*)}{\max_{\pi \in \Pi_{\mathrm{H}}} S(\pi)}.
\]
Breadth Index:
\[
\mathrm{BI} \;=\; \frac{1}{K}\sum_{k=1}^K
\mathbf{1}\!\left[
\mathbb{E}_{e \sim \mathcal{T}_k}\!\big[ U(\pi^\*,e) \big]
\;\ge\; (1+\delta)\, \max_{\pi \in \Pi_{\mathrm{H}}}
\mathbb{E}_{e \sim \mathcal{T}_k}\!\big[ U(\pi,e) \big]
\right].
\]

Robustness radius: the largest $\epsilon$ for which the dominance conditions hold.
Safety gap: the difference in incident rates vs.\ the human frontier at equal performance.

### A Hitchhiker's Timeline of Modern-day AGI

Modern AI rides on more than 70 years of compounding technologies. From the 1947 transistor and 1958 integrated circuit to the 1971 microprocessor and deep-submicron CMOS. This foundation led to the invention fo the modern day AI processor, and in 2006 CUDA made GPUs generally programmable - in 2010-1015, Fermi, Kepler and Maxwell normalized massive parallelism and device memory hierarchies within and across GPUs, to train the intelligence we now call AI. The art of intimate "co-design" between hardware and software evolved to make sure that AI was being fed with meaningful work and data - in 2017 the "Tensor Core" (and corresponding software advances such as CUTLASS) was introduced that revolutionized the possibility of GEMMs (general matrix multiplies) at global economic scale, alongside innovations such as the TPU, Cerebras Wafer and more. Transformers became important enough that entire cores were introduced to accelerate this particular workload (H100s introduced "Transformer Engines" in FP8 to truly optimize attention and MLP). cuDNN, NCCL, Horovod industrialized multi-GPU training; leading to innovations such as Megatron, DeepSpeed that enabled tensor, pipeline, data parallel training paradigms to complement sharding (ZeRO/FSDP) techniques, activation checkpointing and optimizer state partitioning - trillion parameter models could perceivably be accomplished even in light of the memory wall. Every processor could be individually tightly controlled with JAX, Triton and custom-kernels (e.g. FlashAttention), and orchestrated alongside its peers with these new techniques. 

Once compute became abundant, models proliferated in what some called a "cambrian explosion". In 2012, Krizhevsky et al invented AlexNet that used ReLU, dropout and fine-grained control over GPUs to catalyze the initial sparks of representation learning at scale; Then in 2015, He Kaiming and his team's ResNet demonstrated that we could truly train "deep" neural networks through the novel idea of skip connections that allowed us to bypass and communicate across longer ranges between layers, without information decay. In 2016, AlphaGo demonstrated the power of reinforcement learning. In 2017, recurrent neural networks matured to the modern day transformer, which set fire to modern day pre-training, post-training and infrastructure paradigms governed by beautiful scaling laws - delivering BERT in 2018, GPT-2/3 in 2019/20 and multimodal models that could deeply connect representations of images, text, voice and indeed the perceivable universe of digital data. CLIP fusioned vision and language through a contrastive learning paradigm for zero-shot transfer, and we began to create a wave of "generative AI" applications that could design new proteins for human health through the exact same diffusion mechanism that created imaginative text to image tools. With DeepSeek's R1 paper, RLHF, SFT and PPO across mixtures of experts began to take center stage, and the community began to think deeply about to train and serve models more efficiently, and increase the "density" of intelligence per token/weight. vLLM, QLoRA, GPTQ and speculative decoding used student-teacher, distillation, quantization and paged-caching paradigms to deliver world-class performance at scale and new alignment paradigms such as DPO and RLAIF emerged; alongside a booming interest in "agentic" and autonomous systems with multi-step reasoning and action. 

The same scaling narrative is now playing out in _world models_ - systems that learn latent dynamics so they can _predict_, _plan_, and _act_ in the real physical world. How do we create policies, and use spatiotemporal encoders (with Transformer backbones) to reason over compact latent spaces, and close the Sim2Real gap? LeCunn's JEPA paradigm and new data-hungry Vision-Language-Action structures are beginning to take form to truly, deeply and meaningfully embed and embody advanced machine intelligence in the real world. 

Threaded through all of these advancements is the deeply _"human layer"_ that turns capability into dependable tools, through domain-specific post-training the encodes policy, ontology, and constraints. It is paramount to work towards AGI that incrementally widens what is routine; models that read, cite, plan and excute; agents that coorperate under governance; AI training, evaluation and deployment stacks that do not collapse under the auspice of low-quality and potentially adversarial data. From science and medicine to technology and education, codified intelligence is now reshaping and redefining human flourishing. 

If you want to dig deeper into the foundations of AGI/ASI, feel free to read my five principles [data foundations](https://brucechanglongxu.github.io/posts/vagelos-series-v-infectious-disease-and-microbiology/), [infrastructure](https://brucechanglongxu.github.io/posts/ai-infrastructure/), [pretraining](https://brucechanglongxu.github.io/posts/ai-pretraining/), [posttraining](https://brucechanglongxu.github.io/posts/ai-posttraining/) , [applications](https://brucechanglongxu.github.io/posts/ai-applications/). Enjoy!

### From Serendipity to Specificity

The modern arc of medicine is a sequence of precise bets on biology that became therapies -- and then infrastructures. _Penicillin_ leaked from a contaminated plate and revealed that B-lactams sabotage bacterial cell-wall cross-linking - an Achilles' heel unique to microbes. A generation later, ulcers, once blamed on stress adn acid, yielded to _omeprazole_: a prodrug that accumulates in parietal-cell canialiculi and, in its sulfenamide form, irrversibly ties up the H+/K+-ATPase. What started as empirical symptom control became mechanism capture - find the essential machine, jam it, and let physiology recover. 

Consider _Abraxane_. Paclitaxel was a potent anti-mitotic, but the original formulation rode in Cremophor EL, a solvent that caused hypersensitivity reactions and constrained dosing. The bet behind Abraxane was simple and bold: bind paclitaxel to _albumin nanoparticles_ so the body's own transport pathways (gp60/caveolae, SPARC-rich stroma) ferry drug into tumors, no solvent required. The result was a formulation that could be dosed more efficiently and combined more cleanly - firs tin metastatic breast cancer, then non-small cell lung cancer, and ultimately pancreatic cancer with gemcitabine - showing how delivery engineering can turn an existing mechanism into a better medicine. 

Now _Imatinib (Gleevec)_ - the canonical targeted therapy. A strange cytogenetic signature in chronic myeloid leukemia (the _Philadelphia chromosome_, t(9;22)) fused BCR and ABL into a constitutively active kinase. Instead of blunter chemotherapy, the leap was to design a small molecule that inhibits BCR-ABL's ATP site, collapsing the malignant program while largely sparing healthy hematopoiesis. Patients who previously faced a near-certain trajectory to blast crisis suddenly had durable remissions with an oral agent; later, the smae drug transformed GIST by inhibiting mutant KIT. Whilst resistance mapped to on-target mutations (e.g. T315I "gatekeeper"), this was quickly addressed by next-generation TKIs (dasatinib, nilotinib) and then ponatinib for T315I. Imatinib proved a principle: when you can measure the lesion and hit it cleanly, outcomes change orders of magnitude - find a driver, drug the pocket, track the resistance. 

Soon aftter, lung cancer revealed a tapestry of kinase addictions. In never-smokers, EGFR exon 19 del/L858R mutations replaced the smoker's mutational burden as the primary engine. Reversible TKIs (gefitnib/erlotinib) gave way to osimertinib for T970M and CNS activity; exon 20 insertions required different geometries (amivantamab, mobocertinib). Parallel discoveries followed: ALK fusions (EML4-ALK) opened a path from crizotinib to alcetinib/brigatinib to lorlatinib for solvent-front mutations. 

Every blockbuster here sits on the same fundamental principles: identify a causal lesion [^1], a tractable binding site [^2] or circuit node, a biomarker to aim and measure, and a development path that respects resistance and context. Some wins came from single, clean addictions (BCR-ABL, BRAF V600E), whilst others came from surface addresses (HER2, Trop-2) or systems levers (VEGF, androgen receptor). The craft is "knowing which rope to pull" and what tends to fray when we do. Ultimately we see that blockbusters endure when four ingredients align simultaneously:

1. A lesion the tumor cannot route around cheaply (strong dependency or a shared choke)
2. A modality matched to the biology (pocket binder, degrader, delivery vector, radioligand, immune lever)
3. Measurable guidance at the bedside (companion Dx, PD readouts, ctDNA to navigate resistance)
4. Room for combinations that address the predictable failure modes (vertical/horizontal, modality-switch, CNS-aware)

We pull on the rope that actually moves tumor fitness and a tolerable window, instrument the line so we can see tension (biomarkers), and we pack the splices that we will need when fibers snap (resistance). This is why imatinib, osimertinib, BRAF+MEK, CDK4/6 and endocrine therapy, PARP in HRD, BTK and venetoclax and HER2 ADCs each became - and remain - defining therapies. 

I'll stop there because the rabbit hole never ends, if you want to learn more about this space read my posts on [oncology](https://brucechanglongxu.github.io/posts/towards-oncological-superintelligence/), [immunology](https://brucechanglongxu.github.io/posts/vagelos-series-iii-immunological-intelligence/), [neuroscience/neurosurgery](https://brucechanglongxu.github.io/posts/vagelos-series-iv-neuroscience-at-the-edge/), [infectious disease/global health](https://brucechanglongxu.github.io/posts/vagelos-series-v-infectious-disease-and-microbiology/) and [cardiometabolic medicine](https://brucechanglongxu.github.io/posts/cardiometabolic-supercycle/).

If you have read this far, thank you, and welcome to my blog. Enjoy!

[^1]: A causal lesion is a change in the tumor (mutation, fusion, amplification, epigenetic rewiring, microenvironmental dependence) that the cancer _needs_ to maintain growth or survival. You know that it is causal when three lines of evidence align 1. _Genetic necessity:_ Silencing the node (CRISPR/RNAi) collapses viability in lesions that harbor it (e.g. BCR-ABL in CML; EGFR exon19/L858R in NSCLC; KIT in GIST) 2. _Pharmacologic sufficiency:_ A selective inhibitor reproduces that "genetic kill" in models and patients. 3. _On-target resistance:_ When the tumor escapes, it does so by mutating the same node (e.g. ABL T315I, EGFR T790M/C797S, ALK G1202R), proving that we were sitting on the right lever. If we only see the correlation (the lesion is present) without necessity (silencing does not hur tthe cell) or without on-target resistance, we are likely looking at a passenger, and not the engine or driver.
[^2]: There are two ways to drug the same biology. The first is through the _binding site (structure)_ - we physically occupy a pocket that is orthosteric (e.g. ATP site of kinases like imatinib and osimertinib) or allosteric (IDH1/2, SHP2). We win on affinity, selectivity, and residence time. Covalent designs (KRAS G12C, osimertinib) trade higher specificity for lasting occupancy. The second is through a _circuit node (control)_ - we intervent at a control point of a pathway, even if no single mutation sits there. VEGF/VEGFR and androgen receptor are the archetypes: tumors depend on vasacular supply or androgen signaling irrespective of one specific DNA change. Here we win by picking the rung in the cascade with the highest control coefficient (where a unit perturbation produces the largest drop in proliferative or angiogenic output) (this is sort of like choosing the largest descent step in training over a loss curve!) and a tolerable therapeutic window. 