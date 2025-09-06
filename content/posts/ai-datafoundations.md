---
title: "Real world ASI I: Data Foundations"
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


Real-world ASI starts with deliberately engineered data. Write a capability spec first: target tasks, tools the model must call, modalities, latency and safety constraints. From that, draft a mixture plan—what sources you’ll use, in what proportions, on what refresh cadence—and list the gaps you must fill: rare failure modes, non-English domains, long-tail workflows, tool-use traces, time drift. The corpus should teach the behaviors you need, mirror the environment you’ll deploy into (edge cases included), and be clean to ship from a licensing and privacy standpoint.

Stand up a continuous pipeline rather than a one-off ETL. Ingest → normalize → fingerprint (near-dup & family detection) → quality scoring (heuristics + small judges) → PII/PHI handling → semantic clustering → mixture balancing → sharding/streaming. Report effective tokens, not just raw count: $$Neff=N (1−d) (1−n) (1−c)N_{\text{eff}} = N\,(1-d)\,(1-n)\,(1-c)Neff​=N(1−d)(1−n)(1−c)$$, where ddd is the near-duplicate fraction, nnn the estimated noise/unlearnable share, and ccc contamination with evals or restricted licenses. Close the loop with active learning (uncertainty/diversity sampling from training traces), targeted synthetic/simulated data to plug coverage holes, and mixture controllers that re-balance when you see loss spikes or mode collapse. Stream from object stores with deterministic sharding and reproducible snapshots so runs are replayable and auditable.

Ship governance and telemetry with the data. Define a multimodal schema (text/code/image/audio/sensor/EHR) that carries lineage, consent/license tags, and retention windows. Publish a Data Card for each release (sources, proportions, filters, known risks), maintain a living risk register (privacy, bias, safety), and wire online signals—safety incidents, hallucination clusters, user-reported gaps—into the backlog for the next cut. Treat deletion as a first-class feature: fast traceability from sample → shards → checkpoints for “right to be forgotten.” Keep red-team corpora and adversarial prompts as training assets, not side files. Finally, set SLOs for the pipeline itself—freshness, audit latency, NeffN_{\text{eff}}Neff​/week, and cost per curated token—so Data Foundations runs with clear budgets and accountability.