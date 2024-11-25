---
layout: post
title: "RNA Therapeutics and Artificial Intelligence"
date: 2024-11-25
categories: RNA Therapy
author: Bruce Changlong Xu
---

Despite the excitement, challenges remain. Understanding the interplay between different RNA modifications, optimizing delivery systems, and ensuring safety in clinical applications require further research. AI will be instrumental in overcoming these hurdles by enabling simulations, refining experimental designs, and predicting long-term outcomes. Moreover, the ethical implications of RNA editing and AI-driven personalization must be carefully addressed. Transparent regulations and interdisciplinary collaboration will be critical to ensuring that these technologies benefit all patients equitably.

**Evo** 

A beautiful work resulting form a collaboration between Stanford Computer Science and the Arc Institute, __Evo__ leverages **StripedHyena**, a hybrid architecture combing deep signal processing and limited attention layer sto handle long genomic sequences (of context length up to 131,072 tokens) with single-nucleotide resolution; enabling us to design long mRNA transcripts tailored for specific therapeutic proteins or regulatory pathways, and predict how mutations or edits in mRNA sequences affect protein stability and function. Evo is pretrained on OpenGenome (300B nucleotides across diverse genomes) and improves with larger datasets and model sizes, making it suitable for very large RNA sequence databases like transcriptomes.

- We are able to predict the impact of mutatiosn on protein coding/non-coding RNA (ncRNA) sequences without additional fine-tuning in a zero-shot fashion. 
- We are able to predict stability, translation efficiency and codon optimization to improve therapeutic mRNA constructs. 
- We are able to integrate RNA-specific metrics like minimum free energy to ensure mRNA secondary structure stability. 

We are able to generate **synthetic sequences** that mimic natural ones whilst incorporating novel functional elements, and generate optimized mRNA sequences encoding therapeutic proteins or vaccines; as well as designing RNA aptamers or CRISPR RNA guides for gene-editing therapies. 

The synergy between RNA therapeutics and AI represents a new era in medicine—one where we can decode the molecular language of life and intervene with unprecedented precision. In the fight against cardiovascular disease, this partnership holds the promise of transforming care from reactive to proactive, enabling early detection, targeted interventions, and even disease prevention.

As researchers continue to unravel the mysteries of RNA and harness the power of AI, the dream of curing CVD—and perhaps many other complex diseases—moves closer to reality. The future is bright, and it speaks the language of RNA and algorithms. 

**References** 

- "Current RNA strategies in treating cardiovascular diseases", __cellpress__ 
