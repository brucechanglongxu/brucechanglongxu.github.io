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

Evo also evaluates how sequence-level mutations influence biological function, fitness or regulatory interactions, and simulate mRNA sequence mutations to predict their effects on translation efficiency or immune evasion. We are able to co-design therapeutic mRNA and the corresponding target proteins, and understand RNA-protein interaction networks to develop RNA-targeting drugs. 

1. We can first assemble datasets of known mRNA therapeutics and their properties (translation efficiency, stability, immunogenicity), including ncRNA data for exploring regulatory roles in therapeutic design. 
2. We can then fine-tune a version of Evo on human-specific transcriptomes and synthetic mRNA libraries
3. We can then use Evo for optimizing codon sequences for therapeutic proteins, and designing mRNA with low immunogenicity by leveraging evolutionary patterns in untranslated regions (UTRs). 
4. We can then predict therapeutic efficacy using Evo's zero-shot capabilities, and validate predictions experimentally by synthesizing the mRNA and measuring protein output and immune response; ultimately designing mRNA vaccines targeting polygenic CVDs, and engineering RNA aptamers to bind and neutralize specific CVD-related targets. 

**Exact Steps** 

1. **Dataset Curation**: We are aiming to curate high quality CVD-specific datasets from GEO, GTEx, and other repositories, where we focus on high-confidence, experimentally validated sequences rather than excessively large datasets (coding regions of key CVD genes, miRNA/mRNA pairs, lncRNAs). We do not need large scale retraining as using Evo's base model as a pretrained foundation allows us to capture genomic-scale patterns. 
2. **Optimize Tokens**: We can use Evo's single-nucleotide tokenizer to preprocess sequences to remove redundancies (e.g. repetitive elements or low-complexity regions) to reduce computational overhead. 
3. **Intelligent Fine-tuning:** We do not need to retrain Evo entirely (this would be very inefficient), and instead we can fine-tune its embeddings or add lightweight layers (e.g. adapters) for mutatione ffect predictions (e.g. SNPs in CVD-related genes), RNA secondary structure stability prediction, and therapeutic RNA generation (mRNA, siRNA, aptamers). We can use methods like LoRA (low-rank adaptation) or prompt tuning to adjust a subset of Evo's parameters and drastically reduce compute costs; we can also freeze the lower layers of Evo that capture general genomic knowledge and fine-tuning only the higher layers for CVD-specific patterns. 
4. **Incremental Training:** We can start with shorter sequences (e.g. specific RNA regions or UTRs) before scaling to longer genomic contexts, and fine-tune for one task (e.g. mRNA design) before expanding to additional tasks (e.g. RNA-protein interactions). 

The synergy between RNA therapeutics and AI represents a new era in medicine—one where we can decode the molecular language of life and intervene with unprecedented precision. In the fight against cardiovascular disease, this partnership holds the promise of transforming care from reactive to proactive, enabling early detection, targeted interventions, and even disease prevention.

As researchers continue to unravel the mysteries of RNA and harness the power of AI, the dream of curing CVD—and perhaps many other complex diseases—moves closer to reality. The future is bright, and it speaks the language of RNA and algorithms. 

The most high-impact question that we could answer in the field with these tools is the following (that I could think of): 

**"How can RNA-based therapies be designed and optimized to reverse cardiac fibrosis and restore myocardial function in heart failure patients, using accessible datasets to identify and target key molecular drivers of fibrosis?"**

Cardiac fibrosis is a central, unresolved issue in heart failure, contributing to impaired cardiac function and progression to end-stage disease. Existing antifibrotic therapies are limited and often non-specific, creating an urgent need for precise molecular interventions. Targeting fibrosis-related pathways (e.g. TGF-B signaling, collagen production) using RNA-based therapies like siRNA, antisense oligonucleotides (ASOs) or mRNAs is emerging as a promising approach. There are publicly available datasets (e.g. GEO, GTEx, and dbGaP) which include transcriptomics from heart failure patients and cardiac fibroblasts. Tools like *Reactome, KEGG, STRING* provide data on fibrosis-related genes and pathways, and ample studies provide insights into fibrosis dynamics in animal models, which offers us benchmarks for validating RNA therapeutics. 

**References** 

- "Current RNA strategies in treating cardiovascular diseases", **Molecular Therapy**, Shirley Pei Shan Chia, Jeremy Kah Sheng Pang, Boon-Seng Soh, 2024
- "Sequence Modeling and Design from Molecular to Genome Scale with Evo", **Generative Genomics**, Eric Nguyen et. al, 2024
- "RNA secondary structure prediction using deep learning with thermodynamic regulation", **Nature Communications**, Kengo Sato et. al, 2021 (RNAfold) 
