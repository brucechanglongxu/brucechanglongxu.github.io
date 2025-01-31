---
layout: post
title: "AlphaFold, RFDiffusion and ESM: Unravelling Nature's Deepest Secrets"
date: 2024-12-22
categories: AI oncology
author: Bruce Changlong Xu
---

Predicting the structure of biomolecular interactions has long been a formidable challenge in computational biology. With the advent of [AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w), DeepMind has once again redefined what is possible—expanding beyond proteins to accurately model complexes involving nucleic acids, small molecules, ions, and modified residues. This advancement is not merely an iteration of previous models; it represents a paradigm shift in structural biology. By incorporating diffusion-based deep learning, AlphaFold 3 surpasses specialized docking tools and nucleic acid predictors, demonstrating unprecedented accuracy in modeling biomolecular interactions. The implications are vast, from drug discovery to understanding cellular mechanisms at an atomic level. This post delves into how AlphaFold 3 achieves its groundbreaking accuracy, what it means for the future of computational biology, and how this single, unified deep-learning framework is paving the way for a new era in biomolecular modeling. Let's talk first talk about [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2). 

## AlphaFold and AlphaFold 3

The central goal of the AlphFold system is to **predict a protein's three dimensional structure based solely on its amino acid sequence**, which the system does with near experimental accuracy (this has been historically bottlenecked by laborious experimental detemrination of protein structure). **Levinthal's Paradox** states that a protein could theoretically explore _astronomical conformations_ before reaching its native folded state. In reality however, proteins fold in _milliseconds to seconds_, which suggests that there are **constraints and patterns** in fold that can be learned. Indeed, X-ray crystallography and Cryo-EM are slow and expensive, and to interrogate protein structure and folding is paramount towards the endeavour of designing potent therapeutics for patients. 

Prior to AlphaFold, two primary computational approaches have been used for protein structure prediction; AlphaFold merges and integrates both of these perspectives with AI:

1. _Physics-based approaches:_ Through molecular dynamics and statistical approximations but often leads to computaitons that are intractable. 
2. _Bioinformatics:_ Leveraging evolutionary history and sequence homology but faltering when homologous structures are unavailable. 

The backbone of the AlphaFold architecture is a _transformer-based_ model designed to predict 3D protein structures from amino acid sequences by integrating evolutionary, geometric and physical constraints. THere are three primary components: Input processing (MSAs and Templates), Core Model (Evoformer and Structure), Output (3D Structure Prediction and Confidence Scoring). Indeed, one of AlphaFold's major breakthroughs arises from how it **processes input data**. Unlike traditional protein modeling solutions that rely soley on phyiscs-based simulations, AlphaFold _learns directly from sequence and evolutionary data_ using deep learning, with the premise that **protein folding is heavily constraint by evolution**. 

_**What is a [Multiple Sequence Alignment (MSA)](https://www.nature.com/articles/s41598-019-56499-4)?**_ Proteins do not evolve randomly, instead functional constraints force certain amino acids to remain conserved. Some amino acids mutate together, meaning if one changes, another must also change to maintain stability. Hence evolution leaves behind a hidden pattern of co-evolution which can be mined from large sequence databases. A _[MSA (Multiple Sequence Alignment)](https://brucechanglongxu.github.io/oncology/2024/12/29/kinasesandmsa.html)_ is a _matrix representation_ of how similar proteins from different species align. Given a target protein sequence, we search large genetic databases such as Uniprot to find homologous sequences, and align them. The output is an alignment of many sequences with columns showing _evolutionary variation_ at each position. In short, it is a method used to align three or more biological sequences (DNA, RNA, protein) to identify regions of similarity that may indicate functional, structural, or evolutionary relationships. 

## ESM: Evolutionary Scale Modeling

**ESM-2** is the largest protein language model developed to data, which is trained on protein sequences with up to _15 billion parameters_. The training objective uses a masked language modeling approach (similar to BERT), where 15 percent of amino acids in a sequence are masked, and the model is trained to predict them. This forces the model to learn relationships between amino acids, evolutionary constraints, and structural patterns. As the model grows from 8M to 15B parameters, it is observed to learn increasing detailed protein structural features (where improvements are non-linear). 

Standard models like AlphaFold2 and RoseTTAFold rely on MSA's, whereas ESM-2's international representations learn structural features purely from sequences, eliminating the need for explicit evolutionary homologs. The model runs entirely on GPUs, 6x faster than AlphaFold2 for long sequences, and 60x for shorter sequences, and can also run on consumer hardware (e.g. Apple M1). 

**ESMFold** uses ESM-2's learned representations to predict _3D protein structures_ directly from a single amino acid sequence. It runs an order of magnitude faster than AlphaFold2, and does not require MSAs for structure predictions -- it only needs a single protein sequence. Whilst it is comparable to AlphaFold2 on seequences it understands well (low perplexity), it performs slightly worse for more challenging sequences. 

## RoseTTAFold and RFDiffusion

Traditional _de novo_ protein design involves engineering proteins to fold into specific structures and perform desired functions, such as binding to a target or acting as an enzyme. Deep learning has improved protein design but struggles with complex relationships between protein backbone geometry and sequences. **[RFDiffusion](https://www.nature.com/articles/s41586-023-06415-8)** extends the success of diffusion models from image/text to protein design, augmenting RoseTTAFold to a generative AI framework which is capable of designing diverse protein structures. 

**RoseTTAFold**

Unlike AlphaFold, which primarily focuses on pairwise relationships between residues, RoseTTAFold operates on a _three-track_ network that integrates **1D information** (sequence features), **2D information** (residue-residue relationships), **3D information** (atomic coordinates of protein structures). This simultaneous processing allows [RoseTTAFold](https://www.nature.com/articles/s41587-024-02395-w) to refine its predictions iteratively, which subsequently improves structural accuracy. Indeed RoseTTAFold is not only used for structure prediction, but also in _de novo protein design_, where it helps generate new proteins with specific structures and functions. 

**RFDiffusion** 

RFdiffusion uses a _denoising process_ to iteratively construct new protein backbones whilst maintaining biologically feasible structures, and adds an extra fine-tuning layer on top of RoseTTAFold. The core idea is to train a network to _gradually reverse a noising process_ that corrupts protein structures, allowing it to generate new, functional proteins from simple constraints. We first use a diffusion process where protein structures are gradually corrupted with Gaussian noise, the model then learns to reverse this noise, progressively refining a protein backbone from an initial random structure. 

- Watson, J.L., Juergens, D., Bennett, N.R. et al. De novo design of protein structure and function with RFdiffusion. Nature 620, 1089–1100 (2023). https://doi.org/10.1038/s41586-023-06415-8
- Lisanza, S.L., Gershon, J.M., Tipps, S.W.K. et al. Multistate and functional protein design using RoseTTAFold sequence space diffusion. Nat Biotechnol (2024). https://doi.org/10.1038/s41587-024-02395-w
- Jumper, J., Evans, R., Pritzel, A. et al. Highly accurate protein structure prediction with AlphaFold. Nature 596, 583–589 (2021). https://doi.org/10.1038/s41586-021-03819-2