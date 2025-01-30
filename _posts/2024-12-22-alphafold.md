---
layout: post
title: "AlphaFold, RFDiffusion and ESM: Unravelling Nature's Deepest Secrets"
date: 2024-12-22
categories: Software Engineering
author: Bruce Changlong Xu
---

Predicting the structure of biomolecular interactions has long been a formidable challenge in computational biology. With the advent of [AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w), DeepMind has once again redefined what is possible—expanding beyond proteins to accurately model complexes involving nucleic acids, small molecules, ions, and modified residues. This advancement is not merely an iteration of previous models; it represents a paradigm shift in structural biology. By incorporating diffusion-based deep learning, AlphaFold 3 surpasses specialized docking tools and nucleic acid predictors, demonstrating unprecedented accuracy in modeling biomolecular interactions. The implications are vast, from drug discovery to understanding cellular mechanisms at an atomic level. This post delves into how AlphaFold 3 achieves its groundbreaking accuracy, what it means for the future of computational biology, and how this single, unified deep-learning framework is paving the way for a new era in biomolecular modeling. Let's talk first talk about [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2). 

The central goal of the AlphFold system is to **predict a protein's three dimensional structure based solely on its amino acid sequence**, which the system does with near experimental accuracy (this has been historically bottlenecked by laborious experimental detemrination of protein structure). Prior to AlphaFold, two primary computational approaches have been used for protein structure prediction; AlphaFold merges and integrates both of these perspectives with AI:

1. _Physics-based approaches:_ Through molecular dynamics and statistical approximations but often leads to computaitons that are intractable. 
2. _Bioinformatics:_ Leveraging evolutionary history and sequence homology but faltering when homologous structures are unavailable. 