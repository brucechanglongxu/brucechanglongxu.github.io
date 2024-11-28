---
layout: post
title: "Adapters, LoRA and Fine-tuning"
date: 2024-11-26
categories: RNA Therapy
author: Bruce Changlong Xu
---

In recent years, the landscape of artificial intelligence (AI) has been transformed by foundation models capable of tackling complex, domain-specific tasks. While these models—such as GPT-4 in natural language processing or Evo in genomics—are powerful, they often require customization to perform optimally in niche applications. This is where adapters and other fine-tuning techniques come into play, offering efficient, scalable ways to tailor these models without the computational burden of retraining from scratch. This blog post explores the concepts of adapters and fine-tuning methods, using Evo, a genomic foundation model, as a concrete example to illustrate their application in advancing cardiovascular disease (CVD) research.

Fine-tuning is the process of adapting a pre-trained model to a specific task or dataset. While fine-tuning is widely used, traditional approaches often involve updating all the parameters of a model. For large models like Evo, which has 7 billion parameters, this can be computationally expensive and time-consuming. Moreover, fine-tuning on a new domain risks **catastrophic forgetting**, where the model loses knowledge of its original training data.

- Full fine-tuning on all parameters requires immense computational power
- Risks overfitting, especially on small domain-specific datasets

In this blog post we will discuss several more efficient ways of fine-tuning that enable adaptation of large models without retraining them entirely from scratch. The first is that of **adapters**. 

1. Adapters

*Adapters* are lightweight neural network modules inserted between the layers of a pretrained model. Instead of updating all the model's parameters, adapters learn task-specific transformations while keeping the core model frozen. Each adapter layer consists of a down-projection (reducing dimensionality), a non-linear transformation, and an up-projection (back to the original dimensionality). During training, only the adapter layers are updated, whilst the rest of the model remains unchanged. 

2. LoRA (Low-Rank Adaptation)

LoRA is a technique that adds low-rank decomposition matrices to a model's layers, and introduces additional learnable parameters whilst keeping the original model frozen. It works by inserting low-rank matrices into the linear layers of a model, and the rank of these amtrices is much smaller than the dimensionality of the model's layers, reducing computational and memory overhead. 

LoRA is highly efficient, even for extremely large models, and the base model remains intact, which simplifies deployment. We can use LoRA to fine-tune Evo for predicting secondary RNA structures whilst retaining its ability to model other genomic tasks. 

```
from loralib import lora

model = lora.apply_lora(model, rank=8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

In the scenario of fine-tuning Evo for cardiovascular disease, we could first define the tasks that we would like to focus on e.g. predicting RNA stability for sequences involved in angiogenesis (e.g. VEGF), generating siRNA sequences to silence fibrosis-related genes (e.g. TGF-B), and subsequenty choose which fine-tuning technique to use. We then incorporate domain-specific datasets (e.g. RNA sequences from CVD patients) and train with lightweight fine-tuning techniques to minimize computational costs. 

- We can start off with designing mRNA sequences encoding cardioprotective proteins (e.g. VEGF, FDF2) and validate their therapeutic effects in vitro and in vivo. 
- We can develop siRNA or antisense oligonucleotides targeting fibrosis-related genes and test their efficacy in reducing fibrotic markers. 

**Case Study 1: mRNA Design for Cardioprotective Proteins** 

We can try to design mRNAs encoding cardioprotective proteins (e.g. NRG-1 for myocardial repair) to prevent chemotherapy induced cardiomyopathy. We can develop siRNA/ASOs to modulate inflammatory pathways linked to cardiovascualr toxicity (e.g. IL-6, TNF-a). We can also create RNA aptamers to neutralize circulating pro-inflammatory cytokines without systemic immune suppression. 

- Can RNA-based therapies enhance cardioprotection without reducing cancer treatment efficacy? 
- How can Evo-designed RNA sequences target biomarkers predicting cardiotoxicity (e.g. troponins, natriuretic peptides)? 
- Can Evo optimize RNA designs for patient-specific genetic risk factors (e.g. polymorphisms in SOD2 or RARG)? 
