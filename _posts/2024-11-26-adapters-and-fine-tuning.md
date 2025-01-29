---
layout: post
title: "Adapters, LoRA, QLoRA"
date: 2024-11-26
categories: AI
author: Bruce Changlong Xu
---

In recent years, the landscape of artificial intelligence (AI) has been transformed by foundation models capable of tackling complex, domain-specific tasks. While these models—such as GPT-4 in natural language processing or Evo in genomics—are powerful, they often require customization to perform optimally in niche applications. This is where adapters and other fine-tuning techniques come into play, offering efficient, scalable ways to tailor these models without the computational burden of retraining from scratch. This blog post explores the concepts of adapters and fine-tuning methods, using Evo, a genomic foundation model, as a concrete example to illustrate their application in advancing cardiovascular disease (CVD) research.

Fine-tuning is the process of adapting a pre-trained model to a specific task or dataset. While fine-tuning is widely used, traditional approaches often involve updating all the parameters of a model. For large models like Evo, which has 7 billion parameters, this can be computationally expensive and time-consuming. Moreover, fine-tuning on a new domain risks **catastrophic forgetting**, where the model loses knowledge of its original training data. Hence we run into two key issues with this methodology: 

- Full fine-tuning on all parameters requires immense computational power
- Risks overfitting, especially on small domain-specific datasets

In this blog post we will discuss several more efficient ways of fine-tuning that enable adaptation of large models without retraining them entirely from scratch. These are commonly referred to as _parameter efficient fine tuning_ methods (adapters, LoRA, QLoRA), which enables large models to adapt efficiently whilst keeping most of their parameters frozen. 

- **Adapters**

*Adapters* are lightweight neural network modules inserted between the layers of a pretrained model. Instead of updating all the model's parameters, adapters learn task-specific transformations while keeping the core model frozen. Each adapter layer consists of a down-projection (reducing dimensionality), a non-linear transformation, and an up-projection (back to the original dimensionality). During training, only the adapter layers are updated, whilst the rest of the model remains unchanged. 

Adapters are typically added *between layers* or *inside layers* of a pre-trained model, where each adapter consists of **down-projection** (reduces the dimensionality of the input features), **non-linearity** (applies an activation function for expressiveness) and **up-projection** (maps the reduced features back to the original feature space). Mathematically, this can be expressed as:

$$\mathcal{AO} = x + W_{up}\sigma(W_{down}x)$$

where $$W_{down}$$ (a small projection matrix that reduces dimensions), $$W_{up}$$ (a small expansion matrix that restores dimensions), $$\sigma$$ (activation function e.g. ReLU) and $$x$$ (original input to the adapter. The original model's parameters remain frozen, meaning they do not change during training. Only the parameters of the adapter are updated, drastically reducing the number of trainable parameters. For each downstream task, a separate adapter is trained and inserted into the model. The base model remains shared across tasks, and task-specific knowledge is encoded in the adapter. 

- **LoRA (Low-Rank Adaptation)**

LoRA is a technique that adds low-rank decomposition matrices to a model's layers, and introduces additional learnable parameters whilst keeping the original model frozen. It works by inserting low-rank matrices into the linear layers of a model, and the rank of these amtrices is much smaller than the dimensionality of the model's layers, reducing computational and memory overhead. 

LoRA is highly efficient, even for extremely large models, and the base model remains intact, which simplifies deployment. We can use LoRA to fine-tune Evo for predicting secondary RNA structures whilst retaining its ability to model other genomic tasks. 

```
from loralib import lora

model = lora.apply_lora(model, rank=8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

Let $$W$$ represent a pre-trained weight matrix in the model, instead of updating $$W$$ directly during fine-tuning, LoRA adds a low-rank adaptation:
$$W' = W + \Delta W$$ 
where $$\Delta W = A \times B$$ and

- $$A \in \mathbb{R}^{d \times r}$$, a small trainable matrix
- $$B \in \mathbb{R}^{r \times d}$$, another small, trainable matrix
- $$r$$ is the rank, a hyperparameter that controls the trade-off between memory savings and adaptation capacity

The original weight matrix $$W$$ is **frozen**, meaning it does not change during fine-tuning. Only the low-rank parameters $$A$$ and $$B$$ are trained, significantly reducing the number of updated parameters. The advantages of this approach is that LoRA drastically reduces the number of trainable parameters. For a large model, this can be orders of magnitude smaller than full fine-tuning. Instead of saving a new version of the entire model for each task, you only store the low-rank matrices $$A$$ and $$B$$. LoRA allows a single pre-trained model to adapt to many tasks efficiently by simply swapping out the low-rank matrices. Despite the parameter savings, LoRA often achieves nearly state-of-the-art performance compared to full fine-tuning. 

**Case Study 1: mRNA Design for Cardioprotective Proteins** 

In the scenario of fine-tuning Evo for cardiovascular disease, we could first define the tasks that we would like to focus on e.g. predicting RNA stability for sequences involved in angiogenesis (e.g. VEGF), generating siRNA sequences to silence fibrosis-related genes (e.g. TGF-B), and subsequenty choose which fine-tuning technique to use. We then incorporate domain-specific datasets (e.g. RNA sequences from CVD patients) and train with lightweight fine-tuning techniques to minimize computational costs. 

- We can start off with designing mRNA sequences encoding cardioprotective proteins (e.g. VEGF, FDF2) and validate their therapeutic effects in vitro and in vivo. 
- We can develop siRNA or antisense oligonucleotides targeting fibrosis-related genes and test their efficacy in reducing fibrotic markers. 

We can try to design mRNAs encoding cardioprotective proteins (e.g. NRG-1 for myocardial repair) to prevent chemotherapy induced cardiomyopathy. We can develop siRNA/ASOs to modulate inflammatory pathways linked to cardiovascualr toxicity (e.g. IL-6, TNF-a). We can also create RNA aptamers to neutralize circulating pro-inflammatory cytokines without systemic immune suppression. 

- Can RNA-based therapies enhance cardioprotection without reducing cancer treatment efficacy? 
- How can Evo-designed RNA sequences target biomarkers predicting cardiotoxicity (e.g. troponins, natriuretic peptides)? 
- Can Evo optimize RNA designs for patient-specific genetic risk factors (e.g. polymorphisms in SOD2 or RARG)? 
