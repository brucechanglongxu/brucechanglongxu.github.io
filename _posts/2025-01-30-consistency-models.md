---
layout: post
title: "Consistent Intelligence"
date: 2025-01-28
categories: Software Engineering
author: Bruce Changlong Xu
---

The field of generative AI in video has long been dominated by diffusion models, celebrated for their ability to create high-quality images through a stepwise denoising process. However, their Achilles' heel has always been efficiency -- requiring hundreds or even thousands of iterative steps to produce a single sample. _Consistency Models_ represent a transformative leap forward, offering a means to generate high fidelity images in a single step whilst retaining the flexibility of multi-step refinement. 

Diffusion models operate by progressively refining noise into meaningful structure; whilst this iterative approach yields exceptionals ample quality, it is computationally expensive and impractical for real-time applications. Researchers have explored techniques such as distillation and acceleration strategies, but these methods often come with trade-offs in sample quality or require significant retraining efforts. Yang Song et al. introduced **Consistency Models**, which is a novel class of generative models that address the efficiency limitations of diffusion models. Unlike their predecessors, CMs learn a direct mapping from noise to data in a single forward pass. This is achieved by enforcing _self-consistency_ i.e. ensuring that the model's outputs remain invariant across different noise levels, allowing it to converge to a high-fidelity image without iterative refinement. 