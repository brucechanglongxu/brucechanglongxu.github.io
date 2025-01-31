---
layout: post
title: "Consistent Visual Perception"
date: 2025-01-30
categories: AI vision
author: Bruce Changlong Xu
---

The field of generative AI in video has long been dominated by diffusion models, celebrated for their ability to create high-quality images through a stepwise denoising process. However, their Achilles' heel has always been efficiency -- requiring hundreds or even thousands of iterative steps to produce a single sample. _Consistency Models_ represent a transformative leap forward, offering a means to generate high fidelity images in a single step whilst retaining the flexibility of multi-step refinement. 

Diffusion models operate by progressively refining noise into meaningful structure; whilst this iterative approach yields exceptionals ample quality, it is computationally expensive and impractical for real-time applications. Researchers have explored techniques such as distillation and acceleration strategies, but these methods often come with trade-offs in sample quality or require significant retraining efforts. Yang Song et al. introduced **Consistency Models**, which is a novel class of generative models that address the efficiency limitations of diffusion models. Unlike their predecessors, CMs learn a direct mapping from noise to data in a single forward pass. This is achieved by enforcing _self-consistency_ i.e. ensuring that the model's outputs remain invariant across different noise levels, allowing it to converge to a high-fidelity image without iterative refinement. 

At their core, Consistency Models leverage insights from probability flow ODEs used in diffusion models. Instead of relying on multiple refinement steps, a CM is trained to satisfy a consistency condition along the generative trajectory. This means that whether the model starts from pure noise or an intermediate state, it converges to the same final output. Unlike diffusion models, which require many denoising iterations, CMs can generate high-quality samples in a single forward pass. Whilst capable of one-step generation, CMs can also support multi-step sampling, allowing a trade-off between speed and sample fidelity. Tasks like inpainting, super-resolution, and stroke-guided editing can be performed without additional retraining, making CMs a versatile one-shot generative tool. 

In diffusion-based generative models, a stochastic differential equation describes how noise is gradually transformed into data. The probability flow ODE is a deterministic counterpart to this process:

$$dx = f(x, t) dt$$ 

where $$f(x, t)$$ is the probability flow vector field derived from the score function:

$$f(x, t) = s_{\theta}(x, t) - \frac{1}{2} \nabla \cdot s_{\theta}(x, t)$$

here $$s_{\theta}(x, t)$$ is the score function, which estimates the gradient of the log density function. This describes how an image is gradually refined along a continuous path from noise to data. Indeed, rather than refining an image iteratively like traditional diffusion models, _consistency models_ enforce a **self-consistency constraint** that allows direct mapping from noisy input to the final output:

$$\mathbb{E}_{p(x_t : x_0)}[C_{\theta}(x_t, t)] = x_0$$

this equation means that at any given itme step $$t$$, the model learns to output the original clean image $$x_0$$, regarless of the noise $$x_t$$. By training the model to satisfy this condition, CMs effectively compress the iterative denoising process into a single-step transformation. Training CMs require enforcing _consistency constraints_ during the learning process; this is done through the following loss function:

$$\mathcal{L}(\theta) = \mathbb{E}_{p(x_t, x_0)}[||C_{\theta}(x_t, t) - x_0||^2]$$

this ensures that for any intermediate state $$x_t$$, the model learns to reconstruct the original data point $$x_0$$ accurately. Hence this allows consistency models to generate high quality images in **one step** whilst still supporting multi-step sampling for better quality. This direct mapping makes CMs more computationally efficient than diffusion models. 

**Appendix**

Two primary training approaches have emerged:

1. **Distillation:** A pre-trained diffusion model generates training pairs of intermediate states and their final outputs, the CM learns to reconstruct the final output regardless of the initial noise level, distilling the iterative process into a direct mapping. 
2. **Direct Training:** Rather than relying on an existing diffusion model, CMs can be trained from scratch by solving a self-consistency constraint using a unbiased score-matching estimator. 

