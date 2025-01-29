---
layout: post
title: "Fine Tuning, your Grace"
date: 2025-01-29
categories: Software Engineering
author: Bruce Changlong Xu
---

Mastering fine-tuning can push one's skillset from an ML Engineer to an AI Research Scientist. First we will look at _Direct Preference Optimization_. Fine-tuning large language models (LLMs) to align with human preferences is a crucial step in making them more useful, safe, and controlled.  Historically, **Reinforcement Learning from Human Feedback (RLHF)** has been the go-to method for this process. However, RLHF comes with significant computational and stability challenges, requiring reward modeling and complex reinforcement learning (RL) techniques like _Proximal Policy Optimization (PPO)_. A new paradigm, _Direct Preference Optimization (DPO)_, proposed by Rafael Rafailov and colleagues at Stanford, offers a compelling alternative. In this blog post we will explore the paper's key insights, and highlight how DPO eliminates the need for explicit RL, whilst achieving superior or comparable performance in aligning LLMs with human preferences. 

**The Problem with RLHF** 

LLMs acquire broad knowledge and reasoning abilities through unsupervised pre-training. However, their training data contains conflicting viewpoints, misleading information, and varying levels of quality. To refine their outputs, _preference-based fine-tuning_ is applied, where models learn from human feedback on generated responses. RLHF, the dominant approach, involves three steps:

- **Supervised Fine-Tuning (SFT):** The base model is trained on human-written high-quality examples. 
- **Reward Modeling:** A separate model is trained to predict human preference scores based on a dataset of ranked model outputs.
- **Reinforcement Learning (RL) with PPO:** The model is further fine-tuned to maximize reward whilst staying close to the original model distribution. 

The _reward model_ assigns a scalar reward to a given response, which is then used to fine-tune the policy. Training the reward model involves optimizing over a pairwise ranking loss: 

$$L_{REWARD}(\phi) =  \mathbb{E}_{(x, y^A, y^B) \sim D} \log \sigma(r_{\phi}(x, y^A) - r_{\phi}(x, y^B))$$

where $$r_{\phi}(x, y)$$ is the reward model parameterized by $$phi$$ and $$\sigma(z) = \frac{1}{1 + e^{-z}}$$ is the sigmoid function. $$D$$ is the dataset of human-labeled comparisons; this loss function ensures that **preferred responses receive higher rewards**. Once the reward model is trained, it guides the policy optimization using RL -- PPO is used as the core RL algorithm due to its stability and efficiency in optimizing large scale models. 

A policy $$\pi_{\theta}(y|x)$$ is optimizezd to maximize the expected rewards:

$$J(\theta) = \mathbb{E}_{(x, y) \sim \pi_{\theta}}[r_{\phi}(x, y)]$$

rather than directly optimizing this however, PPO introduces a "clipped surrogate loss" to prevent overly large/destabilizing updates (as mentioned in our previous blog post - see Deep Dive into DeepSeek R1 for explicit loss). Essentially, this new clipped objective ensures that updates to the policy do not drastically alter its behavior. To ensure that the policy does not diverge too far from its pretrained state, an additional KL-divergence penalty is introduced:

$$L_{KL}(\theta) = \beta \mathbb{E}_{(x, y) \sim \pi_{\theta}}[D_{KL}(\theta_{\pi}(y|x) || \pi_0(y | x))]$$

where $$\beta$$ controls the regularization strength. This ensures that the fine-tuned model retains generalization capabilities whilst aligning with human preferences. Despite PPO's success in RLHF however, it tends to reward overfitting (the policy may exploit the reward model rather than truly align with human preferences), it is expensive due to involved policy gradient computations, and it requires careful hyperparameter tuning (which can get involved depending on the size of the model). 

Whilst RLHF has produced impressive results in aligning AI behavior, it is computationally expensive, unstable and sensitive to hyperparameters. DPO challenges the necessity of RL in RLHF by introducing a _closed-form optimal policy update_. The core insight is that the RLHF objective (reward maximization with a KL-divergence constraint) can be directly optimized as a simple _binary classification loss_. Instead of training an explicit reward model and optimizing it via RL, DPO reparameterizes the reward function and directly optimizes the model's likelihood of preferred responses relative to dispreferred ones. The DPO oibjective can be expressed as:

$$L_{DPO}(\theta) = -\mathbb{E}_{(x, y^A, y^B) \sim D} \log \frac{\pi_{\theta}(y^A | x)}{\pi_{\theta}(y^A | x) + \pi_{\theta}(y^B | x)}$$

Indeed, DPO implicitly learns the reward funciton within the policy update, removing the need to train and fine-tune a separate reward model. Unlike PPO, which requires sampling and reward estimation at each step, DPO simply modifies the probability ratios of preferred vs. dispreferred responses. By reducing the optimization to a classification problem, DPO avoids the instabilities of RL methods; it is computationally lightweight and requires minimal hyperparameter tuning compared to PPO-based RLHF. Indeed the paper demonstrates that DPO achieves the highest reward at every KL-divergence level, surpassing PPO-based methods, and outperforms PPO in generating high-quality summaries as judged by GPT-4 (whilst requiring less computation). Now, it is a matter of investigating whether DPO can scale to larger LLMs (e.g. GPT-4 scale), multimodal models (e.g. VLMs). 