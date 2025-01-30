---
layout: post
title: "AGI: Apple General Intelligence"
date: 2024-12-09
categories: Software Engineering
author: Bruce Changlong Xu
---

Apple around half a year ago introduced its foundation language models as part of the Apple Intelligence ecosystem, marking a significant milestone in AI development. The AI models are designed for both on-device and cloud-based processing, emphasizing efficiency, privacy, and user-centric AI applications. A standout feature is their focus on _privacy_ and _responsible AI_ (similar to Anthropic). Unlike many of their competitors, Apple ensures that user data is not used in training, and implements Private Cloud Compute to handle more tasks securely. Their AI system consists of two main foundation models:

1. **AFM-on-device:** A ~3 billion parameter model optimized for local processing, ensuring fast and secure interactions. 
2. **AFM-server:** A more powerful model running in the cloud, handling complex computations while maintaing privacy protections.

Apple's foundation models build upon a (decoder-only) transformer backbone, with several key architectural refinements. They employ **Grouped-Query Attention (GQA)** instead of standard multi-head self-attention (MHSA), reducing computational overhead whilst maintaing expressivity. This leads to faster inference time (30 percent less computation versus full MHSA), lower memory footprint, and retains competitive performance in reasoning tasks. In vanilla self-attention, each input token attends to all others using Query (Q), Key (K) and Value (V) projections, computed as follows:

$$Q = XW_Q, K = XW_K, V = XW_V$$ 

where $$x \in \mathbb{R}^{T \times d}$$ is the embedded input sequence of tokens (input sequence of length T, embedding size d), and $$W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$$ are learnable weight matrices. The attention scores are then computed:

$$A = \frac{QK^T}{\sqrt{d_k}}$$

where $$d_k$$ is the per-head key dimension, and we subsequently apply softmax normalization to obtain the weighted values:

$$\textbf{Attention}(Q, K, V) = \textbf{softmax}(A) \cdot V$$

and subsequently apply multi-head projection:

$$\textbf{MHSA}(X) = \textbf{Concat}(\textbf{head}_1, \cdots, \textbf{head}h) \cdot W_)$$

where $$W_O$$ is the output projection. This leads to powerful feature extraction across all tokens, expressive multi-head representations and is used in most SOTA transformers today (e.g. GPT-4, PaLM, LLaMA); the downside is that it is very expensive, redundant and memory heavy (i.e. $$O(T^2d)$$ complexity for long sequences). 