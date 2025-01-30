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

**GQA over MHSA** 

Apple's foundation models build upon a (decoder-only) transformer backbone, with several key architectural refinements. They employ **Grouped-Query Attention (GQA)** instead of standard multi-head self-attention (MHSA), reducing computational overhead whilst maintaing expressivity. This leads to faster inference time (30 percent less computation versus full MHSA), lower memory footprint, and retains competitive performance in reasoning tasks. In vanilla self-attention, each input token attends to all others using Query (Q), Key (K) and Value (V) projections, computed as follows:

$$Q = XW_Q, K = XW_K, V = XW_V$$ 

where $$x \in \mathbb{R}^{T \times d}$$ is the embedded input sequence of tokens (input sequence of length T, embedding size d), and $$W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$$ are learnable weight matrices. The attention scores are then computed:

$$A = \frac{QK^T}{\sqrt{d_k}}$$

where $$d_k$$ is the per-head key dimension, and we subsequently apply softmax normalization to obtain the weighted values:

$$\textbf{Attention}(Q, K, V) = \textbf{softmax}(A) \cdot V$$

and subsequently apply multi-head projection:

$$\textbf{MHSA}(X) = \textbf{Concat}(\textbf{head}_1, \cdots, \textbf{head}h) \cdot W_O$$

where $$W_O$$ is the output projection. This leads to powerful feature extraction across all tokens, expressive multi-head representations and is used in most SOTA transformers today (e.g. GPT-4, PaLM, LLaMA); the downside is that it is very expensive, redundant and memory heavy (i.e. $$O(T^2d)$$ complexity for long sequences). Let's break the above down, intuitively, and step by step. 

MHSA operates at both training and inference time, but the way it functions at inference is slightly different. The key is that during training, the MHSA network can process _all of the input/training tokens in parallel_. Whereas at inference time, because it is generating text in a sequential fashion token by token (depending on the previous token), it can only do so sequentially. The **model weights are only updated during training** (through backpropagation and gradient descent), not at inference (the model only uses its pretrained weights/matrices to compute outputs). Indeed, if the model kept adjusting weights at inference time, every new response would change the model's behavior unpredictably. The model can continue to learn after deployment, but typically in a controlled fashion for instance through reinforcement learning with human feedback. 

At inference time, let's say we input a prompt such as "explain AGI", the input text is first tokenized into numerical representations (embeddings) $$X = [t_1, t_2, \cdots, t_n] \to [{E[t_i]}] \in \mathbb{R}^{n \times d}$$, which are then passed through our transformer architecture; since transformers lack inherent notion of a sequence order, a positional encoding vector is added to each token embedding:

$$X_i = X_e + PE$$ 

where $$PE$$ is a sinusoidal or learned positional encoding matrix (another post). Now with the learned weight matrices, we compute self-attention on the input prompt Q (queries) captures the current token's influence on other tokens, K (keys) represents the content in each token, and V (values) contains the actual token information. 

 Our MHSA architecture determines relationships between the tokens (Q, K, V), and the model then generates a probability distribution over the possible next words. The output seqeunce (response) is generated one token at at ime using our fixed/trained model weights. 