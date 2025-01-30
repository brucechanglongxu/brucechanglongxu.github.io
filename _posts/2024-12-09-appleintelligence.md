---
layout: post
title: "Apple General Intelligence (Part I): GQA vs MHSA"
date: 2024-12-09
categories: Software Engineering
author: Bruce Changlong Xu
---

Apple around half a year ago introduced its foundation language models as part of the Apple Intelligence ecosystem, marking a significant milestone in AI development. The AI models are designed for both on-device and cloud-based processing, emphasizing efficiency, privacy, and user-centric AI applications. A standout feature is their focus on _privacy_ and _responsible AI_ (similar to Anthropic). Unlike many of their competitors, Apple ensures that user data is not used in training, and implements Private Cloud Compute to handle more tasks securely. Their AI system consists of two main foundation models:

1. **AFM-on-device:** A ~3 billion parameter model optimized for local processing, ensuring fast and secure interactions. 
2. **AFM-server:** A more powerful model running in the cloud, handling complex computations while maintaing privacy protections.

## Multihead Self Attention (MHSA) 

Apple's foundation models build upon a (decoder-only) transformer backbone, with several key architectural refinements. They employ **Grouped-Query Attention (GQA)** instead of standard multi-head self-attention (MHSA), reducing computational overhead whilst maintaing expressivity. This leads to faster inference time (30 percent less computation versus full MHSA), lower memory footprint, and retains competitive performance in reasoning tasks. In vanilla self-attention, each input token attends to all others using Query (Q), Key (K) and Value (V) projections, computed as follows:

$$Q = XW_Q, K = XW_K, V = XW_V$$ 

where $$x \in \mathbb{R}^{T \times d}$$ is the embedded input sequence of tokens (input sequence of length T, embedding size d), and $$W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$$ are learnable weight matrices. The attention scores are then computed:

$$A = \frac{QK^T}{\sqrt{d_k}}$$

where $$d_k$$ is the per-head key dimension, and we subsequently apply softmax normalization to obtain the weighted values:

$$\textbf{Attention}(Q, K, V) = \textbf{softmax}(A) \cdot V$$

and subsequently apply multi-head projection:

$$\textbf{MHSA}(X) = \textbf{Concat}(\textbf{head}_1, \cdots, \textbf{head}h) \cdot W_O$$

where $$W_O$$ is the output projection. This leads to powerful feature extraction across all tokens, expressive multi-head representations and is used in most SOTA transformers today (e.g. GPT-4, PaLM, LLaMA); the downside is that it is very expensive, redundant and memory heavy (i.e. $$O(T^2d)$$ complexity for long sequences). Let's break the above down, intuitively, and step by step. 

MHSA operates at both training and inference time, but the way it functions at inference is slightly different. The key is that during training, the MHSA network can process _all of the input/training tokens in parallel_. Whereas at inference time, because it is generating text in a sequential fashion token by token (depending on the previous token), it can only do so sequentially. The **model weights are only updated during training** (through backpropagation and gradient descent), not at inference (the model only uses its pretrained weights/matrices to compute outputs). Indeed, if the model kept adjusting weights at inference time, every new response would change the model's behavior unpredictably. The model can continue to learn after deployment, but typically in a controlled fashion for instance through reinforcement learning with human feedback (**note that both the universal set of word embeddings and transformer weights are learned during training and are frozen during inference**).

At inference time, let's say we input a prompt such as "how do transformer work", the input text is first tokenized into numerical representations (embeddings) $$X = [t_1, t_2, \cdots, t_n] \to [{E[t_i]}] \in \mathbb{R}^{n \times d}$$, which are then passed through our transformer architecture; since transformers lack inherent notion of a sequence order, a positional encoding vector is added to each token embedding:

$$X_i = X_e + PE$$ 

where $$PE$$ is a sinusoidal or learned positional encoding matrix (another post). Now with the learned weight matrices, we compute self-attention on the input prompt Q (queries) captures the current token's influence on other tokens, K (keys) represents the content in each token, and V (values) contains the actual token information. For each attention head, the input sequence $$X$$ (now a series of vector embeddings) is projected onto query, key, value subspaces $$Q, K, V$$ using learned weight matrices $$W_Q, W_K, W_V$$ from training (note that these weight matrices have dimensions $$d_e \times d_e$$ and are learned linear transformations fo the input embedding space). 

$$Q = XW_Q, K = XW_K, V = XW_V$$

the attention scores (an n by n matrix indicating how much attention each of the input tokens pays to each of the other input tokens) are then computed using the scaled dot product attention formula. We can think of attention like Google Search, where $$Q$$ represents the search query (what we're looking for), the key $$K$$ represents the stored index of webpages (is this webpage relevant to what we're looking for?) and $$V$$ being the actual content of the webpages (great, now let's read the webpage to see what we learn). Hence the overall attention scores provides us with:

$$A = \frac{QK^T}{\sqrt{d_k}}$$ 

which represents an $$n \times n$$ matrix that includes all the attention weights between all tokens in the sequence (i.e. how much attention does each input token pay to another). Think of it like a heatmap of token relationships, each row $$i$$ tells us how much token $$i$$ should attend to every other token. Indeed, $$Q, K \in \mathbb{R}^{n \times d_k}$$ represents how much each token wants to "ask" and "offer" information respectively. Now since $$A_{ij} = q_i \cdot k_j$$, we see that the matrix multiplication gives us a set of entries where the $$(i, j)$$-th entry represents how well token $$i$$ attends to token $$j$$. 

Now why do we typically use _multiple attention heads?_. The reason is, each attention head is assigned a **different attention focus**, for example one head might focus on syntax (e.g. subject-verb agreement), another might focus on semantic meaning, and another could track long-range dependencies. Instead of compressing all the information into a _single_ attention map, multiple attention heads allow the model to learn different types of relationships in the input prompt. Each head captures a different "view" of the sequence, and combining multiple heads allows a richer understanding of the input, which leads to a better response. 

Hence we have multiple set of $$\{W^{(i)}_Q, W^{(i)}_K, W^{(i)}_V\}$$ weight matrices for each attention head. Each head operates on a _lower dimensional_ subspace of the model's total embedding dimension. Each of the learned attention head matrices are linear projections that transform the original embedding space down to a smaller subspace before computing attention. We map each input token from the full embedding space of size $$d_{m}$$ down to $$d_m / h$$ where $$h$$ is the number of attention heads i.e. each attention head works in a fraciton of the space. Finally, after each head has computed its attention output, they are concatenated back together and mapped back to the full embedding space. 

## Grouped Query Attention (GQA) 

The key difference between GQA and MHSA is that in MHSA, each attention head learns its own independent set of Q, K, V matrices, computing self-attention independently and producing separate attention outputs (which are then concatenated and linearly transformed into the final output -- each attention head $$h$$ has its own $$Q, K, V$$).

$$A_h = \textbf{softmax}(\frac{Q_h K_h^T}{\sqrt{d_k}}) \cdot V_h$$

 GQA reduces the number of learned weights whilst aiming to retain the expressivity of multi-head attention -- instead of each attention head having its own Q, K and V, GQA **shares the key and value matrices across multiple heads** (whilst still having a unique query matrix). This leads to faster inference speed and lower memory usage, whilst maintaining diverse query interactions (multiple heads share the same $$K_g, V_g$$ but still use unique $$Q_h$$, reducing redundancy in key, value computations since queries tend to be unique, but keys and values are often redundant across heads e.g. "cat" seems to have a similar meaning and lookup characteristics across multiple input prompts). 

 $$A_h = \textbf{softmax}(\frac{Q_h K_g^T}{\sqrt{d_k}}) \cdot V_g$$

This technique is employed not only in open-source models like LLaMA2, but at a smaller scale in SOTA frontier models such as GPT-4 and Claude-opus on an ad-hoc basis. 
