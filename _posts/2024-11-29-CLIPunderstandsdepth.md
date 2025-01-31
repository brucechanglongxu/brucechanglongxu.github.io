---
layout: post
title: "CLIP understanding depth"
date: 2024-11-29
categories: vision
author: Bruce Changlong Xu
---

In a future post we talk about CLIP, in this post, we analyze how CLIP can understand depth. In particular, we dive deep into the paper *CLIP can understand depth* by Dunam Kim and Seokju Lee. Let us begin the odyssey. 

The main technical innovation that they present is that of *mirror*, which is a learnable set of embeddings that modulate image features to reflect depth-related cues without relying on human-language tokens. The embeddings effectively leverage CLIP's semantic knowledge for pixel-level depth prediction. The *mirror embedding* is a small, trainable matrix acting as a static, non-human language prompt for CLIP's text encoder; this avoids the need for traditional fine-tuning, preserving CLIP's general purpose alignment whilst adapting it to the depth domain. 

Prior methods struggled with suboptimal image-text alignment in depth-related tasks, relying heavily on depth-specific promtps in human language; CLIP2Depth bypasses this limitation by replacing human-lgnauge token embeddings with the *mirror* embeddings, achieving dense monocular depth estimation through non-human language supervision. 

Fine-tuning involves retraining parts (or all) of a pretrained model on a new dataset or task; for CLIP, this could involve adjusting the image encoder (modifying the vision backbone e.g. ResNet or ViT) to better extract features relevant to the new task (e.g. depth cues); or adjusting the text encoder (altering the language backbone to encode task-specific text prompts e.g. depth-related descriptors like "far" or "near"). Updating the joint embedding space to align images with text in a way that emphasizes the new task, potentially at the expense of general-purpose alignment. Or we could retrain the entire model (both encoders and joint emebdding space) for the downstream task. The downside of this approach is that it could be computationally expensive, and risk of overfitting / loss of versatility that could degrade performance. 

Instead of fine-tuning CLIP2Depth uses **mirror embedding** that serves as an input to CLIP's text encoder bypassing the need to retrain the encoder itself. It replaces human-language tokens like "far" or "close" with task-specific, non-human latent representations. These embeddings are optimized during trainign to adapt the outputs of the frozen text encoder for depth estimation. Both the **image encoder** and **text encoder** remain unchanged during training, and the pretrained joint embedding space is preserved, ensuring that CLIP retains its general-purpose vision-language alignment. Outputs from the text encoder (processed mirror embeddings) are used to modulate image embeddings through **FiLM** layers. This dynamically adjusts the image features for depth estimation without altering the underlying vision model. Whilst the image and text encoders are frozen, *the mirror embeddings and decoders are trainable in this "fine tuning" phase*. 

**Decoder Parameters:** The decoder is a lightweight module that translates the modulated image features into pixel-wise depth predictions; it includes components like deconvolutional layers and FiLM locks that integrate the outputs of the image and text encoders. While this and the mirror emebddings are updated during training, the **core CLIP model** remains *frozen* which means that the pre-trained vision-language alignment learned by CLIP is preserved, the *image encoder* continues to extract general-purpose visual features, the *text encoder* processes the mirror embeddings without any changes to its weights.  

Let's look at mirror embeddings. The **mirror embedding matrix** is initialized to a static set of randomized vectors $$M \in \mathbb{R}^{s \times d}$$, where $$s$$ is the number of trainable latent tokens (MIRROR embeddings), and $$d$$ is the embedding dimension (typically $$512$$ for CLIP's text encoder). Initially, $$M$$ is initialized with random values, often sampled from a Gaussian distribution $$M_{ij} \sim \mathcal{N}(0, \sigma^2)$$, where $$\sigma$$ is a small constant (e.g. $$\sigma = 0.02$$). 

1. **Pretraining CLIP:** Before adapting it for depth estimation, the CLIP base model is pretrained on a massive dataset of image-text pairs, during this pretraining, CLIP learns to align images and text in a shared embedding space using a contrastive loss. Here, CLIP becomes a general-purpose vision-language model capable of understanding diverse visual and textual concepts, performing tasks like image-text retrieval or zero-shot classification. 

After pretraining, CLIP has *frozen weights* for the image encoder, and the text encoder. The pretrained CLIP base model is *not modified* during the CLIP2Depth training process. Once the base model is pretrained, it is adapted for depth estimation (image encoder and text encoder weights are frozen), MIRROR embeddings and a decoder are introduced as task-specific, trainable components. 




