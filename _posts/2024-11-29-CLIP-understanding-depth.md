---
layout: post
title: "CLIP understanding depth"
date: 2024-11-29
categories: AI
author: Bruce Changlong Xu
---

In a future post we talk about CLIP, in this post, we analyze how CLIP can understand depth. In particular, we dive deep into the paper *CLIP can understand depth* by Dunam Kim and Seokju Lee. Let us begin the odyssey. 

The main technical innovation that they present is that of *mirror*, which is a learnable set of embeddings that modulate image features to reflect depth-related cues without relying on human-language tokens. The embeddings effectively leverage CLIP's semantic knowledge for pixel-level depth prediction. The *mirror embedding* is a small, trainable matrix acting as a static, non-human language prompt for CLIP's text encoder; this avoids the need for traditional fine-tuning, preserving CLIP's general purpose alignment whilst adapting it to the depth domain. 

Prior methods struggled with suboptimal image-text alignment in depth-related tasks, relying heavily on depth-specific promtps in human language; CLIP2Depth bypasses this limitation by replacing human-lgnauge token embeddings with the *mirror* embeddings, achieving dense monocular depth estimation through non-human language supervision. 

Fine-tuning involves retraining parts (or all) of a pretrained model on a new dataset or task; for CLIP, this could involve adjusting the image encoder (modifying the vision backbone e.g. ResNet or ViT) to better extract features relevant to the new task (e.g. depth cues); or adjusting the text encoder (altering the language backbone to encode task-specific text prompts e.g. depth-related descriptors like "far" or "near"). Updating the joint embedding space to align images with text in a way that emphasizes the new task, potentially at the expense of general-purpose alignment. Or we could retrain the entire model (both encoders and joint emebdding space) for the downstream task. The downside of this approach is that it could be computationally expensive, and risk of overfitting / loss of versatility that could degrade performance. 

Instead of fine-tuning CLIP2Depth uses **mirror embedding** that serves as an input to CLIP's text encoder bypassing the need to retrain the encoder itself. It replaces human-language tokens like "far" or "close" with task-specific, non-human latent representations. These embeddings are optimized during trainign to adapt the outputs of the frozen text encoder for depth estimation. Both the **image encoder** and **text encoder** remain unchanged during training, and the pretrained joint embedding space is preserved, ensuring that CLIP retains its general-purpose vision-language alignment. Outputs from the text encoder (processed mirror embeddings) are used to modulate image embeddings through **FiLM** layers. This dynamically adjusts the image features for depth estimation without altering the underlying vision model. Whilst the image and text encoders are frozen, *the mirror embeddings and decoders are trainable in this "fine tuning" phase*. 

**FiLM Layers** 

