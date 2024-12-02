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