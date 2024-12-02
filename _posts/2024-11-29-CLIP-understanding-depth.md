---
layout: post
title: "CLIP understanding depth"
date: 2024-11-29
categories: AI
author: Bruce Changlong Xu
---

In a future post we talk about CLIP, in this post, we analyze how CLIP can understand depth. In particular, we dive deep into the paper *CLIP can understand depth* by Dunam Kim and Seokju Lee. Let us begin the odyssey. 

The main technical innovation that they present is that of *mirror*, which is a learnable set of embeddings that modulate image features to reflect depth-related cues without relying on human-language tokens. The embeddings effectively leverage CLIP's semantic knowledge for pixel-level depth prediction. 