---
layout: post
title: "Training CLIP"
date: 2024-12-02
categories: AI
author: Bruce Changlong Xu
---

CLIP is a model trained to match **images with text descriptions** by learning a shared embedding space for both modalities. The model learns to bring embeddings of matching image-text pairs closer together and *push embeddings of mismatched pairs further apart*; which is achived through a **contrastive learning objective**. The key components are as follows:

1. **Image Encoder:** This is usually a vision transformer (ViT) or a convolutional neural network (e.g. ResNet), which converts images into high-dimensional embeddings. 
2. **Text Encoder:** This is usually based on a transformer model, and converts text (e.g. captions or descriptions) into high-dimensional embeddings in the same space as image embeddings. 
3. **Shared Embedding Space:** Both the image and text embeddings are projected into a common space where their similarity can be computed. Typically, *cosine similarity* is used to measure the alignment between image and text embeddings.

CLIP is typically trained on a *massive dataset of image-text pairs* collected from the web, which enables it to generalize well to many tasks. The training process is as follows:

- **Input Preparation**: We feed the image into an image encoder to produce an embedding vector $$v_i$$, and a text description corresponding to the image is tokenized and fed into the text encoder to produce an embedding vector $$v_t$$. 
- **Contrastive Loss**: CLIP uses a *contrastive loss function* to train the model, which ensures that the embedding vectors $$v_i$$ and $$v_t$$ for a matching image-text pair have a high similarity, and non-matching pairs are less similar. 