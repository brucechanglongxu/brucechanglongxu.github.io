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

If we consider a batch of $$N$$ image-text pairs, for each pair $$(i, t)$$ the cosine similarity $$s(i, t)$$ is computed between all image and text embeddings in the batch:

$$s(i, t) = \frac{v_i \cdot v_t}{||v_i||||v_t||}$$

The model then computes two distributions:

1. Image-to-Text Similarity:

$$p_{i \to t} = \frac{\exp{s(i, t)}}{\sum_{k = 1}^N \exp{s(i, t_k)}}$$

2. Text-to-Image Similarity:

$$p_{t \to i} = \frac{\exp{s(i, t)}}{\sum_{k = 1}^N \exp{s(i_k, t)}}$$

The contrastive loss for each batch is the average of two cross-entropy losses:

$$\mathcal{L} = -\frac{1}{2N} \sum_{i = 1}^N [\log p_{i \to t} + \log p_{t \to i}]$$

The loss maximizes the similarity $$s(i, t)$$ for correct pairs, and minimizes the similarity $$s(i, t')$$ for mismatched pairs. The model then backpropagates the loss and updates the parameters of both the image encoder and text encoder using a gradient descent algorithm like Adam. The process repeats over many batches until the model converges. 