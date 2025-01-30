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

Note that CLIP aligns bidirectionality - image to text and text to images, which enables zero-shot classification where text prompts can label images. Since CLIP is trained on diverse image-text pairs, it generalizes to unseen tasks without additional fine-tuning. After training, CLIP can *match images and text* and perform *zero-shot classification* (i.e. classify inputs (image/text) into categories it was not explicitly trained on, using only natural language descriptions or prompts - this means that the model doesn't need labeled examples of the new categories to perform classification, it generalizes to unseen classes based on the relationships it learned during training). 

For CLIP zero-shot classification works as follows; during training, CLIP learns a shared embedding space for images and text by aligning image embeddings with text embeddings, which allows the model to understand relationships between visual features and semantic concepts described in the text. Since CLIP's training is diverse and broad (e.g. image-text pairs from the web), it develops a generalized understanding of various objects, scenes and concepts, and can recognize new categories. To classify an image into one of the several categories, you can provide text prompts representing the categories - e.g. to classify an image of an animal, the text prompts could be "a dog", "a cat" and "a bird"; CLIP calculates the similarity between the image embedding and each fo the text embeddings (generated from prompts) and the category with the highest similarity score is selected as the predicted class. 

The benefits of zero-shot classification is that we don't need to fine-tune the model or collect labeled data for new classes, saving time/resources and making this highly efficient. We can even define arbitrary categories using natural language prompts (e.g. "a blue dog") and unlike traditional supervised methods which require labeled data for each class, zero-shot classification can handle tasks with an unlimited number of categories. The caveat is that we could be sensitive to the prompts (e.g. "a Bengal tiger" would confuse the model over "a tiger"), and could get confused. 

**Limitations of CLIP** 

CLIP (Contrastive Language – Image Pretraining) has revolutionized the way we think about aligning vision and language, offering unprecedented flexibility and generalization across tasks. However, like any technology, it comes with its own set of strengths and limitations.

*Strengths* 

One of CLIPs most remarkable strengths is its ability to perform zero-shot classification, which allows it to tackle tasks it wasnt explicitly trained for. Using natural language prompts, users can define new categories or tasks without needing additional labeled data or retraining. This makes CLIP incredibly versatile and scalable for real-world applications.

The foundation of CLIPs power lies in its shared embedding space, which aligns images and text in a way that enables seamless cross-modal reasoning. This allows CLIP to understand both visual and textual representations in context, supporting tasks like image retrieval, captioning, and even creative uses like art description or meme generation. Its ability to perform bidirectional alignment means it can match images to text or vice versa with equal ease.

CLIPs training process, based on a massive dataset of web-sourced image-text pairs, equips it with a broad understanding of diverse concepts. This extensive exposure allows it to generalize across domains, from everyday objects to creative compositions, even combining concepts it may not have explicitly seen during training. Its flexibility and efficiency are further enhanced by its ability to operate without task-specific fine-tuning, simplifying deployment and saving computational resources.

The applications of CLIP are vast and practical. It excels in tasks like content moderation, where it can identify inappropriate content using simple textual descriptions. It also shines in search and organization, retrieving assets or images based on dynamic natural language queries. Moreover, its ability to interpret complex relationships and support descriptive queries has significant potential for accessibility, such as aiding visually impaired users.

*Limitations* 

Despite its strengths, CLIP has some notable limitations. One of its challenges stems from its reliance on web-sourced training data, which can embed biases and uneven representation into the model. These biases may lead to skewed outputs, especially in sensitive or underrepresented domains. Additionally, if a concept was poorly represented in the training data, CLIP may struggle to understand or recognize it effectively.

Another limitation lies in CLIP’s sensitivity to prompt phrasing. The model’s performance can vary significantly based on how a query is worded. For instance, prompts like “a tiger” might yield different results than “a Bengal tiger” or “a striped tiger,” highlighting the need for thoughtful prompt engineering. Furthermore, while CLIP can handle broad categorization, it often struggles with fine-grained distinctions, such as differentiating between visually similar objects like leopards and cheetahs.

CLIP’s computational requirements can also be a barrier for certain use cases. Encoding both images and text into high-dimensional embeddings requires significant resources, making it less accessible for low-resource environments or real-time applications. Similarly, while its training process ensures global semantic understanding, it sacrifices local details, making it less suited for tasks requiring pixel-level precision, such as depth estimation or fine object segmentation.

Lastly, CLIP does not account for temporal information, limiting its applicability to static image tasks. For dynamic contexts, such as video analysis, its lack of time-based reasoning becomes a notable drawback. Additionally, its outputs can sometimes feel like a black box, making interpretability and debugging challenging, especially in edge cases or out-of-distribution scenarios.

