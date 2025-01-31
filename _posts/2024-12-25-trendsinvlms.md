---
layout: post
title: "YOLO, Flamingo, Chameleon and AIMv2: Trends in Vision Language Models"
date: 2024-12-25
categories: AI vision
author: Bruce Changlong Xu
---

For decades, artificial intelligence models were constrained to a single modality—language, vision, or audio—limiting their ability to process information in a way that resembles human cognition. However, the rapid rise of multimodal AI has revolutionized how models understand and generate information, enabling them to bridge the gap between language and perception. Vision-language models (VLMs) lie at the heart of this transformation, integrating text and images to power applications such as image captioning, visual question answering, and text-based image retrieval. From early contrastive learning-based models like [OpenAI's CLIP](https://arxiv.org/pdf/2103.00020) to the emergence of Large Multimodal Models (LMMs) like DeepMind’s [Flamingo](https://arxiv.org/pdf/2204.14198), the field has experienced rapid advancements in training methodologies, scalability, and downstream applications.

Let's work with the example of surgical AI. Indeed for an intelligent, autonomous surgical agent to be developed, such a model must unavoidably integrate multiple sources of information (video (RGB, depth, IR), robot kinematics, audio, sensor data); now there is a nuanced question of _when_ and _how_ we combine such multimodal inputs in a deep learning model. [Early fusion](https://arxiv.org/abs/2405.09818) is when we merge all input modalities at the _feature level_ before deep feature extraction. [Late fusion](https://link.springer.com/article/10.1007/s11042-020-08836-3) is when we process each modality independently, and subsequently merge outputs at the _decision level_. Indeed, Lilian Weng's blog post captures this idea beautifully in the context of the different ways vision langauge models can fuse language and visual information (i.e. _"pre-trained generalized language models capable of consuming visual signals"_). 

Intuitively, we want to fuse data modalities early when we have tighly coupled and interdependent modalities (e.g. video, depth and kinematics during robotic surgery) and when we want joint feaeture learning across modalities. This is particularly amenable for low-dimensional structured inputs, where we jointly encode multimodal features with self-supervised learning methodologies (e.g. contrastive learning, masked prediction). This lets the model learn generalizable, shared features without overfitting on a particular data-set. Late fusion is much better when the modalities aren't directly aligned, each modality needs independent deep feature extraction before merging, or some modalities might be missing. 



- Weng, Lilian. (Jun 2022). _"Generalized visual language models."_ Lil’Log. https://lilianweng.github.io/posts/2022-06-09-vlm/.
- Fini, Enrico, et al. _"Multimodal Autoregressive Pre-training of Large Vision Encoders."_ Apple, 21 Nov. 2024, arXiv:2411.14402. https://github.com/apple/ml-aim.
- Chameleon Team. _"Chameleon: Mixed-Modal Early-Fusion Foundation Models."_ FAIR at Meta, 17 May 2024, arXiv:2405.09818.
- Redmon, Joseph, et al. _"You Only Look Once: Unified, Real-Time Object"_ Detection. University of Washington, Allen Institute for AI, and Facebook AI Research, 2016, arXiv:1506.02640.