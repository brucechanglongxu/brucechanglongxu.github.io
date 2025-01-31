---
layout: post
title: "YOLO, Flamingo, Chameleon and AIMv2: Trends in Vision Language Models"
date: 2024-12-25
categories: AI vision
author: Bruce Changlong Xu
---

For decades, artificial intelligence models were constrained to a single modality—language, vision, or audio—limiting their ability to process information in a way that resembles human cognition. However, the rapid rise of multimodal AI has revolutionized how models understand and generate information, enabling them to bridge the gap between language and perception. Vision-language models (VLMs) lie at the heart of this transformation, integrating text and images to power applications such as image captioning, visual question answering, and text-based image retrieval. From early contrastive learning-based models like OpenAI's CLIP to the emergence of Large Multimodal Models (LMMs) like DeepMind’s Flamingo, the field has experienced rapid advancements in training methodologies, scalability, and downstream applications.

Let's work with the example of surgical AI. Indeed for an intelligent, autonomous surgical agent to be developed, such a model must unavoidably integrate multiple sources of information (video (RGB, depth, IR), robot kinematics, audio, sensor data); now there is a nuanced question of _when_ and _how_ we combine such multimodal inputs in a deep learning model. 

https://huyenchip.com/2023/10/10/multimodal.html
https://arxiv.org/pdf/2405.09818
https://arxiv.org/pdf/2411.14402
https://arxiv.org/pdf/1506.02640