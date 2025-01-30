---
layout: post
title: "AGI: Apple General Intelligence"
date: 2024-12-09
categories: Software Engineering
author: Bruce Changlong Xu
---

Apple around half a year ago introduced its foundation language models as part of the Apple Intelligence ecosystem, marking a significant milestone in AI development. The AI models are designed for both on-device and cloud-based processing, emphasizing efficiency, privacy, and user-centric AI applications. A standout feature is their focus on _privacy_ and _responsible AI_ (similar to Anthropic). Unlike many of their competitors, Apple ensures that user data is not used in training, and implements Private Cloud Compute to handle more tasks securely. Their AI system consists of two main foundation models:

1. **AFM-on-device:** A ~3 billion parameter model optimized for local processing, ensuring fast and secure interactions. 
2. **AFM-server:** A more powerful model running in the cloud, handling complex computations while maintaing privacy protections.

Apple's foundation models build upon a (decoder-only) transformer backbone, with several key architectural refinements. They employ **Grouped-Query Attention (GQA)** instead of standard multi-head self-attention (MHSA), reducing computational overhead whilst maintaing expressivity. This leads to faster inference time (30 percent less computation versus full MHSA), lower memory footprint, and retains competitive performance in reasoning tasks. 