---
layout: post
title: "SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training"
date: 2025-02-04
categories: AI RL
author: Bruce Changlong Xu
---

This paper investigates the generalization and memorization capabilities of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) in post-training foundation models across text-based and visual environments. The authors introduce GeneralPoints, an arithmetic reasoning card game, and V-IRL, a real-world navigation task, to assess how well models trained using SFT and RL generalize to unseen variations. They find that RL, particularly with an outcome-based reward system, excels at generalization, effectively transferring learned rules to novel scenarios in both textual and visual domains. In contrast, SFT mainly memorizes training data, struggling with out-of-distribution (OOD) tasks. Further analysis reveals that RL enhances visual recognition abilities, contributing to its superior generalization. However, SFT plays a critical role in stabilizing output formats, enabling RL to build upon structured outputs. These findings emphasize RL’s advantage in acquiring transferable knowledge for complex, multimodal tasks.