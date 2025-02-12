---
layout: post
title: "Optimizing Llama memory footprint for inference"
date: 2025-02-11
categories: AI
author: Bruce Changlong Xu
---

The challenge of inferencing models with ever longer inputs (augmenting LLMs with a repository of domain specific data, and thereafter performing inference), leads to massive memory overhead due to caching key/value activations (KV cache). 