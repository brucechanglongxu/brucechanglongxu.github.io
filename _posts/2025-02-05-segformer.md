---
layout: post
title: "NVIDIA's Segformer"
date: 2025-02-05
categories: AI vision
author: Bruce Changlong Xu
---

Semantic segmentation has long been dominated by CNN-based architectures, but transformers have rapidly emerged as a powerful alternative. NVIDIA’s SegFormer offers an efficient, lightweight, and scalable transformer-based model that balances accuracy, speed, and flexibility across diverse segmentation tasks. Unlike previous transformer-based segmentation models, SegFormer eliminates complex post-processing, heavy computations, and fixed-resolution constraints—making it an attractive solution for real-world applications.Semantic segmentation has long been dominated by CNN-based architectures, but transformers have rapidly emerged as a powerful alternative. NVIDIA’s SegFormer offers an efficient, lightweight, and scalable transformer-based model that balances accuracy, speed, and flexibility across diverse segmentation tasks. Unlike previous transformer-based segmentation models, SegFormer eliminates complex post-processing, heavy computations, and fixed-resolution constraints—making it an attractive solution for real-world applications.

Most transformer-based segmentation models rely on computationally expensive architectures with _pixel-wise_ attention and rigid dependencies on CNN-based decoders. NVIDIA's SegFormer takes a different approach:

1. **Hierarchical Multi-Scale Feature Extraction:** SegFormer extracts features at multiple resolutions, mimicking CNN-like spatial hierarchies whilst leveraging the global context-awareness of transformers.
2. **Removes Position Embeddings:** In contrast to standard ViTs, SegFormer removes positional embeddings which allows it to generalize across different image resolutions without requiring fixed input sizes.
3. **Lightweight MLP Decoder:** Instead of heavy segmentation heads, SegFormer uses a simple MLP-based decoder, efficiently aggregating multi-scale features while keeping computational costs low.

with these innovations, SegFormer is able to achieve SOTA perfrmance with fewer parameters than other leading ViT (or CNN) models. SegFormer is able to presrve local features in early layers, whilst capturing long-range dependencies at deeper layers; hence combining the best of CNNs and Transformers (CNN-like local receptive fields and Transformer-like global context). 

## Mix Transformer (MiT) Backbone

At the core of Segformer is MiT (Mix Transformer), an efficient vision transformer designed for dense prediction tasks. Instead of traditional ViT patch embeddings, MiT _overlaps patches_ to better capture spatial structures. Unlike global self-attention (which scales quadratically), MiT uses _local attention_ within each hierarchical stage, making computation more scalable. Similar to CNNs, tokens are progressively downsampled to reduce spatial redundancy and capture multi-scale features. 

Unlike prior Transformer-based segmentation models such as SETR (which extracts single-scale low-resolution feature maps), SegFormer employs a hierarchical Transformer encoder (MiT), which processes multi-scale feature maps, removes positional encoding, and uses overlapping patch merging which retains spatial continuity. 
