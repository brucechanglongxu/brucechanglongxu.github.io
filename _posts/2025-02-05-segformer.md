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

Unlike prior Transformer-based segmentation models such as SETR (which extracts single-scale low-resolution feature maps), SegFormer employs a hierarchical Transformer encoder (MiT), which processes multi-scale feature maps, removes positional encoding, and uses overlapping patch merging which retains spatial continuity. Traditional segmentation models rely on heavy decoders involving dilated convolutions (DeepLabV3+) or multi-head attention (SETR), which makes them computationally expensive. SegFormer replaces complex decoders with a simple MLP-based design which aggregates multi-scale Transformer features without computaitonally expensive operationsl; effectively fuses local and global attention through simple MLP layers and achieves SOTA accuracy whilst reducing FLOPs and memory consumption. 

## Sequence Reduction

In self-attention, every patch in an image attends to every other patch. Given an image of size $$H \times W$$ (height times width), we divide it into N patches:
$$N = \frac{H \times W}{P^2}$$
where $$P$$ is the patch size. In vanilla ViTs, self-attention operates on a matrix of shape $$(N, N)$$, which means that the computational complexity is $$O(N^2)$$. For example, if we process an image of size $$512 \times 512$$, with $$16 \times 16$$ patches, we have:
$$N = \frac{512 \times 512}{16^2} = 1024$$
so the attention operation would require computing a $$1024 \times 1024$$ matrix, which is costly. Instead of computing full self-attention on all $$N$$ patches, SegFormer introduces _sequence reduction_, which groups patches together and reduces the number of tokens before applying self-attention. Instead of directly applying self-attention to $$N$$ patches, SegFormer downsamples the key ($$K$$) and value ($$V$$) matrices by a factor of $$R$$, the sequence reduction ratio. 

The keys and values are reshaped _before_ applying attention:
$$K' = \textbf{Reshape}(\frac{N}{R}, C \times R) (K)$$
where $$R$$ is the reduction ratio ($$4, 8$$) $$C$$ is the number of channels in the feature representation, which means that the reshaped $$K'$$ matrix is $$\frac{N}{R} \times C$$ instead of $$N \times C$$, meaning that we **attend to fewer tokens**. Since $$K'$$ and $$V'$$ have fewer tokens, the attention computation now operates over a smaller $$\frac{N}{R} \times N$$ matrix rather than an $$N \times N$$ matrix. This reduces the computational cost to:
$$O(\frac{N^2}{R})$$
which is $$R$$ times more efficient than standard self-attention.

Think of sequence reduction as compressing the "memory" of the image before performing attention. Instead of making every pixel interact with every other pixel, SegFormer groups patches together to form "super-patches", allowing 
