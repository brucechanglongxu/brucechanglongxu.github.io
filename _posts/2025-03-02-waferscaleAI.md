---
layout: post
title: "Wafer-scale Artificial Intelligence"
date: 2025-03-02
categories: AI
author: Bruce Changlong Xu
---

Large AI requires large infrastructure, and wafer-scale processors are optimized for maximum dataflow execution, SIMD-heavy parallelism, and efficient inter-tile communication. Each **processing element** is the fundamental compute unit in Heisenberg, where a single PE is responsible for executing SIMD operations, managing wavelets, and handling local data storage. PEs are designed to process vectorized workloads efficiently using SIMD (Single Instruction, Multiple Data). 

PEs support the following canonical SIMD modes:
- **SIMD-256:** 8 FP32, 16 FP16, 32 FP8 operations per cycle. 
- **SIMD-512:** 16 FP32, 32 FP16, 64 FP8 operations per cycle. 
- **Sparse SIMD:** Selectively executes only on active data elements (which is optimized for sparse AI). 

Each processing element follows a five stage execution pipeline, _instruction decode_ (which fetches wavelet instructions and decodes SIMD operations), _operand fetch_ (which loads vector data from the register file or scratchpad memory), _compute execution_ (which performs FMAC or other arithmetic ops), _result forwarding_ (sends results to local memory or across the interconnect), _wavelet completion_ (signals execution completion and readiness for the next task). PEs have a memory hierarchy consisting of 1. Register files (32 KB) that store operands for SIMD computations 2. Scratchpad memory (256 KD) that functions as tile-local memory for fast-access data and 3. Global SRAM (100 MB) that is a shared memory pool for multi-tile workloads. 

