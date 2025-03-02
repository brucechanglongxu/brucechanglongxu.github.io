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

A _tile_ is a self-sufficient compute unit consisting of 1. Multiple PEs (typically 16-64 per tile) 2. Tile-level scratchpad memory (~256 KB per tile) 3. Wavelet queue and execution controller (for managing tasks) 4. Interconnect router (for tile to tile communication). Each tile autonomously schedules wavelets and determines whether to execute a wavelet immediately or defer based on operand availability; as well as if the results should be stored locally or forwarded to another tile. 

Since a wafer-scale chip has thousands of tiles, not all tiles are active at once; Tiles shut down when idle (dynamic power gating), tiles adjust execution speed based on workload intensity (adaptive frequency scaling). 

## Interconnect Fabric

The wafer-scale interconnect fabric is a network-on-chip (NoC) that is designed for high-bandwidth, low-latency communication between tiles. Distributed execution through wavelet propagation, and scalability to thousands of tiles without bottlenecks. The interconnect uses a hybrid topology combining 2D mesh (short-distance communication and fast local tile to tile communication), a hierarchical tree (long distance aggregation which leads to efficient global data reduction) and broadcast/multiple networks (optimized for AI workloads requiring data replication). 

The interconnect can be routed through 1. Static routing (pre-assigned tile to tile paths for structured workloads like GEMM, FFTs) 2. Adaptive routing (dynamic path selection based on tile congestion and execution readiness) 3. Load-balanced execution (which ensures fair distribution of wavelets to prevent bottlenecks). 

## Memory Subsystem

Wafer-scale memory architecture minimizes off-chip DRAM usage by relying on on-chip memory tiers. PE registers (limited ot a single PE), scratchpad memory (limited to a single tile), global SRAM (multi-tile access) and HBM (global, accessible by CPU and Wafer Scale Engine) are a hierarchical tier of memories. 

Instead of traditional cache hierarchies, Heisenberg tiles explicitly fetch and store data using wavelets. This avoids cache coherence overhead, and minimizes memory stalls by prefetching wavelets. It supports sparse memory access patterns (Transformers, GNNs etc.). Unlike CPUs, which use a program counter, or GPUs, which use thread blocks, Wafer Scale Processors execute **wavelets**. 

The host system first injects tasks as wavelets, the interconnect then routes wavelets to available tiles, a tile processes the wavelet using SIMD PEs, and the result is either stored or forwarded to another tile. There are three wavelet types:

1. **Compute Wavelets:** For SIMD operations (GEMM, Convolutions, FFTs)
2. **Memory Wavelets:** For Data Movement (Load/Store)
3. **Synchronization Wavelets:** For barrier synchronization across tiles. 

## Pipeline Bypass

To support wider datapaths, the waferscale pipeline introduces new bypass logic. Without this, any floating-point or memory result could not be used until 3 cycles later; however this new logical allows certain results to be forwarded with only a 2-cycle delay. This 2-cycle bypass applies under limited conditions (e.g. only for the _src0_ operand of an instruction, and only if producer and consumer execute on the same datapath lane). Integer pipeline results are generally more freely bypassable, whereas FP results can only fast-bypass to another FP operation of the same type (FP16 to FP16 or FP32 to FP32). We define specific FP ops that support the 2-cycle bypass path; However wider SIMD also necessitates some longer paths (the upper 128 bits of a 256-bit operation do **not** participate in the full bypass crossbar) so if a value in those upper lanes must be forwarded to a different lane (or to a lower portion), an extra cycle penalty is incurred. 

Essentially, this design traded a bit of complexity to accelerate common dependent sequences (e.g. accumulation loops) with faster bypass, but accepted a latency hit on less-common cross-lane dependencies when usign the very wide SIMD modes. The _partial bypass_ scheme is a classic pipeline trade-off: complete bypass networks maximize performance but significantly impact cycle time, area, and power, so the wafer opts for a balanced approach rather than a fully global forward network. 

## Numerics

The wafer scale adds support for new numeric formats geared towards machine learning; notably it introduces 8-bit floating point (FP8) with two variants (E5M2 and E4M3 i.e. 5-bit or 4-bit exponent). These ultra-low precision floats enable higher AI throughput and memory savings, for example NVIDIA's Hopper GPU similarly introduced FP8 and achieved 2 times performance and memory efficiency gains versus FP16 on training workloads. 