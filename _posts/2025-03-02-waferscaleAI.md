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

The wafer scale adds support for new numeric formats geared towards machine learning; notably it introduces 8-bit floating point (FP8) with two variants (E5M2 and E4M3 i.e. 5-bit or 4-bit exponent). These ultra-low precision floats enable higher AI throughput and memory savings, for example NVIDIA's Hopper GPU similarly introduced FP8 and achieved 2 times performance and memory efficiency gains versus FP16 on training workloads. Heisenberg can operate on FP8 values (packed two per 16-bit word) and provides hardware translation for 4-bit or 5-bit exponent formats depending on the opcode. It also introduced 64-bit data support for both integer and double precision floating point, following IEEE-754 for FP64 (previously the architecture was limited to only 16/32-bit types). Double precision math can now run natively, which is important for HPC workloads or accumulator precision, though certain FP64 operations are simplified (e.g. some fused ops don't support denormals to keep hardware manageable). Overall, HBG's ISA is _widened at both ends: very low precision for AI and high precision for scientific computing_ which makes the chip more versatile. 

Accommodating the wider data types required ar edesigin of the general-purpose register file. All 16 architectural registers are now at least 32 bits, and 8 of them are extended to 64 bits. In prior generations, a 32-bit value was formed by pairing two 16-bit registers, which is an awkward scheme that Heisenberg removes in favor of true wide registers. The _widening_ is one of the few backward incompatibilites, since code that assumed register pairs must be adjusted. The architecture defines clear rules for writes/reads of mixed size registers (e.g. writing a 16-bit value into a 64-bit register will zero out the upper 48 bits) to ensure deterministic behavior. The benefit is a cleaner programming model and support for 64-bit operands, at the cost of more silicon area for the larger registers. 

Numerous smaller tweaks support the above changes, for example the instruction set was pruned of seldom-used conversion operations to claw back area for the new SIMD hardware. New control registers and modes were added to managing things like FP8 format selection, and 64-bit addressing modes (for bigger memory and index ranges). There are new "task launch mechanisms" and "wavelet"-based triggers (specific to Cerebras' fabric architecture) updated to support 64-bit and the wider vector outputs, which allow the massive array of tiles to coordinate and dispatch work efficiently across the wafer. 

## Performance Tradeoffs

Heisenberg's architectural upgrades aim to boost per-tile throughput, especially for AI workloads but they come with carefull tradeoffs. The doubled SIMD width greatly improves raw compute bandwidth per cycle, _provided the software can supply enough parallel data_. In dense linear algebra or deep learning kernels this is achievable, yielding big speedups. However wider vectors also stress the memory subsystem (more bytes fetched per instruction) and can expose more altency if code has cross-lane dependencies (as noted with the 4-cycle bypass penalty). We mitigate some latency by introducing selective 2-cycle bypassing, which is helpful for acceleration accunulators in MAC-heavy loops (common in ML).

There is a modest frequency/power hit for the very widest operations. Historically, other high performance chips show that adding 512-bit execution can reduce max clock speed and increase power if not carefully managed (e.g. Intel's early AVX-512 implementations ran at lower frequencies because the 512-bit units were physically distant from the main pipeline and drew heavy power). HBG design uses physical partitioning (and conditional activation) of the wide SIMD units to minimize impact on common 16/32-bit operations. Because the wafer packs hundreds/thousands of tiles on a wafer, each tile must not bloat too much in area or power. 

"We conciously traded a small increase in instruction latency for huge area and power savings on each core, allowing us to fit more cores on the wafer and keep power per operation low, which is why Cerebras can offer $$210$$ times speedups over GPUs in certain tasks". 

## RTL Design

The 256-bit SIMD ALU is implemented as an array of lane units operating in parallel. For example, a 256-bit wide adder can be build as 16 times 16 bit adders that operate concurrently under a single vector instruction. Each lane handles a portion of the data (e.g. element 0 in lane0, element 1 in lane 1 etc.) and the RTL should be parameterized for different element widths (8, 16, 32, 64-bit) selecting the appropriate number of lanes. 

All lanes execute the same operation in lockstep, controlled by a single decoded instruction, which simplifies control logic. Care must be taken for SIMD mode control e.g. if an instruction is only 8-wide (32-bit each), the hardware might deactivate the unused lanes or merge results to reduce power. Ensuring data aligns to lane boundaries is important so that no lane is idle or handling misaligned chunks. For Heisenberg's special SIMD-32 mode (for 8-bit ops), this likely involves an extra set of 8 ALUs ("wart" datapath) or time-multiplexing existing lanes, which the RTL must integrate so that a single instruction can drive twice as many operations. 

**Pipeline Depth and Latency:** To minimize latency, we kept the pipeline relatively shallow for arithmetic operations (the goal being 2-3 cycle result latency for most ops). In RTL, this means combining logic stages carefully. For instance, an FMA (fused multiply-add) might be split into a multiple stage and an add stage with intermediate register, achiving a 2-cycle latency. Bypasses are then inserted from the output of the add stage back to the input of a subsequent operation. Wherever a single-cycle bypass wasn't feasible (due to wide datapath or long wire delay (*)), the architecture employs multi-cycle bypass with scoreboard control. 

(*) Modern wide SIMD architectures (e.g. Intel AVX-512, Cerebras Wafer Scale, NVIDIA Tensor Cores) adopt a 2-cycle bypass instead of a single-cycle one. This relaxes critical timing paths (i.e. the bypassed data is latched for 1 cycle reducing clock speed constraints), allows for larger datapaths (avoids wire congestion and fanout issues), reduces power consumption (fewer high speed, long-distance wires toggling in a single cycle), and enables higher clock frequencies (by avoiding the critical path delays of a single-cycle bypass the overall system can run at a higher frequency). Indeed, instead of forwarding the data immediately, the result is latched for **one additional cycle** before the dependent instruction consumes it. This way, instead of forcing long wires to close timing in one cycle, we insert an extra pipeline stage which allows data to be forwarded more efficiently. 

## DFT and Testability

