# gpu-optimization-benchmarks
Collection of GPU-accelerated algorithms and performance benchmarks using CUDA C++ and OpenACC on NVIDIA A100s.

# High-Performance Computing Portfolio

This repository documents my work in optimizing compute-bound algorithms for parallel architectures. It focuses on porting CPU-based serial code to NVIDIA A100 GPUs using **CUDA C++** and **OpenACC**, with a specific focus on memory hierarchy optimization and warp utilization.

## Performance Benchmarks

| Project | Technology | Hardware | Baseline Time | Optimized Time | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Parallel Reduction (Sum)** | CUDA C++ | NVIDIA A100 | 4.687 ms (Atomic) | **0.065 ms** (Hybrid) | **72.1x** |
| **Prime Number Search** | OpenACC | NVIDIA A100 | 90.771 s (AMD EPYC) | **1.432 s** | **63.4x** |
| **Julia Set Visualization** | CUDA C++ | NVIDIA A100 | N/A (Serial CPU) | **Real-time** (2D Grid) | **Accelerated** |

---

## Project Breakdowns

### 1. Hybrid Parallel Reduction (CUDA)
**Goal:** Optimize a massive array summation by minimizing global memory contention.

**Performance Analysis:**
* **Serial CPU:** 1.761 ms
* **Naive GPU (Atomic):** 4.687 ms (*Slower due to serialization*)
* **Optimized GPU (Hybrid):** 0.065 ms (**27x faster than CPU**, **72x faster than Atomic**)

**The Bottleneck:**
A naive "Atomic-Only" approach on the GPU causes massive serialization as thousands of threads fight to update a single global address (`d_sum`).

**Optimization Strategy:**
* **Two-Stage Reduction:** Implemented a **Hybrid** approach combining Shared Memory with Global Atomics.
    1.  **Block-Level:** Threads perform a binary tree reduction in fast **Shared Memory** (`__shared__`), reducing 256 values to 1 per block.
    2.  **Grid-Level:** Only thread 0 of each block atomically adds to Global Memory.
* **Impact:** Reduced atomic collisions from **2,000,000** (one per thread) to just **8,192** (one per block).

**Code:** [View Source](./cuda-reduction/hybrid_reduction.cu)

---

### 2. Julia Set Fractal Visualization (CUDA)
**Goal:** Map a pixel-independent mathematical function across a 2D grid structure.

**Optimization Strategy:**
* **Coordinate Mapping:** Engineered a divergence-free thread mapping strategy where each CUDA thread calculates the color intensity for a unique `(x, y)` pixel coordinate.
* **Dynamic Grid:** Implemented flexible grid dimensioning `(DIM + block.x - 1) / block.x` to handle arbitrary image resolutions without memory access violations.
* **Result:** Offloaded intensive floating-point arithmetic to the GPU, allowing for high-resolution fractal generation that scales with core count.

**Code:** [View Source](./cuda-julia/julia_gpu.cu)

---

### 3. Accelerated Prime Number Generation (OpenACC)
**Goal:** Optimize a compute-heavy nested loop algorithm for identifying prime numbers.

**Optimization Strategy:**
* **Loop Collapsing:** Utilized `#pragma acc parallel loop collapse(2)` to flatten nested loops, maximizing thread saturation on the GPU.
* **Data Privatization:** Explicitly scoped variables (`x`, `y`, `ymax`, `success`) as `private` to prevent race conditions during parallel execution.
* **Parallel Reduction:** Implemented a sum reduction on the `count` variable to eliminate atomic collisions.

**Code:** [View Source](./openacc-primes/primes_count_acc.c)
