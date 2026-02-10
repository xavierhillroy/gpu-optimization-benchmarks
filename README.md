# gpu-optimization-benchmarks
Collection of GPU-accelerated algorithms and performance benchmarks using CUDA, OpenACC, and MPI on NVIDIA A100s.

# High-Performance Computing Portfolio

This repository documents my work in optimizing compute-bound algorithms for parallel architectures. It focuses on porting CPU-based serial code to NVIDIA A100 GPUs using **CUDA** and **OpenACC**, with a specific focus on memory hierarchy optimization and warp utilization.

## erformance Benchmarks

| Project | Technology | Hardware | Baseline (CPU) | Optimized (GPU) | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Prime Number Search** | OpenACC | NVIDIA A100 | X.XXs (Xeon) | Y.YYs | **63x** |


---

## 📂 Project Breakdowns

### 1. Accelerated Prime Number Generation (OpenACC)
**Goal:** Optimize a compute-heavy nested loop algorithm for identifying prime numbers.

**Optimization Strategy:**
* **Loop Collapsing:** Utilized `#pragma acc parallel loop collapse(2)` to flatten nested loops, maximizing thread saturation on the GPU.
* **Data Privatization:** Explicitly scoped variables (`x`, `y`, `ymax`, `success`) as `private` to prevent race conditions during parallel execution.
* **Parallel Reduction:** Implemented a sum reduction on the `count` variable to eliminate atomic collisions.

**Code:** [View Source](./openacc-primes/primes_count_acc.c)

