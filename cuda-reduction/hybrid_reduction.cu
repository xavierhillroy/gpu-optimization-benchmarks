/*
 * CUDA Hybrid Reduction (Binary Tree + Atomic)
 * * Implements a high-performance parallel reduction summation.
 * Architecture:
 * 1. Block-level reduction: Uses shared memory and binary tree reduction 
 * to sum elements within a thread block.
 * 2. Grid-level reduction: Thread 0 of each block atomically adds its 
 * partial sum to global memory.
 *
 * Compilation: nvcc -O2 hybrid_reduction.cu -o hybrid_reduction
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>

// =================================================================================
// 1. CUDA ERROR HANDLING & HELPER FUNCTIONS
// =================================================================================

#define ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void Is_GPU_present()
{
  int devid, devcount;
  cudaGetDevice(&devid);
  if (cudaGetDeviceCount(&devcount) || devcount==0)
  {
      printf ("No CUDA devices!\n");
      exit (1);
  }
  else
  { 
      cudaDeviceProp deviceProp; 
      cudaGetDeviceProperties (&deviceProp, devid);
      printf ("Device count, devid: %d %d\n", devcount, devid);
      printf ("Device: %s\n", deviceProp.name);
      printf ("Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
      // Optional: Print specific limits if needed
  }
  return;
}

// =================================================================================
// 2. TIMING HELPER FUNCTION
// =================================================================================

int timeval_subtract (double *result, struct timeval *x, struct timeval *y)
{
  struct timeval result0;

  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  result0.tv_sec = x->tv_sec - y->tv_sec;
  result0.tv_usec = x->tv_usec - y->tv_usec;
  *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;

  return x->tv_sec < y->tv_sec;
}

// =================================================================================
// 3. MAIN CUDA REDUCTION LOGIC
// =================================================================================

#define NTESTS 10
#define BLOCK_SIZE 256
#define NMAX 2097152
#define NBLOCKS (NMAX/BLOCK_SIZE)

// Input array (global host memory):
float h_A[NMAX];
// Copy of h_A on device:
__device__ float d_A[NMAX];
// Global accumulator
__device__ float d_sum;

// Kernel to reset the global sum
__global__ void init_kernel ()
{
  d_sum = 0.0;
  return;
}

// The Hybrid Reduction Kernel
__global__ void HybridReductionKernel() 
{
    // Shared memory for block-level reduction
    __shared__ float sum[BLOCK_SIZE];
    
    // Global thread index
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // Load data into shared memory (with boundary check safety)
    if (i < NMAX)
        sum[threadIdx.x] = d_A[i];
    else
        sum[threadIdx.x] = 0.0f;

    __syncthreads(); // Ensure all threads have loaded their data

    // Binary Tree Reduction within the block
    int nTotalThreads = blockDim.x; 
    while (nTotalThreads > 1) {
        int halfPoint = nTotalThreads / 2;
        if (threadIdx.x < halfPoint) {
            int thread2 = threadIdx.x + halfPoint;
            sum[threadIdx.x] += sum[thread2];
        }
        __syncthreads();
        nTotalThreads = halfPoint;
    }
    
    // Thread 0 atomically adds the block's partial sum to global memory
    if (threadIdx.x == 0) {
        atomicAdd(&d_sum, sum[0]);
    }
}

int main (int argc, char **argv)
{
  struct timeval tdr0, tdr1, tdr;
  double sum0, restime;
  float sum;

  // Check for GPU
  Is_GPU_present();

  double avr = 0.0;
  unsigned int seed = 111;

  // --- BENCHMARK LOOP ---
  for (int kk=0; kk<NTESTS; kk++)
  {
      // Initialize random data
      for (int i=0; i<NMAX; i++) {
          h_A[i] = (float)rand_r(&seed)/(float)RAND_MAX;
      }

      // Compute CPU Reference (Double Precision for Accuracy)
      sum0 = 0.0;
      for (int i=0; i<NMAX; i++)
          sum0 = sum0 + (double)h_A[i];

      // Copy to Device
      ERR( cudaMemcpyToSymbol( d_A, h_A, NMAX*sizeof(float), 0, cudaMemcpyHostToDevice) )

      // Reset Device Sum
      init_kernel <<< 1,1 >>> ();
      ERR( cudaDeviceSynchronize() )

      // --- TIMER START ---
      gettimeofday (&tdr0, NULL);

      // Launch Hybrid Kernel
      HybridReductionKernel <<<NBLOCKS, BLOCK_SIZE>>> ();
      ERR( cudaGetLastError() )

      // Copy result back
      ERR( cudaMemcpyFromSymbol (&sum, d_sum, sizeof(float), 0, cudaMemcpyDeviceToHost) )
      ERR( cudaDeviceSynchronize() )
      
      // --- TIMER END ---
      gettimeofday (&tdr1, NULL);
      
      timeval_subtract (&restime, &tdr1, &tdr0);

      if (kk == 0)
        printf ("GPU Sum: %e (Relative Error %e)\n", sum, fabs((double)sum-sum0)/sum0);

      avr = avr + restime;
  } 

  printf ("\nAverage Execution Time: %e seconds\n", avr/NTESTS);

  return 0;
}
