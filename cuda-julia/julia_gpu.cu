/*
 * Julia Set Visualization (GPU)
 * * Computes a 2D visualization of the Julia set using CUDA.
 * * Mapping: Each thread computes the color for one pixel (x,y).
 * * Output: A data file 'julia.dat' viewable with Gnuplot.
 *
 * Compilation: nvcc -O2 julia_gpu.cu -o julia_gpu
 */

#include "stdio.h"
#include <cuda_runtime.h>

#define DIM 1000

// =================================================================================
// ERROR HANDLING MACRO
// =================================================================================
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// =================================================================================
// DEVICE FUNCTION: JULIA SET MATHEMATICS
// =================================================================================
__device__ int julia(int x, int y) {
    const float scaling = 1.5;
    float scaled_x = scaling * (float)(DIM/2 - x)/(DIM/2);
    float scaled_y = scaling * (float)(DIM/2 - y)/(DIM/2);

    float c_real = -0.8f;
    float c_imag = 0.156f;

    float z_real = scaled_x;
    float z_imag = scaled_y;
    float z_real_tmp;

    int iter = 0;
    for(iter = 0; iter < 200; iter++) {
        z_real_tmp = z_real;
        z_real = (z_real * z_real - z_imag * z_imag) + c_real;
        z_imag = 2.0f * z_real_tmp * z_imag + c_imag;

        if((z_real * z_real + z_imag * z_imag) > 1000)
            return 0;
    }

    return 1;
}

// =================================================================================
// KERNEL: PARALLEL EXECUTION
// =================================================================================
__global__ void kernel(int* d_a) {
    // Map thread/block indices to pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check (Critical for stability if grid > DIM)
    if (x < DIM && y < DIM) {
        int i = x + y * DIM;
        d_a[i] = julia(x, y);
    }
}

// =================================================================================
// MAIN HOST CODE
// =================================================================================
int main(void) {
    int *array;
    int *d_a;
    FILE *out;
    size_t memsize = DIM * DIM * sizeof(int);

    // Allocate Host and Device Memory
    cudaCheckError(cudaMallocHost((void **)&array, memsize));
    cudaCheckError(cudaMalloc((void **) &d_a, memsize));

    // Define Grid Strategy (Standard 16x16 threads per block)
    dim3 blockDef(16, 16);
    dim3 gridDef((DIM + blockDef.x - 1) / blockDef.x, 
                 (DIM + blockDef.y - 1) / blockDef.y);

    printf("Computing Julia Set on GPU with Grid: %dx%d blocks\n", gridDef.x, gridDef.y);

    // Launch Kernel
    kernel<<<gridDef, blockDef>>>(d_a);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copy Result Back
    cudaCheckError(cudaMemcpy(array, d_a, memsize, cudaMemcpyDeviceToHost));

    // Write Output to File
    out = fopen("julia.dat", "w");
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;
            if (array[offset] == 1)
                fprintf(out, "%d %d \n", x, y);
        }
    }
    fclose(out);
    printf("Done. Output written to julia.dat\n");

    // Cleanup
    cudaFree(array);
    cudaFree(d_a);
    
    return 0;
}
