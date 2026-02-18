


/*
This code computes a visualization of the Julia set.  Specifically, it computes a 2D array of pixels.

The data can be viewed with gnuplot.

The Julia set iteration is:

z= z**2 + C

If it converges, then the initial point z is in the Julia set.

This code is CPU only but will compile with:

module load cuda
nvcc -O2 julia_cpu.cu

 
*/


#include "stdio.h"
#include <cuda.h>
#define DIM 1000


__device__ int julia(int x, int y){
    const float scaling = 1.5;
    float scaled_x = scaling * (float)(DIM/2 - x)/(DIM/2);
    float scaled_y = scaling * (float)(DIM/2 - y)/(DIM/2);

    float c_real=-0.8f;
    float c_imag=0.156f;

    float z_real=scaled_x;
    float z_imag=scaled_y;
    float z_real_tmp;

    int iter=0;
    for(iter=0; iter<200; iter++){

        z_real_tmp = z_real;
        z_real =(z_real*z_real-z_imag*z_imag) +c_real;
        z_imag = 2.0f*z_real_tmp*z_imag + c_imag;

        if( (z_real*z_real+z_imag*z_imag) > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel(int* d_a  ){
    int x,y;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y *blockDim.y + threadIdx.y;
   int  i = x + y * DIM;
   int juliaValue = julia(x,y);
  // int offset = x + y * DIM;
   d_a[i] = juliaValue;
     
    //for (y=0; y<DIM; y++) {
      //  for (x=0; x<DIM; x++) {
         //   int offset = x + y * DIM;

          //  int juliaValue = julia( x, y );
            //arr[offset] = juliaValue;
        //}
    //}
}

int main( void ) {
    int x,y;
    int *array;
    int *d_a;
    FILE *out;
    size_t memsize;

    memsize = DIM * DIM * sizeof(int);

    cudaMallocHost((void **)&array, memsize);
    cudaMalloc((void**) &d_a, memsize);
    //__device__ arr[DIM][DIM];
    

   dim3 gridDef(100,100,1);
   dim3 blockDef(10,10,1);
   kernel<<<gridDef, blockDef>>>(d_a);
   cudaMemcpy(array, d_a, memsize,cudaMemcpyDefault); 

    out = fopen( "julia.dat", "w" );
    for (y=0; y<DIM; y++) {
        for (x=0; x<DIM; x++) {
            int offset = x + y * DIM;
            if(array[offset]==1)
                fprintf(out,"%d %d \n",x,y);  
        } 
    } 
    fclose(out);

   cudaFree(array);
   cudaFree(d_a);
}

