/* CUDA reduction exercise (summation in this case). The initial code
   reduction.cu is a CUDA code using the most primitive type of
   reduction - purely atomic.

   Your task is to

   1) Compile both the serial code and the purely atomic code, and
   measure the (in)efficiency of the purely atomic solution.

   2) Modify reduction.cu to implement a hybrid reduction approach:
   binary reduction at the low level (beginning of the kernel), atomic
   reduction at the top level (end of the kernel). More specifically, the
   results of binary reductions should be added up globally using
   atomicAdd function at the end of the kernel.


   The code always computes the "exact result" sum0 (using double
   precision, serially) - don't touch this part, it is needed to
   estmate the accuracy of your computation.

   The initial copying of the array to device is not timed. We are
   only interested in timing different reduction approaches.

   At the end, you will have to copy the reduction result (sum) from
   device to host, using cudaMemcpyFromSymbol.

   You will discover that for large NMAX, atomic summation is much
   slower than serial code. How about hybrid reduction?


To compile on graham:
  module load cuda
  nvcc -arch=sm_60 -O2 reduction.cu -o reduction


≈*/

#include "../part2.h"
#include "../cuda_errors.h"

// Number of times to run the test (for better timings accuracy):
#define NTESTS 1

// Number of threads in one block (possible range is 32...1024):
#define BLOCK_SIZE 256

// Total number of threads (total number of elements to process in the kernel):
#define NMAX 2097152

#define NBLOCKS NMAX/BLOCK_SIZE

// Input array (global host memory):
float h_A[NMAX];
// Copy of h_A on device:
__device__ float d_A[NMAX];

__device__ float d_sum;


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// Kernel(s) should go here:


// The only purpose of the kernel is to initialize one global variable, d_sum
__global__ void init_kernel ()
{
  d_sum = 0.0;
  return;
}


__global__ void MyKernel ()
{
 __shared__ float sum[BLOCK_SIZE];
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  sum[threadIdx.x] = d_A[i];
  __syncthreads(); // make sure all sum elements are initialized 
  int nTotalThreads = blockDim.x; // Num active thread
  while (nTotalThreads >1) {
    int halfPoint = nTotalThreads/2; // Binary reduction of active threads
    if (threadIdx.x < halfPoint){
    int thread2 = threadIdx.x + halfPoint;
    sum[threadIdx.x] += sum[thread2];
    }
  __syncthreads();
  nTotalThreads = halfPoint;
  }
  
  if (threadIdx.x == 0) 
  {  atomicAdd(&d_sum, sum[0]);}
  // Not needed, because NMAX is a power of two:
  //  if (i >= NMAX)
  //    return;


  return;
}




int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double sum0, restime;
  float sum;

  // Testing if a GPU is present, if yes - reporting main GPU details, if not - exiting:
  // This is included from cuda_errors.h (not part of CUDA!)
  Is_GPU_present();

// Loop to run the timing test multiple times:
  double avr = 0.0;
  int kk;
  unsigned int seed = 111;
for (kk=0; kk<NTESTS; kk++)
{

  // We don't initialize randoms, because we want to compare different strategies:
  // Initializing random number generator:
  //  srand((unsigned)time(0));

  // Initializing the input array:
  for (int i=0; i<NMAX; i++)
    {
      h_A[i] = (float)rand_r(&seed)/(float)RAND_MAX;
    }

  // Don't modify this: we need the double precision result to judge the accuracy:
  sum0 = 0.0;
  for (int i=0; i<NMAX; i++)
    sum0 = sum0 + (double)h_A[i];

  // Copying the data to device (we don't time it):
  ERR( cudaMemcpyToSymbol( d_A, h_A, NMAX*sizeof(float), 0, cudaMemcpyHostToDevice) )

// Set d_A to zero on device:
  init_kernel <<< 1,1 >>> ();

  //--------------------------------------------------------------------------------
  ERR( cudaDeviceSynchronize() )
  gettimeofday (&tdr0, NULL);   // First timing point


  // The kernel call:
  MyKernel <<<NBLOCKS, BLOCK_SIZE>>> ();


  // Copying the result back to host (we time it):
  ERR( cudaMemcpyFromSymbol (&sum, d_sum, sizeof(float), 0, cudaMemcpyDeviceToHost) )
  ERR( cudaDeviceSynchronize() )

  gettimeofday (&tdr1, NULL);  // Second timing point
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);
  if (kk == 0)
    printf ("Sum: %e (relative error %e)\n", sum, fabs((double)sum-sum0)/sum0);

  avr = avr + restime;
  //--------------------------------------------------------------------------------

} // kk loop

  printf ("\n Average time: %e\n", avr/NTESTS);

  return 0;

}

