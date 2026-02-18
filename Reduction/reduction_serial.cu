/* CUDA reduction exercise. The serial version (written as *.cu
file). Use it to compare your CUDA code performance to a serial code
performance.

To compile on graham, cedar etc:
  module load cuda
  nvcc -arch=sm_80 -O2 reduction_serial.cu -o reduction_serial

*/

#include "../part2.h"
#include "../cuda_errors.h"


// Number of times to run the test (for better timings accuracy):
#define NTESTS 100

// Number of threads in one block (possible range is 32...1024):
#define BLOCK_SIZE 256

// Total number of threads (total number of elements to process in the kernel):
// For simplicity, use a power of two:
#define NMAX 2097152

// Number of blocks
// This will be needed for the second kernel in two-step binary reduction
// (to declare a shared memory array)
#define NBLOCKS NMAX/BLOCK_SIZE


// Input array (global host memory):
float h_A[NMAX];
// Copy of h_A on device:
__device__ float d_A[NMAX];


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// Kernel(s) should go here:


int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double sum0, restime;
  float sum;

  // Testing if a GPU is present, if yes - reporting main GPU details, if not - exiting:
  // This is included from cuda_errors.h (not part of CUDA!)
  Is_GPU_present();


  if (BLOCK_SIZE>1024)
    {
      printf ("Bad BLOCK_SIZE: %d\n", BLOCK_SIZE);
      exit (1);
    }


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
  ERR( cudaMemcpyToSymbol (d_A, h_A, NMAX*sizeof(float), 0, cudaMemcpyHostToDevice) )

  //--------------------------------------------------------------------------------
  ERR( cudaDeviceSynchronize() )
  gettimeofday (&tdr0, NULL);


  // This serial summation will have to be replaced by calls to kernel(s):
  sum = 0.0;
  for (int i=0; i<NMAX; i++)
    sum = sum + h_A[i];


  ERR( cudaDeviceSynchronize() )
  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);

  // We are printing the result here, after cudaDeviceSynchronize (this will matter
  // for CUDA code - why?)
  if (kk == 0)
    printf ("Sum: %e (relative error %e)\n", sum, fabs((double)sum-sum0)/sum0);

  avr = avr + restime;
  //--------------------------------------------------------------------------------

} // kk loop
  printf ("\n Average time: %e\n", avr/NTESTS);

  return 0;

}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

