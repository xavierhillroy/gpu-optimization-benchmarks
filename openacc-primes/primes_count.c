
/* Counting prime numbers. Serial version.

Compiling instructions:
  module load nvhpc/25.1

 - Serial code:
  nvc -O3 primes_count.c -o primes_count

 - OpenACC code (for A100; cc90 for H100):
  nvc -O3 -acc -gpu=cc80 -Minfo=accel primes_count_acc.c -o primes_count_acc

*/

#include "part2.h"

// Range of k-numbers for primes search:
#define KMIN 1
// Should be smaller than 357,913,941 (because we are using signed int)
#define KMAX 30000000


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double restime;
  int success;
  int ymax, x, y, k, j, count;

  gettimeofday (&tdr0, NULL);


  count = 0;

  for (k=KMIN; k<=KMAX; k++)
    {
      // testing "-1" and "+1" cases:
      for (j=-1; j<2; j=j+2)
	{
	  // Prime candidate:
	  x = 6*k + j;
	  // We should be dividing by numbers up to sqrt(x):
	  ymax = (int)ceil(sqrt((double)x));

	  // Primality test:
          y = 3;
          success = 1;
	  while (success && y<=ymax)
	    {
	      // To be a success, the modulus should not be equal to zero:
	      success = x % y;
              y = y + 2;
	    }

	  if (success)
	    {
	      count++;
	    }
	}
    }

  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);
  printf ("N_primes: %d\n", count);
  printf ("time: %e\n", restime);
  //--------------------------------------------------------------------------------



  return 0;

}
