/**
 * cuda_tsp.cu — GPU-accelerated Travelling Salesman Problem solver
 *
 * Solves TSP via brute-force Monte Carlo sampling on CUDA.  Each thread
 * generates random permutations of the city visit order and tracks the
 * shortest round-trip distance found.  Block-level reduction narrows the
 * per-thread results down to one best route per block; a final host-side
 * scan selects the global optimum.
 *
 * Key optimisations
 *   - Distance matrix cached in shared memory (fits for small N).
 *   - Per-thread sequential loop (NLOOP iterations) amortises launch overhead.
 *   - Pinned host memory for fast PCIe transfers.
 *   - IEEE-754 bit-reinterpretation trick for atomicMin on floats.
 *   - atomicCAS tiebreaker prevents race when two threads share the same
 *     minimum distance within a block.
 *
 * Build:
 *   nvcc -arch=sm_80 -O2 tsp.cu -o cuda_tsp
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <curand_kernel.h>

/* ---------------------------------------------------------------------------
 * Problem parameters
 * -------------------------------------------------------------------------*/
#define N_CITIES  11
#define N_MC      117440512        // total Monte Carlo trials
#define BSIZE     512              // threads per block
#define NBLOCKS   336              // thread blocks (multiple of 56 SMs)
#define NLOOP     (N_MC / (BSIZE * NBLOCKS))
#define SIZE      1000             // side length (km) of the city region

/* Host-side city data */
static float x[N_CITIES], y[N_CITIES];
static float dist[N_CITIES][N_CITIES];

/* ---------------------------------------------------------------------------
 * Portable wall-clock timer
 * -------------------------------------------------------------------------*/
static int timeval_subtract(double *result,
                            struct timeval *end, struct timeval *start)
{
    if (end->tv_usec < start->tv_usec) {
        int nsec = (start->tv_usec - end->tv_usec) / 1000000 + 1;
        start->tv_usec -= 1000000 * nsec;
        start->tv_sec  += nsec;
    }
    if (end->tv_usec - start->tv_usec > 1000000) {
        int nsec = (start->tv_usec - end->tv_usec) / 1000000;
        start->tv_usec += 1000000 * nsec;
        start->tv_sec  -= nsec;
    }
    struct timeval diff;
    diff.tv_sec  = end->tv_sec  - start->tv_sec;
    diff.tv_usec = end->tv_usec - start->tv_usec;
    *result = (double)diff.tv_sec + (double)diff.tv_usec / 1e6;
    return end->tv_sec < start->tv_sec;
}

/* ---------------------------------------------------------------------------
 * TSP kernel — one block produces one (distance, route) candidate
 * -------------------------------------------------------------------------*/
__global__ void tsp_kernel(const float *__restrict__ dist_d,
                           float *global_best_dists,
                           int   *global_best_routes)
{
    const int global_idx = threadIdx.x + blockDim.x * blockIdx.x;

    /* Per-thread cuRAND state */
    curandState rng;
    curand_init(111ULL, global_idx, 0, &rng);

    /* Load the distance matrix into shared memory (flat N×N layout) */
    __shared__ float s_dist[N_CITIES * N_CITIES];
    if (threadIdx.x < N_CITIES * N_CITIES)
        s_dist[threadIdx.x] = dist_d[threadIdx.x];
    __syncthreads();

    /* Block-level best distance (initialised by thread 0) */
    __shared__ float s_block_best;
    if (threadIdx.x == 0)
        s_block_best = 1e30f;
    __syncthreads();

    /* Thread-local Monte Carlo search */
    int perm[N_CITIES];
    int best_perm[N_CITIES];
    float best_d = 1e30f;
    perm[0] = 0;   // city 0 is always the starting point

    for (int k = 0; k < NLOOP; k++) {
        /* Reset identity permutation for cities 1..N-1 */
        for (int l = 1; l < N_CITIES; l++)
            perm[l] = l;

        /* Fisher–Yates shuffle on positions 1..N-2, accumulating distance */
        float d = 0.0f;
        for (int l = 1; l < N_CITIES - 1; l++) {
            int l1 = l + curand(&rng) % (N_CITIES - l);
            int tmp  = perm[l];
            perm[l]  = perm[l1];
            perm[l1] = tmp;

            d += s_dist[perm[l - 1] * N_CITIES + perm[l]];
        }

        /* Close the loop: last→first and second-last→last edges */
        d += s_dist[perm[N_CITIES - 1] * N_CITIES + perm[0]];
        d += s_dist[perm[N_CITIES - 1] * N_CITIES + perm[N_CITIES - 2]];

        if (d < best_d) {
            best_d = d;
            for (int l = 0; l < N_CITIES; l++)
                best_perm[l] = perm[l];
        }
    }

    /*
     * Block-level reduction via atomicMin.
     * IEEE-754 guarantees that for positive floats, integer bit ordering
     * preserves the floating-point comparison, so reinterpreting as int
     * lets us use the hardware integer atomicMin.
     */
    atomicMin((int *)&s_block_best, __float_as_int(best_d));
    __syncthreads();

    /*
     * Identify the winning thread.  atomicCAS on a sentinel (-1) ensures
     * exactly one thread writes its route even if multiple threads share
     * the same minimum distance.
     */
    __shared__ int s_winner_route[N_CITIES];
    __shared__ int s_winner_tid;
    if (threadIdx.x == 0)
        s_winner_tid = -1;
    __syncthreads();

    if (__float_as_int(best_d) == __float_as_int(s_block_best)) {
        int prev = atomicCAS(&s_winner_tid, -1, threadIdx.x);
        if (prev == -1) {
            for (int i = 0; i < N_CITIES; i++)
                s_winner_route[i] = best_perm[i];
        }
    }
    __syncthreads();

    /* Write the block's result to global memory */
    if (threadIdx.x == 0)
        global_best_dists[blockIdx.x] = s_block_best;

    if (threadIdx.x < N_CITIES)
        global_best_routes[blockIdx.x * N_CITIES + threadIdx.x] =
            s_winner_route[threadIdx.x];
}

/* ---------------------------------------------------------------------------
 * Host driver
 * -------------------------------------------------------------------------*/
int main(int argc, char **argv)
{
    struct timeval t0, t1;
    double elapsed;
    int perm_min[N_CITIES];

    /* Deterministically scatter cities in a SIZE×SIZE region */
    unsigned int seed = 222;
    for (int i = 0; i < N_CITIES; i++) {
        x[i] = (float)SIZE * (float)rand_r(&seed) / ((float)RAND_MAX + 1.0f);
        y[i] = (float)SIZE * (float)rand_r(&seed) / ((float)RAND_MAX + 1.0f);
    }

    /* Build symmetric distance matrix */
    for (int i = N_CITIES - 1; i >= 0; i--) {
        for (int j = 0; j < N_CITIES; j++) {
            if (j < i)
                dist[i][j] = sqrtf(powf(x[j] - x[i], 2) +
                                    powf(y[j] - y[i], 2));
            else if (j == i)
                dist[i][j] = 0.0f;
            else
                dist[i][j] = dist[j][i];
        }
    }

    /* ---- Memory allocation ------------------------------------------------*/
    const size_t dist_bytes   = sizeof(dist);
    const size_t dists_bytes  = sizeof(float) * NBLOCKS;
    const size_t routes_bytes = sizeof(int)   * NBLOCKS * N_CITIES;

    float *d_dist, *d_best_dists;
    int   *d_best_routes;
    cudaMalloc(&d_dist,        dist_bytes);
    cudaMalloc(&d_best_dists,  dists_bytes);
    cudaMalloc(&d_best_routes, routes_bytes);

    float *h_best_dists;
    int   *h_best_routes;
    cudaMallocHost(&h_best_dists,  dists_bytes);
    cudaMallocHost(&h_best_routes, routes_bytes);

    cudaHostRegister(dist, dist_bytes, cudaHostRegisterDefault);

    /* ---- Timed region: transfer → compute → transfer → reduce ------------*/
    gettimeofday(&t0, NULL);

    cudaMemcpy(d_dist, dist, dist_bytes, cudaMemcpyHostToDevice);
    tsp_kernel<<<NBLOCKS, BSIZE>>>(d_dist, d_best_dists, d_best_routes);
    cudaMemcpy(h_best_dists,  d_best_dists,  dists_bytes,  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_routes, d_best_routes, routes_bytes, cudaMemcpyDeviceToHost);

    gettimeofday(&t1, NULL);
    timeval_subtract(&elapsed, &t1, &t0);

    /* Host-side global reduction across blocks */
    float  best_dist = 1e30f;
    int    best_block = 0;
    for (int i = 0; i < NBLOCKS; i++) {
        if (h_best_dists[i] < best_dist) {
            best_dist  = h_best_dists[i];
            best_block = i;
        }
    }

    const int row_offset = best_block * N_CITIES;
    for (int i = 0; i < N_CITIES; i++)
        perm_min[i] = h_best_routes[row_offset + i];

    /* ---- Results ----------------------------------------------------------*/
    printf("Shortest total distance: %f\n", best_dist);
    printf("Shortest itenerary: ");
    for (int l = 0; l < N_CITIES; l++)
        printf("%d ", perm_min[l]);
    printf("\n");
    printf("time: %e\n", elapsed);

    /* Write itinerary coordinates for external plotting */
    FILE *fp = fopen("tsp.dat", "w");
    for (int l = 0; l < N_CITIES; l++)
        fprintf(fp, "%f %f\n", x[perm_min[l]], y[perm_min[l]]);
    fprintf(fp, "%f %f\n", x[perm_min[0]], y[perm_min[0]]);
    fclose(fp);

    /* ---- Cleanup ---------------------------------------------------------*/
    cudaFree(d_dist);
    cudaFree(d_best_dists);
    cudaFree(d_best_routes);
    cudaFreeHost(h_best_dists);
    cudaFreeHost(h_best_routes);
    cudaHostUnregister(dist);

    return 0;
}
