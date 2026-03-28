
///
/// vecadd.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-02-03
/// Last Modified: 2011-03-03 DVN
///
/// Add two Vectors A and B in C on GPU using
/// a kernel defined according to vecAddKernel.h
/// Students must not modify this file. The GTA
/// will grade your submission using an unmodified
/// copy of this file.
/// 

// Includes
#include <stdio.h>
#include <math.h>
#include "timer.h"

// Kernel prototype (defined below) and helper to run+time scenarios
__global__ void AddVectors(const float* A, const float* B, float* C, int N);
static double run_scenario(dim3 grid, dim3 block, const float* dA, const float* dB, float* dC, int N);

// (helper moved below variable declarations to ensure globals are visible)

// Variables for host and device vectors.
float* h_A;
float* h_B;
float* h_C;
float* d_A;
float* d_B;
float* d_C;

// Helper to run one managed-memory scenario: init inputs, run kernel, verify
static void execute_scenario_q3(dim3 grid, dim3 block, int N,
                                 double* out_init, double* out_kernel, int scenario_id)
{
    size_t size = (size_t)N * sizeof(float);

    // init (touch / first-touch) on host
    initialize_timer(); start_timer();
    for (int j = 0; j < N; ++j) {
        d_A[j] = (float)j;
        d_B[j] = (float)(N - j);
    }
    stop_timer(); double t_touch = elapsed_time();

    // Prefetch to current device and include prefetch time in out_init
    int dev = 0;
    cudaGetDevice(&dev);
    initialize_timer(); start_timer();
    cudaMemPrefetchAsync(d_A, size, dev);
    cudaMemPrefetchAsync(d_B, size, dev);
    cudaDeviceSynchronize();
    stop_timer(); double t_prefetch = elapsed_time();
    if (out_init) *out_init = t_touch + t_prefetch;

    // kernel
    double t_kernel = run_scenario(grid, block, d_A, d_B, d_C, N);
    if (out_kernel) *out_kernel = t_kernel;

    int i;
    for (i = 0; i < N; ++i) {
        float val = d_C[i];
        if (fabs(val - N) > 1e-5)
            break;
    }
    printf("Scenario %d verify %s\n", scenario_id, (i == N) ? "PASSED" : "FAILED");
}

// Utility Functions
void Cleanup(bool);
void checkCUDAError(const char *msg);

// Host code performs setup and calls the kernel.
int main(int argc, char** argv)
{
    // int ValuesPerThread; // number of values per thread
    
	// Parse arguments.
    if (argc != 2)
    {
        printf("Usage: %s K \n", argv[0]);
        printf("K is the size in millions of the vectors to be added.\n");
        exit(0);
    }
    unsigned long long K = 0ULL;
    sscanf(argv[1], "%llu", &K);
    int N = (int)(K * 1000000ULL);

    // TO DO: Calculate the number of threads/blocks to use by reversing this
    // N = ValuesPerThread * GridWidth * BlockWidth;

    // size_t is the total number of bytes for a vector.
    size_t size = (size_t)N * sizeof(float);
    double t_alloc = 0.0, t_free = 0.0;
    cudaError_t error;

    // Always use per-scenario prefetch (no additional program arguments required)

    // Allocate managed memory (time allocation)
    initialize_timer(); start_timer();
    error = cudaMallocManaged((void**)&d_A, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMallocManaged((void**)&d_B, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMallocManaged((void**)&d_C, size);
    if (error != cudaSuccess) Cleanup(false);
    stop_timer(); t_alloc = elapsed_time();

    /* no one-time prefetch; per-scenario prefetch occurs inside execute_scenario_q3 */

    // We'll reinitialize inputs before each scenario to include per-scenario page/migration effects.
    double t_init_arr[3] = {0.0, 0.0, 0.0};
    double t_kernel_arr[3] = {0.0, 0.0, 0.0};

    // Scenario 1
    execute_scenario_q3(dim3(1), dim3(1), N, &t_init_arr[0], &t_kernel_arr[0], 1);

    // Scenario 2
    execute_scenario_q3(dim3(1), dim3(256), N, &t_init_arr[1], &t_kernel_arr[1], 2);

    // Scenario 3: multiple blocks
    {
        int threadsPerBlock = 256;
        int blocksNeeded = (N + threadsPerBlock - 1) / threadsPerBlock;
        if (blocksNeeded < 1) blocksNeeded = 1;
        execute_scenario_q3(dim3((unsigned int)blocksNeeded), dim3((unsigned int)threadsPerBlock), N,
                    &t_init_arr[2], &t_kernel_arr[2], 3);
    }

    double t_init_total = t_init_arr[0] + t_init_arr[1] + t_init_arr[2];

    // Free managed memory (time free)
    initialize_timer(); start_timer();
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);
    d_A = d_B = d_C = NULL;
    stop_timer(); t_free = elapsed_time();

        printf("Scenario kernel times (s): 1=%.6f 2=%.6f 3=%.6f\n", t_kernel_arr[0], t_kernel_arr[1], t_kernel_arr[2]);
        double t_combined = t_alloc + t_init_total + t_free;

        // Per-scenario total program time: include allocation, per-scenario init (first-touch), kernel, and free
        double t_total_arr[3];
        for (int s = 0; s < 3; ++s) {
            t_total_arr[s] = t_alloc + t_init_arr[s] + t_kernel_arr[s] + t_free;
        }
        printf("Scenario total times (s): 1=%.6f 2=%.6f 3=%.6f\n", t_total_arr[0], t_total_arr[1], t_total_arr[2]);
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free managed/device memory
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);
        
    error = cudaDeviceReset();
    
    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      exit(-1);
    }                         
}

// Grid-stride kernel: each thread processes multiple elements spaced by stride
__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// Helper: launch kernel, synchronize and return elapsed time in seconds.
static double run_scenario(dim3 grid, dim3 block, const float* dA, const float* dB, float* dC, int N)
{
    cudaError_t err;
    initialize_timer();
    start_timer();

    AddVectors<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize(); // wait for completion

    stop_timer();
    double kernelTime = elapsed_time();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1.0;
    }
    return kernelTime;
}