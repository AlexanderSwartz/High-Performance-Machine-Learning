
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
// run_scenario now reports warmup and timed kernel separately
static void run_kernel(dim3 grid, dim3 block, const float* dA, const float* dB, float* dC, int N,
                         double* out_warmup, double* out_kernel);

// Utility Functions
void Cleanup(bool);
void checkCUDAError(const char *msg);

float* d_A;
float* d_B;
float* d_C;

// Helper to run one managed-memory scenario: init inputs, run kernel, verify
static void run_scenario(dim3 grid, dim3 block, int N,
                                 double* out_warmup, double* out_kernel, int scenario_id)
{
    // Reinitialize inputs before each scenario since inputs are identical across scenarios
    for(int i=0; i<N; ++i){
        d_A[i] = (float)i;
        d_B[i] = (float)(N-i);
    }

    // kernel: measure warmup and timed kernel separately (write directly into caller's buffers)
    run_kernel(grid, block, d_A, d_B, d_C, N, out_warmup, out_kernel);

    int i;
    for (i = 0; i < N; ++i) {
        float val = d_C[i];
        if (fabs(val - N) > 1e-5)
            break;
    }
    printf("Scenario %d verify %s\n", scenario_id, (i == N) ? "PASSED" : "FAILED");
}

// Host code performs setup and calls the kernel.
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Usage: %s K \n", argv[0]);
        printf("K is the size in millions of the vectors to be added.\n");
        exit(0);
    }
    unsigned long long K = 0ULL;
    sscanf(argv[1], "%llu", &K);
    int N = (int)(K * 1000000ULL);

    // size_t is the total number of bytes for a vector.
    size_t size = (size_t)N * sizeof(float);
    cudaError_t error;

    // Allocate managed memory (timing removed — only kernel timings are kept)
    error = cudaMallocManaged((void**)&d_A, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMallocManaged((void**)&d_B, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMallocManaged((void**)&d_C, size);
    if (error != cudaSuccess) Cleanup(false);

    // Initialize inputs once (touch pages) since inputs are identical across scenarios
    for(int i=0; i<N; ++i){
        d_A[i] = (float)i;
        d_B[i] = (float)(N-i);
    }

    // We'll reinitialize inputs before each scenario; measure warmup and timed kernel only.
    double t_warmup_arr[3] = {0.0, 0.0, 0.0};
    double t_kernel_arr[3] = {0.0, 0.0, 0.0};

    run_scenario(dim3(1), dim3(1), N, &t_warmup_arr[0], &t_kernel_arr[0], 1);
    run_scenario(dim3(1), dim3(256), N, &t_warmup_arr[1], &t_kernel_arr[1], 2);
    
    int threadsPerBlock = 256;
    int blocksNeeded = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksNeeded < 1) blocksNeeded = 1;
    run_scenario(dim3((unsigned int)blocksNeeded), dim3((unsigned int)threadsPerBlock), N,
            &t_warmup_arr[2], &t_kernel_arr[2], 3);

    // Only report warmup and kernel times per scenario.
    printf("Scenario warmup times (s): 1=%.6f 2=%.6f 3=%.6f\n", t_warmup_arr[0], t_warmup_arr[1], t_warmup_arr[2]);
    printf("Scenario kernel times (s): 1=%.6f 2=%.6f 3=%.6f\n", t_kernel_arr[0], t_kernel_arr[1], t_kernel_arr[2]);

    // Use centralized Cleanup to free resources and exit.
    Cleanup(true);
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free vectors
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
static void run_kernel(dim3 grid, dim3 block, const float* dA, const float* dB, float* dC, int N,
                         double* out_warmup, double* out_kernel)
{
    // Warm-up launch (timed)
    initialize_timer(); start_timer();
    AddVectors<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();
    stop_timer();
    *out_warmup = elapsed_time();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);

    // Timed launch
    initialize_timer(); start_timer();
    AddVectors<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();
    stop_timer();
    *out_kernel = elapsed_time();
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
}