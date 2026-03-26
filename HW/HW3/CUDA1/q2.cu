
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
#include "timer.h"

// Kernel prototype (defined below) and helper to run+time scenarios
__global__ void AddVectors(const float* A, const float* B, float* C, int N);
static double run_scenario(dim3 grid, dim3 block, const float* dA, const float* dB, float* dC, int N);

// Variables for host and device vectors.
float* h_A; 
float* h_B; 
float* h_C; 
float* d_A; 
float* d_B; 
float* d_C; 

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

    printf("Total vector size: %d\n", N); 
    // size_t is the total number of bytes for a vector.
    size_t size = (size_t)N * sizeof(float);               

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    if (h_A == 0) Cleanup(false);
    h_B = (float*)malloc(size);
    if (h_B == 0) Cleanup(false);
    h_C = (float*)malloc(size);
    if (h_C == 0) Cleanup(false);

    // Allocate vectors in device memory.
    cudaError_t error;
    error = cudaMalloc((void**)&d_A, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&d_B, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&d_C, size);
    if (error != cudaSuccess) Cleanup(false);

    // Initialize host vectors h_A and h_B
    int i;
    for(i=0; i<N; ++i){
     h_A[i] = (float)i;
     h_B[i] = (float)(N-i);   
    }

    // Copy host vectors h_A and h_B to device vectores d_A and d_B
    error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);

    // Run three scenarios using the same kernel via run_scenario helper.
    double t1, t2, t3;

    // Scenario 1: one block, 1 thread
    {
        dim3 g(1);
        dim3 b(1);
        printf("Scenario 1: grid=(1) block=(1)\n");
        t1 = run_scenario(g, b, d_A, d_B, d_C, N);
    }

    // Scenario 2: one block, 256 threads
    {
        dim3 g(1);
        dim3 b(256);
        printf("Scenario 2: grid=(1) block=(256)\n");
        t2 = run_scenario(g, b, d_A, d_B, d_C, N);
    }

    // Scenario 3: multiple blocks, 256 threads per block covering N
    {
        int threadsPerBlock = 256;
        int blocksNeeded = (N + threadsPerBlock - 1) / threadsPerBlock;
        if (blocksNeeded < 1) blocksNeeded = 1;
        dim3 g((unsigned int)blocksNeeded);
        dim3 b((unsigned int)threadsPerBlock);
        printf("Scenario 3: grid=(%u) block=(%u) total threads=%zu\n", g.x, b.x, (size_t)g.x * (size_t)b.x);
        t3 = run_scenario(g, b, d_A, d_B, d_C, N);
    }

    double times[3] = {t1, t2, t3};
    for (int s = 0; s < 3; ++s) {
        double time = times[s];
        if (time <= 0.0) {
            printf("Scenario %d: error\n", s+1);
            continue;
        }
        int nFlops = N;
        double nFlopsPerSec = nFlops / time;
        double nGFlopsPerSec = nFlopsPerSec * 1e-9;
        double nBytes = (double)N * 3.0 * sizeof(float);
        double gBytesPerSec = (nBytes / time) / 1e9;
        printf("Scenario %d: Time: %lf s, GFlops: %lf, GBytes/s: %lf\n", s+1, time, nGFlopsPerSec, gBytesPerSec);
    }
     
    // Copy result from device memory to host memory
    error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) Cleanup(false);

    // Verify & report result
    for (i = 0; i < N; ++i) {
        float val = h_C[i];
        if (fabs(val - N) > 1e-5)
            break;
    }
    printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");

    // Clean up and exit.
    Cleanup(true);
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free device vectors
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
        
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

// Helper: launch kernel (with a warm-up), synchronize and return elapsed time in seconds.
static double run_scenario(dim3 grid, dim3 block, const float* dA, const float* dB, float* dC, int N)
{
    cudaError_t err;

    // Warm-up launch
    AddVectors<<<grid, block>>>(dA, dB, dC, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Warmup kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1.0;
    }
    cudaDeviceSynchronize();

    initialize_timer();
    start_timer();

    // Timed launch
    AddVectors<<<grid, block>>>(dA, dB, dC, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Timed kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1.0;
    }
    cudaDeviceSynchronize();

    stop_timer();
    return elapsed_time();
}