///
/// vecAddKernel00.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// By David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-16 DVN
///
/// This Kernel adds two Vectors A and B in C on GPU
/// WITH  coalesced memory access.

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // stride by total number of threads
    // iter 1: thread[0] computes C[0], thread[1] computes C[1]...
    // iter 2: thread[0] computes C[stride], thread[1] computes C[stride+1]...
    int stride = blockDim.x * gridDim.x;
    int vectorLength = blockDim.x * gridDim.x * N;

    for (int i = index; i < vectorLength; i+= stride) {
        C[i] = A[i] + B[i];
    }
}
