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
/// without using coalesced memory access.
/// 

// __global__ void AddVectors(const float* A, const float* B, float* C, int N)
// {
//     int t = threadIdx.x;
//     int blockStart = blockIdx.x * blockDim.x * N;
//     int totalSize = gridDim.x * blockDim.x * N;

//     for (int k = 0; k < N; ++k) {
//         int idx = blockStart + k * blockDim.x + t;
//         if (idx < totalSize) C[idx] = A[idx] + B[idx];
//     }
// }

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int vectorLength = blockDim.x * gridDim.x * N;

    for (int i = index; i < vectorLength; i+= stride) {
        C[i] = A[i] + B[i];
    }
}