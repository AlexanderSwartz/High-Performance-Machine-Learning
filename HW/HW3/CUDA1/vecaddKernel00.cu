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

// This version results in poorer performance because each thread works on memory far apart from the other threads in the same block
// For example, for ValuesPerThread=500, thread 0 works on A[0] - A[499], thread 1 works on A[500] - A[999]...
__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{

    // blockDim.x * N  = threads per block * ValuesPerThread = values per block
    int blockStartIndex  = blockIdx.x * blockDim.x * N;
    int threadStartIndex = blockStartIndex + (threadIdx.x * N);
    int threadEndIndex   = threadStartIndex + N;
    int i;

    for( i=threadStartIndex; i<threadEndIndex; ++i ){
        C[i] = A[i] + B[i];
    }
}
