///
/// matmultKernel00.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments. 
///

#include "matmultKernel.h"
#include <stdio.h>

#define COARSE_FACTOR 4

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Csub points to the top-left of the 32x32 output block
    float* Csub = &C.elements[C.stride * FOOTPRINT_SIZE * by + FOOTPRINT_SIZE * bx];

    // 1. ALLOCATE REGISTERS FOR THE 4 OUTPUTS
    // c_00: top-left, c_01: top-right, c_10: bottom-left, c_11: bottom-right
    float c_00 = 0.0f;
    float c_01 = 0.0f;
    float c_10 = 0.0f;
    float c_11 = 0.0f;

    // Loop over all sub-matrices of A and B
    for (int m = 0; m < (A.width / FOOTPRINT_SIZE); ++m) {
        
        // Asub and Bsub point to the top-left of the current 32x32 input tiles
        float* Asub = &A.elements[A.stride * FOOTPRINT_SIZE * by + FOOTPRINT_SIZE * m];
        float* Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * bx];

        __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
        __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

        // 2. COLLABORATIVE LOAD (Strided for Coalescing)
        // Each 16x16 thread block loads 1 element for each of the 4 quadrants.
        // Because 'tx' is contiguous for threads 0-15, these reads result in 
        // clean 64-byte coalesced memory transactions!
        shared_A[ty][tx]           = Asub[ty * A.stride + tx];                   // Top-Left quadrant
        shared_A[ty][tx + BLOCK_SIZE]      = Asub[ty * A.stride + tx + BLOCK_SIZE];              // Top-Right quadrant
        shared_A[ty + BLOCK_SIZE][tx]      = Asub[(ty + BLOCK_SIZE) * A.stride + tx];            // Bottom-Left quadrant
        shared_A[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = Asub[(ty + BLOCK_SIZE) * A.stride + tx + BLOCK_SIZE];       // Bottom-Right quadrant

        shared_B[ty][tx]           = Bsub[ty * B.stride + tx];
        shared_B[ty][tx + BLOCK_SIZE]      = Bsub[ty * B.stride + tx + BLOCK_SIZE];
        shared_B[ty + BLOCK_SIZE][tx]      = Bsub[(ty + BLOCK_SIZE) * B.stride + tx];
        shared_B[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = Bsub[(ty + BLOCK_SIZE) * B.stride + tx + BLOCK_SIZE];
        
        __syncthreads();

        // 3. COMPUTE USING MAXIMUM REGISTER REUSE
        #pragma unroll
        for (int e = 0; e < FOOTPRINT_SIZE; ++e) {
            
            // Fetch 2 elements from A and 2 from B into extremely fast registers
            float a_top    = shared_A[ty][e];
            float a_bottom = shared_A[ty + BLOCK_SIZE][e];
            float b_left   = shared_B[e][tx];
            float b_right  = shared_B[e][tx + BLOCK_SIZE];

            // Use those 4 registers to do 4 separate math operations!
            c_00 += a_top * b_left;
            c_01 += a_top * b_right;
            c_10 += a_bottom * b_left;
            c_11 += a_bottom * b_right;
        }

        __syncthreads();
    }

    // 4. WRITE BACK TO GLOBAL MEMORY (Strided for Coalescing)
    // Debug: print which global C elements this thread (0,0) will write
    // if (ty == 0 && tx == 0 && by == 0 && bx == 0) {
    //     int base_row = by * FOOTPRINT_SIZE;
    //     int base_col = bx * FOOTPRINT_SIZE;
    //     printf("Block (%d,%d) thread (0,0) writing: C[%d,%d]=c_00, C[%d,%d]=c_01, C[%d,%d]=c_10, C[%d,%d]=c_11\n",
    //            by, bx,
    //            base_row, base_col,
    //            base_row, base_col + BLOCK_SIZE,
    //            base_row + BLOCK_SIZE, base_col,
    //            base_row + BLOCK_SIZE, base_col + BLOCK_SIZE);
    // }

    Csub[ty * C.stride + tx]                               = c_00;
    Csub[ty * C.stride + tx + BLOCK_SIZE]                  = c_01;
    Csub[(ty + BLOCK_SIZE) * C.stride + tx]                = c_10;
    Csub[(ty + BLOCK_SIZE) * C.stride + tx + BLOCK_SIZE]   = c_11;
}

