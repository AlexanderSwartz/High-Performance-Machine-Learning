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

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    float *Asub, *Bsub, *Csub;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;

    // Csub points to the top-left of the 32x32 output block
    Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_col + FOOTPRINT_SIZE * block_row];

    // Cvalue00: top-left, Cvalue01: top-right, Cvalue10: bottom-left, Cvalue11: bottom-right
    float Cvalue00 = 0.0f;
    float Cvalue01 = 0.0f;
    float Cvalue10 = 0.0f;
    float Cvalue11 = 0.0f;

    // Loop over all sub-matrices of A and B
    for (int m = 0; m < (A.width / FOOTPRINT_SIZE); ++m) {
        
        // Asub and Bsub point to the top-left of the current 32x32 input tiles
        Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_col + FOOTPRINT_SIZE * m];
        Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_row];

        __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
        __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

        // Each 16x16 thread block loads 1 element for each of the 4 quadrants (top-left, top-right, bottom-left, bottom-right)
        shared_A[thread_row][thread_col]           = Asub[thread_row * A.stride + thread_col];
        shared_A[thread_row][thread_col + BLOCK_SIZE]      = Asub[thread_row * A.stride + thread_col + BLOCK_SIZE];
        shared_A[thread_row + BLOCK_SIZE][thread_col]      = Asub[(thread_row + BLOCK_SIZE) * A.stride + thread_col];
        shared_A[thread_row + BLOCK_SIZE][thread_col + BLOCK_SIZE] = Asub[(thread_row + BLOCK_SIZE) * A.stride + thread_col + BLOCK_SIZE];

        shared_B[thread_row][thread_col]           = Bsub[thread_row * B.stride + thread_col];
        shared_B[thread_row][thread_col + BLOCK_SIZE]      = Bsub[thread_row * B.stride + thread_col + BLOCK_SIZE];
        shared_B[thread_row + BLOCK_SIZE][thread_col]      = Bsub[(thread_row + BLOCK_SIZE) * B.stride + thread_col];
        shared_B[thread_row + BLOCK_SIZE][thread_col + BLOCK_SIZE] = Bsub[(thread_row + BLOCK_SIZE) * B.stride + thread_col + BLOCK_SIZE];
        
        __syncthreads();

        #pragma unroll
        for (int e = 0; e < FOOTPRINT_SIZE; ++e) {
            float a_top    = shared_A[thread_row][e];
            float a_bottom = shared_A[thread_row + BLOCK_SIZE][e];
            float b_left   = shared_B[e][thread_col];
            float b_right  = shared_B[e][thread_col + BLOCK_SIZE];

            Cvalue00 += a_top * b_left;
            Cvalue01 += a_top * b_right;
            Cvalue10 += a_bottom * b_left;
            Cvalue11 += a_bottom * b_right;
        }

        __syncthreads();
    }

    // Debug: print which global C elements this thread (0,0) will write
    // if (thread_row == 0 && thread_col == 0 && block_col == 0 && block_row == 0) {
    //     int base_row = block_col * FOOTPRINT_SIZE;
    //     int base_col = block_row * FOOTPRINT_SIZE;
    //     printf("Block (%d,%d) thread (0,0) writing: C[%d,%d]=Cvalue00, C[%d,%d]=Cvalue01, C[%d,%d]=Cvalue10, C[%d,%d]=Cvalue11\n",
    //            block_col, block_row,
    //            base_row, base_col,
    //            base_row, base_col + BLOCK_SIZE,
    //            base_row + BLOCK_SIZE, base_col,
    //            base_row + BLOCK_SIZE, base_col + BLOCK_SIZE);
    // }

    Csub[thread_row * C.stride + thread_col]                               = Cvalue00;
    Csub[thread_row * C.stride + thread_col + BLOCK_SIZE]                  = Cvalue01;
    Csub[(thread_row + BLOCK_SIZE) * C.stride + thread_col]                = Cvalue10;
    Csub[(thread_row + BLOCK_SIZE) * C.stride + thread_col + BLOCK_SIZE]   = Cvalue11;
}

