///
/// matmult.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-19 DVN
///
/// Do not modify this file. The GTA will grade your
/// code using the master copy of this file, not your
/// copy, so any modifications you make will not play
/// a role in the grading.
///

// Includes
#include <stdio.h>
#include "timer.h"
#include "matmultKernel.h"

// Defines
#define epsilon (float)1e-4
#define verbose false
// Max kernel entries (channels * FH * FW) for FH=3,FW=3,channels=3
// Must be a compile-time constant to size shared memory array.
#define KERNEL_ENTRIES 27
#define CHANNELS 3
#define BDIMX 16
#define BDIMY 16
#define PADDING 1
#define TILE_W (BDIMX + 2 * PADDING) // 18
#define TILE_H (BDIMY + 2 * PADDING) // 18

Matrix MakeDeviceMatrix(Matrix M, bool copy){
  // Create a new matrix in device memory.
  Matrix newDeviceMatrix;
  newDeviceMatrix.width = M.width;
  newDeviceMatrix.stride = M.width;
  newDeviceMatrix.height = M.height;
  size_t size = M.width * M.height * sizeof(float);
  cudaMalloc((void**) &newDeviceMatrix.elements, size);
  if (copy)
    cudaMemcpy(newDeviceMatrix.elements, M.elements, size, cudaMemcpyHostToDevice);
  return newDeviceMatrix;
}

// Create a matrix in host memory.
Matrix MakeHostMatrix(int width, int height){
  Matrix newHostMatrix;
  newHostMatrix.width = width;
  newHostMatrix.height = height;
  size_t size = newHostMatrix.width * newHostMatrix.height * sizeof(float);
  newHostMatrix.elements = (float*)malloc(size);
  return newHostMatrix;
}

// Double-precision matrix descriptor for this convolution test only.
typedef struct {
  int width;
  int height;
  double* elements;
} DMatrix;

// Create a double matrix in host memory.
DMatrix MakeHostMatrixD(int width, int height){
  DMatrix M;
  M.width = width;
  M.height = height;
  size_t size = (size_t)M.width * M.height * sizeof(double);
  M.elements = (double*)malloc(size);
  return M;
}

// Create a double matrix in device memory.
DMatrix MakeDeviceMatrixD(DMatrix M, bool copy){
  DMatrix newDeviceMatrix;
  newDeviceMatrix.width = M.width;
  newDeviceMatrix.height = M.height;
  size_t size = (size_t)M.width * M.height * sizeof(double);
  cudaMalloc((void**) &newDeviceMatrix.elements, size);
  if (copy)
    cudaMemcpy(newDeviceMatrix.elements, M.elements, size, cudaMemcpyHostToDevice);
  return newDeviceMatrix;
}

// Print a double matrix stored in host memory.
void printMatrixD(DMatrix M, const char* name) {
  printf("\n%s \n",name);
  for(int y=0; y<M.height; y++){
    for(int x=0; x<M.width; x++) {
      printf("%f ", M.elements[y * M.width + x]);
    }
    printf("\n");
  }
}

// Print a 3D tensor stored as packed DMatrix with layout [K][H][W].
// If maxRows/maxCols > 0, only print that many rows/cols per slice to avoid huge dumps.
void print3DDlim(DMatrix M, int Kfilters, int H, int W, int maxRows, int maxCols) {
  for (int k = 0; k < Kfilters; ++k) {
    printf("\nSlice k=%d\n", k);
    double* slicePtr = &M.elements[(size_t)k * H * W];
    int rows = (maxRows > 0 && maxRows < H) ? maxRows : H;
    int cols = (maxCols > 0 && maxCols < W) ? maxCols : W;
    for (int y = 0; y < rows; ++y) {
      for (int x = 0; x < cols; ++x) {
        printf("%f ", slicePtr[y * W + x]);
      }
      if (cols < W) printf(" ...");
      printf("\n");
    }
    if (rows < H) printf("... (only first %d of %d rows shown)\n", rows, H);
  }
}

// Print a matrix stored in host memory.
void printMatrix(Matrix M, const char* name) {
  printf("\n%s \n",name);
  for(int y=0; y<M.height; y++){
   for(int x=0; x<M.width; x++) {
      printf("%f ", M.elements[y * M.width + x]);
   }
   printf("\n");
  }
}

// Initialize dummy data in a matrix stored in host memory.
void initMatrix(Matrix M, bool horizontal) {
  for(int y=0; y<M.height; y++) {
    for(int x=0; x<M.width; x++) {
      M.elements[y*M.width+x] = (float)(horizontal?x:y);
    }
  }
}

// Check the specified matrix to be sure it is correct.
// That is, make sure it is the result of multiplying the
// dummy data we created earlier.
void checkResult(Matrix M) {

  Matrix correct = MakeHostMatrix(M.width, M.height);

  for(int y=0; y<M.height; y++) {
    for(int x=0; x<M.width; x++) {
       correct.elements[y*correct.width+x] = (float)M.width*(float)x*y;
    }
  }

  if(verbose){
   // print correct
   printMatrix(correct, "correct");

   // print host_C
   printMatrix(M, "result");
  }


  double maxerror = 0.0;
  int errCnt = 0;
  for(int y=0; y<correct.height; y++) {
    for(int x=0; x<correct.width; x++) {
      float it = correct.elements[y*correct.width+x];
      if(fabs(it - M.elements[y*M.width+x])> epsilon*it) {
        errCnt++;
        double error = fabs(it - M.elements[y*M.width+x])/it;
        if (error > maxerror) maxerror = error;
      }      
    }
  }

  if(errCnt>0){
    printf("\n\nTEST FAILED: number of errors:  %d, max rel error: %f\n", errCnt, maxerror);
  }
  
  free(correct.elements);
}

// Device kernel: compute valid 2D convolution for a filter bank
// A is packed as [ch][H][W] with Matrix.width=W and Matrix.height=H*channels.
// K is packed as [k][ch][FH][FW] with Matrix.width=FW and Matrix.height=FH*channels*Kfilters.
// C is packed as [k][H][W] with Matrix.width=W and Matrix.height=H*k.
__global__ void ConvKernel
(DMatrix A, DMatrix K, DMatrix C,
                           int H_p, int W_p, int FH, int FW, int totalFilters) {
  // compute global output coordinates

  // output coordinates (k,x,y) that this thread computes
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z;

  
  int H = C.height / totalFilters;

  // Load this filter's kernel into shared memory (per-block)
  __shared__ double sK[KERNEL_ENTRIES];
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  // actual kernel size is fixed at compile time (KERNEL_ENTRIES)
  if (tid < KERNEL_ENTRIES) {
    int ch = tid / (FH * FW);
    int rem = tid % (FH * FW);
    int ky = rem / FW;
    int kx = rem % FW;
    int k_base_row = k * (FH * CHANNELS) + ch * FH;
    sK[tid] = K.elements[(k_base_row + ky) * K.width + kx];
  }

  // Each block computes a 16x16 tile of output, but needs an 18x18 tile of input due to the 3x3 kernel and padding.
  int tile_size = TILE_W * TILE_H;
   // Each thread loads one element of the 18x18 tile into shared memory, then jumps by the total number of threads in the block to load the next element until all 324 elements are loaded.
  int stride = blockDim.x * blockDim.y;

  // 18 width * 18 height * 3 channels = 972 elements
  __shared__ double sA[TILE_W * TILE_H * CHANNELS];

  // thread 0,0 block 0,0,0 loads sA[0], sA[256] (for channel 1)
  // then loads sA[324], sA[580] (for channel 2)
  // then loads sA[648], sA[904] (for channel 3)

  // thread 15,15 block 0,0,0 loads sA[255] (for channel 1), sA[579] (for channel 2), sA[903] (for channel 3)
  // then thread 0,0,1 loads sA[1], sA[325], sA[649] (second pixel of each channel), etc.

  // block 0,0,0 loads same elements as block 0,0,1

  for (int ch = 0; ch < CHANNELS; ++ch) {
      for (int i = tid; i < tile_size; i += stride) {
          // Convert 1D loop index 'i' to 2D tile coordinates (0-17)
          int local_y = i / TILE_W;
          int local_x = i % TILE_W;

          // Find the top-left corner of the input area this block needs.
          // blockIdx.x * 16 gives the starting pixel in the output,
          // which corresponds to the same coordinate in the padded input A
          // because the padding (P=1) offsets the actual data.
          int global_y = blockIdx.y * BDIMY + local_y;
          int global_x = blockIdx.x * BDIMX + local_x;
            int sA_index = ch * tile_size + local_y * TILE_W + local_x;
            // helpful prints for debugging
            // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 1 && threadIdx.x == 0 && threadIdx.y == 0) {
            //   printf("thread(%d,%d,%d) block(%d,%d,%d) loading sA idx=%d ch=%d i=%d local=(%d,%d) global=(%d,%d)\n",
            //      threadIdx.x, threadIdx.y, blockIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, sA_index, ch, i, local_y, local_x, global_y, global_x);
            // }
            if (global_y < H_p && global_x < W_p) {
              sA[sA_index] = A.elements[(ch * H_p + global_y) * A.width + global_x];
            } else {
              sA[sA_index] = 0.0;
            }
      }
  }

  __syncthreads();

  if (x >= C.width || y >= H) return;

  double acc = 0.0;
  for (int ch = 0; ch < CHANNELS; ++ch) {
    int a_base_row = ch * H_p;
    for (int j = 0; j < FH; ++j) {
      for (int i = 0; i < FW; ++i) {
        // Uses I0[c, x + i, y + j], except j and i are in proper order
        // a_base_row includes offset from ch
        double a = sA[ch * (TILE_W * TILE_H) + (threadIdx.y + j) * TILE_W + (threadIdx.x + i)];
        double b = sK[ch * (FH * FW) + (FH - 1 - j) * FW + (FW - 1 - i)];
        acc += a * b;
      }
    }
  }

  // 2D Cmatrix is (w, H*k)
  // Convert to 1D by multiplying row (k * H + y) by width (C.width) and adding x
  C.elements[(k * H + y) * C.width + x] = acc;
}

//
// main
//
int main(int argc, char** argv) {

  // For convolution testing with channels and multiple filters.
  // Dimensions: k x channels x H x W
  // channels is a compile-time constant
  const int K = 64; // number of distinct filters
  const int H = 1024, W = 1024;
  const int FH = 3, FW = 3;
  const int P = 1; // padding
  const int W_p = W + 2 * P;
  const int H_p = H + 2 * P;

  // Use double-precision packed tensors for convolution
  DMatrix host_A = MakeHostMatrixD(W_p, H_p * CHANNELS);
  // Pack kernels: width=FW, height=FH*channels*k
  DMatrix host_K = MakeHostMatrixD(FW, FH * CHANNELS * K);
  // Outputs packed as height = H * k
  DMatrix host_C = MakeHostMatrixD(W, H * K);

  // Zero the packed input A (padded buffer)
  size_t sizeA = (size_t)host_A.width * host_A.height;
  for (size_t i = 0; i < sizeA; ++i) host_A.elements[i] = 0.0;

  // Fill I[c,x,y] = c * (x + y) for original image and place into padded I0 at offset P
  for (int ch = 0; ch < CHANNELS; ++ch) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        double v = (double)(ch * (x + y));
        int row = ch * H_p + (y + P);
        int col = x + P;
        host_A.elements[row * host_A.width + col] = v;
      }
    }
  }

  // Fill kernel K per filter and per channel using F[k,c,i,j] = (c + k) * (i + j)
  for (int kf = 0; kf < K; ++kf) {
    for (int ch = 0; ch < CHANNELS; ++ch) {
      int kbase = kf * (FH * CHANNELS) + ch * FH;
      for (int ky = 0; ky < FH; ++ky) {
        for (int kx = 0; kx < FW; ++kx) {
          host_K.elements[(kbase + ky) * host_K.width + kx] = (double)((ch + kf) * (ky + kx));
        }
      }
    }
  }


  if (verbose == 2) {
    // Print each channel of A and K separately (show padded A channel views)
    for (int ch = 0; ch < CHANNELS; ++ch) {
      DMatrix sliceA = host_A;
      sliceA.height = H_p;
      sliceA.elements = &host_A.elements[ch * H_p * host_A.width];
      char nameA[64];
      sprintf(nameA, "host_A ch %d (%dx%d) padded", ch, H_p, W_p);
      printMatrixD(sliceA, nameA);
    }
    for (int kf = 0; kf < K; ++kf) {
      for (int ch = 0; ch < CHANNELS; ++ch) {
        DMatrix sliceK = host_K;
        sliceK.height = FH;
        sliceK.elements = &host_K.elements[(kf * (FH * CHANNELS) + ch * FH) * host_K.width];
        char nameK[64];
        sprintf(nameK, "host_K filt %d ch %d (%dx%d)", kf, ch, FH, FW);
        printMatrixD(sliceK, nameK);
      }
    }
  }

  // Compute 2D convolution on the host with channel summation for each filter
  for (int kf = 0; kf < K; ++kf) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        double acc = 0.0;
        for (int ch = 0; ch < CHANNELS; ++ch) {
          int abase = ch * H_p;
          int kbase = kf * (FH * CHANNELS) + ch * FH;
          for (int ky = 0; ky < FH; ++ky) {
            for (int kx = 0; kx < FW; ++kx) {
              double a = host_A.elements[(abase + y + ky) * host_A.width + (x + kx)];
              double b = host_K.elements[(kbase + (FH - 1 - ky)) * host_K.width + (FW - 1 - kx)];
              acc += a * b;
            }
          }
        }
        host_C.elements[(kf * H + y) * host_C.width + x] = acc;
      }
    }
  }

  if (verbose == 2) {
    printMatrixD(host_C, "host_C (convolution result)");
  }

  // --- Run convolution on the GPU: grid.z = number of filters, each block computes a 16x16 tile ---
  DMatrix device_A = MakeDeviceMatrixD(host_A, true);
  DMatrix device_K = MakeDeviceMatrixD(host_K, true);
  DMatrix device_C = MakeDeviceMatrixD(host_C, false);

  // Launch with 16x16 threads per block and grid.z = number of filters
  dim3 threadsPerBlock(16, 16, 1);
  dim3 numBlocks((W + 15) / 16, (H + 15) / 16, K);
  printf("Launching ConvKernel with grid (%d,%d,%d) and block (%d,%d,%d)\n",
         numBlocks.x, numBlocks.y, numBlocks.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
  // Launch the device kernel (ConvKernel defined in this file)
    // Time the kernel: start timer, launch, synchronize, stop timer
    initialize_timer();
    start_timer();
    ConvKernel<<<numBlocks, threadsPerBlock>>>(device_A, device_K, device_C, H_p, W_p, FH, FW, K);
    cudaDeviceSynchronize();
    stop_timer();
    double kernel_time = elapsed_time();
    printf("ConvKernel time: %lf (sec)\n", kernel_time);

  // Copy result back to host and print/verify (double precision)
  DMatrix host_C_gpu = MakeHostMatrixD(host_C.width, host_C.height);
  size_t sizeC = (size_t)host_C.width * host_C.height * sizeof(double);
  cudaMemcpy(host_C_gpu.elements, device_C.elements, sizeC, cudaMemcpyDeviceToHost);

  // if (verbose == 1) {
  //   // Print first 8x8 of each filter slice for debugging to avoid flooding the console.
  //   print3DDlim(host_C_gpu, K, H, W, 64, 64);
  // }

  // Simple verification against host result
  int mismatches = 0;
  for (size_t i = 0; i < (size_t)host_C.width * host_C.height; ++i) {
    double h = host_C.elements[i];
    double g = host_C_gpu.elements[i];
    if (fabs(h - g) > 1e-6) {
      ++mismatches;
    }
  }
  if (mismatches == 0) {
    printf("GPU result matches host result\n");
  } else {
    printf("GPU result mismatches: %d\n", mismatches);
  }

  // Compute checksums (sum of all elements) on host and GPU copy and compare
  double sum_host = 0.0;
  double sum_gpu = 0.0;
  size_t total = (size_t)host_C.width * host_C.height;
  for (size_t i = 0; i < total; ++i) {
    sum_host += host_C.elements[i];
    sum_gpu += host_C_gpu.elements[i];
  }
  double diff = fabs(sum_host - sum_gpu);
  printf("Checksum host: %.12f, gpu: %.12f, diff: %.12f\n", sum_host, sum_gpu, diff);
  double tol = 1e-6 * fabs(sum_host) + 1e-12;
  if (diff <= tol) {
    printf("Checksum OK (within tolerance)\n");
  } else {
    printf("Checksum FAILED (diff > tol)\n");
  }
  // Free device memory
  cudaFree(device_A.elements);
  cudaFree(device_K.elements);
  cudaFree(device_C.elements);
  free(host_C_gpu.elements);

  // Free allocated memory.
  free(host_A.elements);
  free(host_K.elements);
  free(host_C.elements);

  return 0;
}

