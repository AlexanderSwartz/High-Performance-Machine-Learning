#include <stdio.h>
#include "timer.h"

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

// Print a sub-region (rows x cols) of a packed DMatrix
void printSubMatrixD(DMatrix src, int startRow, int startCol, int rows, int cols) {
  if (rows <= 0 || cols <= 0) return;
  DMatrix small = MakeHostMatrixD(cols, rows);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      small.elements[r * cols + c] = src.elements[(startRow + r) * src.width + (startCol + c)];
    }
  }
  // printSubMatrixD no longer prints a label; caller should print a header if desired
  printMatrixD(small, "");
  free(small.elements);
}

// Device kernel: compute valid 2D convolution for a filter bank
// A is packed as [ch][H][W] with Matrix.width=W and Matrix.height=H*channels.
// K is packed as [k][ch][FH][FW] with Matrix.width=FW and Matrix.height=KH*channels*filters.
// C is packed as [k][H][W] with Matrix.width=W and Matrix.height=H*k.
__global__ void ConvKernel(DMatrix A, DMatrix K, DMatrix C,
                           int channels, int H_p, int KH, int FW, int totalFilters) {
  // output coordinates (k,x,y) that this thread computes
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z;
  
  int H = C.height / totalFilters;
  if (x >= C.width || y >= H) return;

  double acc = 0.0;
  for (int ch = 0; ch < channels; ++ch) {
    int a_base_row = ch * H_p;
    int k_base_row = k * (KH * channels) + ch * KH;
    for (int j = 0; j < KH; ++j) {
      for (int i = 0; i < FW; ++i) {
        // Uses I0[c, x + i, y + j], except j and i are in proper order
        // a_base_row includes offset from ch
        double a = A.elements[(a_base_row + y + j) * A.width + (x + i)];
        double b = K.elements[(k_base_row + (KH - 1 - j)) * K.width + (FW - 1 - i)];
        acc += a * b;
      }
    }
  }

  // 2D Cmatrix is (w, H*k)
  // Convert to 1D by multiplying row (k * H + y) by width (C.width) and adding x
  C.elements[(k * H + y) * C.width + x] = acc;
}

int main(int argc, char** argv) {
  // A dimensions (before padding): C x H x W
  // A dimensions (after padding): C x H_p x W_p
  // K dimensions: K x C x H x W
  // C dimensions: K x H x W
  const int channels = 3;
  const int K = 64;
  const int H = 1024, W = 1024;
  const int FH = 3, FW = 3;
  const int P = 1;
  const int W_p = W + 2 * P;
  const int H_p = H + 2 * P;

  DMatrix host_A = MakeHostMatrixD(W_p, H_p * channels);
  DMatrix host_K = MakeHostMatrixD(FW, FH * channels * K);
  DMatrix host_C = MakeHostMatrixD(W, H * K);

  size_t sizeA = (size_t)host_A.width * host_A.height;
  for (size_t i = 0; i < sizeA; ++i) host_A.elements[i] = 0.0;

  // Fill I[c,x,y] = c * (x + y), using offset of P for padding
  for (int ch = 0; ch < channels; ++ch) {
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
  for (int k = 0; k < K; ++k) {
    for (int ch = 0; ch < channels; ++ch) {
      int kbase = k * (FH * channels) + ch * FH;
      for (int j = 0; j < FH; ++j) {
        for (int i = 0; i < FW; ++i) {
          host_K.elements[(kbase + j) * host_K.width + i] = (double)((ch + k) * (j + i));
        }
      }
    }
  }

  // Compute 2D convolution on the host with channel summation for each filter
  for (int k = 0; k < K; ++k) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        double acc = 0.0;
        for (int ch = 0; ch < channels; ++ch) {
          int abase = ch * H_p;
          int kbase = k * (FH * channels) + ch * FH;
          for (int j = 0; j < FH; ++j) {
            for (int i = 0; i < FW; ++i) {
              double a = host_A.elements[(abase + y + j) * host_A.width + (x + i)];
              double b = host_K.elements[(kbase + (FH - 1 - j)) * host_K.width + (FW - 1 - i)];
              acc += a * b;
            }
          }
        }
        host_C.elements[(k * H + y) * host_C.width + x] = acc;
      }
    }
  }

  DMatrix device_A = MakeDeviceMatrixD(host_A, true);
  DMatrix device_K = MakeDeviceMatrixD(host_K, true);
  DMatrix device_C = MakeDeviceMatrixD(host_C, false);

  // Each block computes 16x16 piece of output
  dim3 threadsPerBlock(16, 16, 1);
  // Grid size is (W/16, H/16, K) rounded up
  dim3 numBlocks((W + 15) / 16, (H + 15) / 16, K);
  printf("Launching ConvKernel with grid (%d,%d,%d) and block (%d,%d,%d)\n",
         numBlocks.x, numBlocks.y, numBlocks.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

  initialize_timer();
  start_timer();
  ConvKernel<<<numBlocks, threadsPerBlock>>>(device_A, device_K, device_C, channels, H_p, FH, FW, K);
  cudaDeviceSynchronize();
  stop_timer();
  double kernel_time = elapsed_time();
  double kernel_ms = kernel_time * 1000.0;
  printf("ConvKernel time: %.3f ms\n", kernel_ms);

  // instead of overwriting host_C, copy to a new matrix (host_C_gpu) so we can compare to host_C results
  DMatrix host_C_gpu = MakeHostMatrixD(host_C.width, host_C.height);
  size_t sizeC = (size_t)host_C.width * host_C.height * sizeof(double);
  cudaMemcpy(host_C_gpu.elements, device_C.elements, sizeC, cudaMemcpyDeviceToHost);

  // Prints for debugging
  // printf("\nFirst channel 8x8 of host_A");
  // printSubMatrixD(host_A, 0, 0, 8, 8);
  // printf("\nSecond channel 8x8 of host_A");
  // printSubMatrixD(host_A, H_p, 0, 8, 8);

  // printf("\nFirst filter, first channel 8x8 of host_K");
  // printSubMatrixD(host_K, 0, 0, 3, 3);
  // printf("\nFirst filter, second channel 8x8 of host_K");
  // printSubMatrixD(host_K, FH, 0, 3, 3);
  // printf("\nSecond filter, first channel 8x8 of host_K");
  // printSubMatrixD(host_K, FH * channels, 0, 3, 3);

  // printf("\nFirst filter 8x8 of host_C_gpu");
  // printSubMatrixD(host_C_gpu, 0, 0, 8, 8);
  // printf("\nSecond filter 8x8 of host_C_gpu");
  // printSubMatrixD(host_C_gpu, H, 0, 8, 8);

  // Compute checksums (sum of all elements) on host and GPU and compare
  double sum_host = 0.0;
  double sum_gpu = 0.0;
  size_t total = (size_t)host_C.width * host_C.height;
  for (size_t i = 0; i < total; ++i) {
    sum_host += host_C.elements[i];
    sum_gpu += host_C_gpu.elements[i];
  }
  printf("Checksum computed by GPU: %.12f\n", sum_gpu);

  double diff = fabs(sum_host - sum_gpu);
  double tol = 1e-12;
  if (diff <= tol) {
    printf("Checksum OK (within tolerance)\n");
  } else {
    printf("Checksum FAILED (diff > tol)\n");
    printf("Checksum host: %.12f, gpu: %.12f, diff: %.12f\n", sum_host, sum_gpu, diff);
  }

  cudaFree(device_A.elements);
  cudaFree(device_K.elements);
  cudaFree(device_C.elements);
  free(host_C_gpu.elements);

  free(host_A.elements);
  free(host_K.elements);
  free(host_C.elements);

  return 0;
}
