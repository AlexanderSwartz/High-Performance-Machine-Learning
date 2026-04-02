#include <stdio.h>
#include "timer.h"

// Max kernel entries (channels * FH * FW) for FH=3,FW=3,channels=3
// Must be a compile-time constant to size shared memory array.
#define KERNEL_ENTRIES 27
#define CHANNELS 3
#define BDIMX 16
#define BDIMY 16
#define PADDING 1
#define TILE_W (BDIMX + 2 * PADDING) // 18
#define TILE_H (BDIMY + 2 * PADDING) // 18

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

// Device kernel: compute valid 2D convolution for a filter bank
// A is packed as [ch][H][W] with Matrix.width=W and Matrix.height=H*channels.
// K is packed as [k][ch][FH][FW] with Matrix.width=FW and Matrix.height=FH*channels*kilters.
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
    int j = rem / FW;
    int i = rem % FW;
    int k_base_row = k * (FH * CHANNELS) + ch * FH;
    sK[tid] = K.elements[(k_base_row + j) * K.width + i];
  }

  // Each block computes a 16x16 tile of output, but needs an 18x18 tile of input due to the 3x3 kernel and padding.
  int tile_size = TILE_W * TILE_H;
   // Each thread loads one element of the 18x18 tile into shared memory, then jumps by the total number of threads in the block to load the next element until all 324 elements are loaded.
  int stride = blockDim.x * blockDim.y;
  // 18 width * 18 height * 3 channels = 972 elements
  __shared__ double sA[TILE_W * TILE_H * CHANNELS];

  int source_offset_x = blockIdx.x * blockDim.x;
  int source_offset_y = blockIdx.y * blockDim.y;

  int total_tile_elements = tile_size * CHANNELS; // 972
  // index i can't map cleanly to channel because TILE_H (18) > blockDim.y (16)
  for (int i = tid; i < total_tile_elements; i += stride) {
    int ch = i / (tile_size);            
    int rem = i % (tile_size);           
    // positions within tile
    int tile_y = rem / TILE_W;
    int tile_x = rem % TILE_W;

    // convert tile coords to source padded A coords
    int source_x = source_offset_x + tile_x;
    int source_y = source_offset_y + tile_y;

    // flatten same in c1, since A is (ch, H_p, W_p)
    int source_idx = (ch * H_p + source_y) * A.width + source_x;

    // optonal debug prints
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //   printf("thread(%d,%d,%d) block(%d,%d,%d) loop i=%d reads from A.elements[%d] -> writes to sA[%d] (ch=%d tile=(%d,%d)\n",
    //         threadIdx.x, threadIdx.y, blockIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, i, source_idx, i, ch, tile_y, tile_x);
    // }

    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 1 && threadIdx.y == 0) {
    //   printf("thread(%d,%d,%d) block(%d,%d,%d) loop i=%d reads from A.elements[%d] -> writes to sA[%d] (ch=%d tile=(%d,%d)\n",
    //         threadIdx.x, threadIdx.y, blockIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, i, source_idx, i, ch, tile_y, tile_x);
    // }

  // loop 1
  // thread(0,0,0) block(0,0,0) loop i=0 reads from A.elements[0] -> writes to sA[0] (ch=0 tile=(0,0)
  // thread(1,0,0) block(0,0,0) loop i=1 reads from A.elements[1] -> writes to sA[1] (ch=0 tile=(0,1)
  // loop 2
  // thread(0,0,0) block(0,0,0) loop i=256 reads from A.elements[14368] -> writes to sA[256] (ch=0 tile=(14,4)
  // thread(1,0,0) block(0,0,0) loop i=257 reads from A.elements[14369] -> writes to sA[257] (ch=0 tile=(14,5)
  // loop 3
  // thread(0,0,0) block(0,0,0) loop i=512 reads from A.elements[1062944] -> writes to sA[512] (ch=1 tile=(10,8)
  // thread(1,0,0) block(0,0,0) loop i=513 reads from A.elements[1062945] -> writes to sA[513] (ch=1 tile=(10,9)

    if (source_y < H_p && source_x < W_p) {
      sA[i] = A.elements[source_idx];
    } else {
      sA[i] = 0.0;
    }
  }
  
  __syncthreads();

  if (x >= C.width || y >= H) return;

  double acc = 0.0;
  for (int ch = 0; ch < CHANNELS; ++ch) {
    for (int j = 0; j < FH; ++j) {
      for (int i = 0; i < FW; ++i) {
        // Uses I0[c, x + i, y + j], except j and i are in proper order
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

int main(int argc, char** argv) {
  // A dimensions (before padding): C x H x W
  // A dimensions (after padding): C x H_p x W_p
  // K dimensions: K x C x H x W
  // C dimensions: K x H x W
  const int K = 64;
  const int H = 1024, W = 1024;
  const int FH = 3, FW = 3;
  const int P = 1;
  const int W_p = W + 2 * P;
  const int H_p = H + 2 * P;

  DMatrix host_A = MakeHostMatrixD(W_p, H_p * CHANNELS);
  DMatrix host_K = MakeHostMatrixD(FW, FH * CHANNELS * K);
  DMatrix host_C = MakeHostMatrixD(W, H * K);

  size_t sizeA = (size_t)host_A.width * host_A.height;
  for (size_t i = 0; i < sizeA; ++i) host_A.elements[i] = 0.0;

  // Fill I[c,x,y] = c * (x + y), using offset of P for padding
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
  for (int k = 0; k < K; ++k) {
    for (int ch = 0; ch < CHANNELS; ++ch) {
      int kbase = k * (FH * CHANNELS) + ch * FH;
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
        for (int ch = 0; ch < CHANNELS; ++ch) {
          int abase = ch * H_p;
          int kbase = k * (FH * CHANNELS) + ch * FH;
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
  ConvKernel<<<numBlocks, threadsPerBlock>>>(device_A, device_K, device_C, H_p, W_p, FH, FW, K);
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
