#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include "timer.h"
#include <cudnn.h>

// Reuse error checking from notes
#define CUDNN_CALL(x) do { \
    cudnnStatus_t ___s = (x); \
    if (___s != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "%s:%d ERROR: %s\n", __FILE__, \
                __LINE__, cudnnGetErrorString(___s)); \
        exit(-1); \
    } \
} while (0)

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

// This function follows steps outlined by TA in # 226 on Ed
// As recommended by TA, uses cudnnFindConvolutionForwardAlgorithm since CUDNN CONVOLUTION FWD PREFER FASTEST
// is not supported by the image Deep Learning VM with CUDA 11.3, M126 
// This function also adapts code from lecture notes
void run_cudnn_convolution(double* d_A, double* d_K, double* d_C,
                           int channels, int H_p, int W_p, 
                           int FH, int FW, int totalFilters) {
    
    // Step 0: Create Handle
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    // Step 1: Set Descriptors
    // Input Descriptor (A)
    cudnnTensorDescriptor_t A_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&A_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(A_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, channels, H_p, W_p));

    cudnnFilterDescriptor_t K_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&K_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(K_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, totalFilters, channels, FH, FW));
        
    // set padding to 0 since we already padded the input
    int pad_h = 0; 
    int pad_w = 0;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

    // Check that cudNN's calculated output dimensions match expected dimensions
    int N, C, H, W;
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(conv_desc, A_desc, K_desc, &N, &C, &H, &W));

    // Compute expected output dimensions using the same conv params
    int expected_N = 1;
    int expected_C = totalFilters;
    int expected_H = H_p - FH + 1;
    int expected_W = W_p - FW + 1;

    if (N != expected_N || C != expected_C || H != expected_H || W != expected_W) {
      fprintf(stderr, "cuDNN output dims mismatch: cudnn=(%d,%d,%d,%d) expected=(%d,%d,%d,%d)\n",
          N, C, H, W,
          expected_N, expected_C, expected_H, expected_W);
      exit(-1);
    }
    printf("cuDNN output dimensions: N=%d, C=%d, H=%d, W=%d\n", N, C, H, W);

    cudnnTensorDescriptor_t out_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W));

    // Step 2: Find Fastest Algorithm
    // This runs a microbenchmark on different convolution algorithms and picks the fastest for this config
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    int returnedAlgoCount = 0;

    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
      cudnn, A_desc, K_desc, conv_desc, out_desc,
      1, &returnedAlgoCount, &perfResults
    ));
    
    // to see which algorithm this enum maps to, look here: 
    // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html#id101:~:text=CUDNN_DATA_INT8x32-,Supported%20Algorithms,-For%20this%20function
    printf("cuDNN chose algorithm: %d\n", perfResults.algo);

    // Step 3: Query Workspace Size
    size_t workspaceSizeInBytes = 0;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, A_desc, K_desc, conv_desc, out_desc, 
        perfResults.algo, &workspaceSizeInBytes
    ));

    // Step 4: Allocate Workspace
    void* d_workspace = nullptr;
    if (workspaceSizeInBytes > 0) {
        cudaMalloc(&d_workspace, workspaceSizeInBytes);
    }

    // Step 5: Run Convolution
    // Output = alpha * Convolution + beta * Output
    // Keep default: alpha=1 keeps output the same, beta=0 clears existing output before writing new values
    double alpha = 1.0;
    double beta = 0.0;

    initialize_timer();
    start_timer();

    CUDNN_CALL(cudnnConvolutionForward(
        cudnn, 
        &alpha, A_desc, d_A, 
        K_desc, d_K, 
        conv_desc, perfResults.algo, 
        d_workspace, workspaceSizeInBytes, 
        &beta, out_desc, d_C
    ));

    cudaDeviceSynchronize(); 
    stop_timer();
    double cudnn_time = elapsed_time();
    double cudnn_ms = cudnn_time * 1000.0;
    printf("Total time of cudnnConvolutionForward: %.3f ms\n", cudnn_ms);

    if (d_workspace != nullptr) cudaFree(d_workspace);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(A_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(K_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroy(cudnn));
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

  run_cudnn_convolution(device_A.elements, device_K.elements, device_C.elements,
                        channels, H_p, W_p, FH, FW, K);

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
  cudaFree(device_C.elements);
  free(host_C_gpu.elements);

  free(host_A.elements);
  free(host_K.elements);
  free(host_C.elements);

  return 0;
}
