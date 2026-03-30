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
#include <cmath>
#include <cstdlib>
#include "timer.h"
#include "matmultKernel.h"

#include <cudnn.h>
#include <iostream>

// Defines
#define EPSILON (float)1e-4
#define verbose false

// Reuse error checking from notes
#define CUDNN_CALL(x) do { \
    cudnnStatus_t ___s = (x); \
    if (___s != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "%s:%d ERROR: %s\n", __FILE__, \
                __LINE__, cudnnGetErrorString(___s)); \
        exit(-1); \
    } \
} while (0)

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
void print3DDlim(DMatrix M, int kilters, int H, int W, int maxRows, int maxCols) {
  for (int k = 0; k < kilters; ++k) {
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
      if(fabs(it - M.elements[y*M.width+x])> EPSILON*it) {
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

// This function follows steps outlined by TA in # 226 on Ed
// and borrows functions calls from lecture notes
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
    printf("Total time of cudnnConvolutionForward: %lf (sec)\n", cudnn_time);

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
  const int K = 64; // number of distinct filters
  const int H = 1024, W = 1024;
  const int FH = 3, FW = 3;
  const int P = 1; // padding
  const int W_p = W + 2 * P;
  const int H_p = H + 2 * P;

  // Use double-precision packed tensors for convolution
  DMatrix host_A = MakeHostMatrixD(W_p, H_p * channels);
  // Pack kernels: width=FW, height=FH*channels*k
  DMatrix host_K = MakeHostMatrixD(FW, FH * channels * K);
  // Outputs packed as height = H * k
  DMatrix host_C = MakeHostMatrixD(W, H * K);

  // Zero the packed input A (padded buffer)
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


  if (verbose == 2) {
    // Print each channel of A and K separately (show padded A channel views)
    for (int ch = 0; ch < channels; ++ch) {
      DMatrix sliceA = host_A;
      sliceA.height = H_p;
      sliceA.elements = &host_A.elements[ch * H_p * host_A.width];
      char nameA[64];
      sprintf(nameA, "host_A ch %d (%dx%d) padded", ch, H_p, W_p);
      printMatrixD(sliceA, nameA);
    }
    for (int k = 0; k < K; ++k) {
      for (int ch = 0; ch < channels; ++ch) {
        DMatrix sliceK = host_K;
        sliceK.height = FH;
        sliceK.elements = &host_K.elements[(k * (FH * channels) + ch * FH) * host_K.width];
        char nameK[64];
        sprintf(nameK, "host_K filt %d ch %d (%dx%d)", k, ch, FH, FW);
        printMatrixD(sliceK, nameK);
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

  if (verbose == 2) {
    printMatrixD(host_C, "host_C (convolution result)");
  }

  // --- Run convolution on the GPU: grid.z = number of filters, each block computes a 16x16 tile ---
  DMatrix device_A = MakeDeviceMatrixD(host_A, true);
  DMatrix device_K = MakeDeviceMatrixD(host_K, true);
  DMatrix device_C = MakeDeviceMatrixD(host_C, false);

  // NEW C2 SECTION **********************************************
  DMatrix device_C_cudnn;
  device_C_cudnn.width = W;
  device_C_cudnn.height = H * K;
  size_t C_size_bytes = device_C_cudnn.width * device_C_cudnn.height * sizeof(double);
  cudaMalloc(&device_C_cudnn.elements, C_size_bytes);
  
  // Initialize with zeros just to be safe
  cudaMemset(device_C_cudnn.elements, 0, C_size_bytes);

  run_cudnn_convolution(device_A.elements, device_K.elements, device_C_cudnn.elements,
                        channels, H_p, W_p, FH, FW, K);

  DMatrix host_C_gpu = MakeHostMatrixD(host_C.width, host_C.height);
  size_t sizeC = (size_t)host_C.width * host_C.height * sizeof(double);
  cudaMemcpy(host_C_gpu.elements, device_C_cudnn.elements, sizeC, cudaMemcpyDeviceToHost);
  
  // if (verbose == 1) {
  //   // Print first 8x8 of each filter slice for debugging to avoid flooding the console.
  //   print3DDlim(host_C_gpu, K, H, W, 64, 64);
  // }

  // Compute checksums (sum of all elements) on host and GPU and compare
  double sum_host = 0.0;
  double sum_gpu = 0.0;
  size_t total = (size_t)host_C.width * host_C.height;
  for (size_t i = 0; i < total; ++i) {
    sum_host += host_C.elements[i];
    sum_gpu += host_C_gpu.elements[i];
  }
  double diff = fabs(sum_host - sum_gpu);
  double tol = 1e-12;
  if (diff <= tol) {
    printf("Checksum OK (within tolerance)\n");
  } else {
    printf("Checksum FAILED (diff > tol)\n");
    printf("Checksum host: %.12f, gpu: %.12f, diff: %.12f\n", sum_host, sum_gpu, diff);
  }

  // Free device memory
  cudaFree(device_A.elements);
  cudaFree(device_K.elements);
  cudaFree(device_C.elements);
  cudaFree(device_C_cudnn.elements);
  free(host_C_gpu.elements);

  // Free allocated memory.
  free(host_A.elements);
  free(host_K.elements);
  free(host_C.elements);

  return 0;
}

