#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

float dp(long N, float *pA, float *pB)
{
    float R = 0.0;
    int j;
    for (j=0;j<N;j++)
        R += pA[j]*pB[j];
    return R;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <size> [measurements]\n", argv[0]);
        return 1;
    }

    long N = atol(argv[1]);
    int measurements = atoi(argv[2]);

    float *A = malloc((size_t)N * sizeof(float));
    float *B = malloc((size_t)N * sizeof(float));
    for (long i = 0; i < N; ++i)
    {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    int half = measurements / 2;
    int timed = measurements - half; // separate vars for half and time to cover odd measurements

    volatile float result = 0.0f;
    for (int i = 0; i < half; ++i)
    {
        result = dp(N, A, B);
    }
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < timed; ++i)
    {
        result = dp(N, A, B);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    time_t sec = end.tv_sec - start.tv_sec;
    long nsec = end.tv_nsec - start.tv_nsec;
    if (nsec < 0)
    {
        sec--;
        nsec += 1e9;
    }
    double elapsed = (double)sec + (double)nsec / 1e9;
    double average_time = elapsed / timed;

    int bytesPerIteration = 8; // 2 loads (A[j], B[j]) * 4 bytes each
    double totalBytes = bytesPerIteration * (double)N * (double)timed;
    double totalGB = totalBytes / 1e9;
    double bandwidth = totalGB / elapsed;

    int flopsPerIteration = 2; // 1 add + 1 mult
    double totalFlops = flopsPerIteration * (double)N * (double)timed;
    double totalGFlops = totalFlops / 1e9;
    double GFLOPS = totalGFlops / elapsed;

    printf("N: %ld  <T>: %.3f sec  B: %.3f GB/sec  F: %.3f GFLOP/sec\n", N, average_time, bandwidth, GFLOPS);

    free(A);
    free(B);
    return 0;
}