
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

void vecAdd(float *A, float *B, float *C, size_t N)
{
    size_t i;
    for (i = 0; i < N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv)
{
    float *a, *b, *c;
    size_t N;

    if (argc != 2)
    {
        printf("Usage: %s K \n", argv[0]);
        printf("K is the size in millions of the vectors to be added.\n");
        exit(0);
    }

    unsigned long long K = 0ULL;
    sscanf(argv[1], "%llu", &K);
    N = (size_t)(K * 1000000ULL);
    size_t size = N * sizeof(float);
    
    float* h_A = (float *)malloc(size);
    float* h_B = (float *)malloc(size);
    float* h_C = (float *)malloc(size);

    size_t i;
    for (i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    initialize_timer();
    start_timer();

    vecAdd(h_A, h_B, h_C, N);

    // Compute elapsed time 
    stop_timer();
    double time = elapsed_time();

    printf( "Time: %lf (sec)\n", time);

    for (i = 0; i < N; ++i) {
        float val = h_C[i];
        if (fabs(val - N) > 1e-5)
            break;
    }
    printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
