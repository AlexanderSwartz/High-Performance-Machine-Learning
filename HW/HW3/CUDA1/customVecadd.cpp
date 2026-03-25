
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void vecAdd(float *A, float *B, float *C, int N)
{
    int i;
    for (i = 0; i < N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv)
{
    float *a, *b, *c;
    int N;

    if (argc != 2)
    {
        printf("Usage: %s N \n", argv[0]);
        printf("N is the size in millions of the vectors to be added.\n");
        exit(0);
    }
    else
    {
        sscanf(argv[1], "%d", &N);
    }
    N = N * 1000000;

    float* h_A = (float *)malloc(N * sizeof(float));
    float* h_B = (float *)malloc(N * sizeof(float));
    float* h_C = (float *)malloc(N * sizeof(float));

    int i;
    for (i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    vecAdd(h_A, h_B, h_C, N);

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
