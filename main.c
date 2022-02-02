#include <stdio.h>
#include <time.h>
#include "fast_mul.h"

int main(int argc, char **argv) {
    char *s_end;
    if (argc != 4){
        printf("Should 3 args!\n");
        return 0;
    }
    int M = strtol(argv[1], &s_end, 10),
            N = strtol(argv[2], &s_end, 10),
            K = strtol(argv[3], &s_end, 10);

    float *A = malloc(M * K * sizeof(float));
    float *B = malloc(K * N * sizeof(float));
    float *C = malloc(M * N * sizeof(float));
    float *R = malloc(M * N * sizeof(float));

    srand(time(NULL));

    for (int i = 0; i < M * K; ++i) {
        A[i] = rand() % 40;
    }


    for (int i = 0; i < K * N; ++i) {
        B[i] = rand() % 40;
    }

    struct timespec start, end;

    clock_gettime(CLOCK_REALTIME, &start);

    multiplication(M, N, K, A, B, C);

    clock_gettime(CLOCK_REALTIME, &end);

//    for (int i = 0; i < M; ++i) {
//        for (int j = 0; j < N; ++j) {
//            R[i * N + j] = 0.f;
//            for (int k = 0; k < K; ++k) {
//                R[i * N + j] += A[i * K + k] * B[k * N + j];
//            }
//
//            if (R[i * N + j] - C[i * N + j] > 0.000001 || R[i * N + j] - C[i * N + j] < -0.000001) {
//                printf("%d %d %lf %lf\n", i, j, R[i * N + j], C[i * N + j]);
//                getchar();
//            }
//        }
//    }

    printf("%lf\n", end.tv_sec - start.tv_sec + (double) (end.tv_nsec - start.tv_nsec) / 1000000000);

    free(A);
    free(B);
    free(C);
    free(R);
    return 0;
}
