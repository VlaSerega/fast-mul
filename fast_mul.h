#ifndef FASTMULHUAWEITASK_FAST_MUL_H
#define FASTMULHUAWEITASK_FAST_MUL_H

#include <stdlib.h>
#include <immintrin.h>
#include <fmaintrin.h>
#include <memory.h>
#include <omp.h>

#define VECTOR_SIZE 32

#define REGA 3
#define REGB 4

#define BM (REGA * 50)
#define BN ((int) (REGB * 4 * VECTOR_SIZE / sizeof(float)))
#define BK 64

#define CACHE_LINE_SIZE 64

#define min(a, b) (a < b ? a : b)

void multiplication(const int M, const int N, const int K, const float *A, const float *B, float *C);

#endif
