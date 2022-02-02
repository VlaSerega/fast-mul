#include "fast_mul.h"

inline static void
mini_multiplication(const float *aligned_block_A, const float *aligned_block_B, float *aligned_block_C) {
    for (int i = 0; i < BM; i += REGA) {
        for (int j = 0; j < BN; j += VECTOR_SIZE * REGB / sizeof(float)) {
            __m256 res[REGA][REGB] = {};

            for (int k = 0; k < BK; ++k) {
                for (int ra = 0; ra < REGA; ++ra) {
                    __m256 a = aligned_block_A[(i + ra) * BK + k] - (__m256) {};
                    for (int rb = 0; rb < REGB; ++rb) {
                        __m256 b = *(__m256 *) &aligned_block_B[k * BN + j + rb * VECTOR_SIZE / sizeof(float)];
                        res[ra][rb] += a * b;
                    }
                }
            }

            for (int ra = 0; ra < REGA; ++ra) {
                for (int rb = 0; rb < REGB; ++rb) {
                    *(__m256 *) &aligned_block_C[(i + ra) * BN + j + rb * VECTOR_SIZE / sizeof(float)] += res[ra][rb];
                }
            }
        }
    }
}

inline static void
copy_block(const float *src, int ld_src, float *dest, int N, int M, int row_count, int column_count) {
    for (int i = 0; i < row_count; ++i) {
        memcpy(dest + i * M, src + ld_src * i, column_count * sizeof(float));
        memset(dest + i * M + column_count, 0, (M - column_count) * sizeof(float));
    }

    for (int i = row_count; i < N; ++i) {
        memset(dest + i * M, 0, M * sizeof(float));
    }
}

inline static void write_block(float *src, int lda_src, const float *dest, int M, int row_count, int column_count) {
    for (int i = 0; i < row_count; ++i)
        for (int j = 0; j < column_count; ++j)
            src[lda_src * i + j] += dest[M * i + j];
}

void multiplication(const int M, const int N, const int K, const float *A, const float *B, float *C) {
    int count_blocks_by_M = (M / BM) + (M % BM ? 1 : 0);
    int count_blocks_by_N = (N / BN) + (M % BN ? 1 : 0);
    int count_blocks = count_blocks_by_M;
    int flag = 1;

    if (count_blocks_by_M < count_blocks_by_N) {
        count_blocks = count_blocks_by_N;
        flag = 0;
    }

    memset(C, 0, M * N * sizeof(float));

#pragma omp parallel proc_bind(spread)
    {
        float *aligned_block_A = aligned_alloc(CACHE_LINE_SIZE, sizeof(float) * BM * BK);
        float *aligned_block_B = aligned_alloc(CACHE_LINE_SIZE, sizeof(float) * BK * BN);
        float *aligned_block_C = aligned_alloc(CACHE_LINE_SIZE, sizeof(float) * BM * BN);

        int start_i, end_i;
        int start_j, end_j;

        int blocks_per_thread = count_blocks / omp_get_num_threads();
        int start = (blocks_per_thread * omp_get_thread_num() +
                     (omp_get_thread_num() < count_blocks % omp_get_num_threads()
                      ? omp_get_thread_num() : count_blocks % omp_get_num_threads()));
        int end = start +
                      (blocks_per_thread + (omp_get_thread_num() < count_blocks % omp_get_num_threads() ? 1 : 0));

        if (flag) {
            start_i = start * BM;
            end_i = min(end * BM, M);

            start_j = 0;
            end_j = N;
        } else {
            start_j = start * BN;
            end_j = min(end * BN, N);

            start_i = 0;
            end_i = M;
        }

        for (int i = start_i; i < end_i; i += BM) {
            for (int j = start_j; j < end_j; j += BN) {
                memset(aligned_block_C, 0, BM * BN * sizeof(float));
                for (int k = 0; k < K; k += BK) {
                    copy_block(A + i * K + k, K, aligned_block_A, BM, BK, min(BM, M - i), min(BK, K - k));
                    copy_block(B + k * N + j, N, aligned_block_B, BK, BN, min(BK, K - k), min(BN, N - j));

                    mini_multiplication(aligned_block_A, aligned_block_B, aligned_block_C);

                }
                write_block(C + i * N + j, N, aligned_block_C, BN, min(BM, M - i), min(BN, N - j));
            }
        }


        free(aligned_block_A);
        free(aligned_block_B);
        free(aligned_block_C);
    }
}
