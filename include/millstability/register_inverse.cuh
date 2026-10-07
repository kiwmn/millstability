#pragma once
#include <cuda_runtime.h>

namespace millstability::detail {

// Unroll the 4 x 8 augmented matrix so pivoted elimination can use registers.
__device__ __forceinline__ void inverse_register_4x4(const float* A, float* result) {
    float aug[4][8];
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            aug[row][col] = A[row * 4 + col];
            aug[row][col + 4] = row == col ? 1.0f : 0.0f;
        }
    }
    #pragma unroll
    for (int pivot = 0; pivot < 4; ++pivot) {
        int max_row = pivot;
        float max_value = fabsf(aug[pivot][pivot]);
        #pragma unroll
        for (int row = pivot + 1; row < 4; ++row) {
            if (fabsf(aug[row][pivot]) > max_value) {
                max_row = row;
                max_value = fabsf(aug[row][pivot]);
            }
        }
        #pragma unroll
        for (int row = pivot + 1; row < 4; ++row) {
            if (max_row == row) {
                #pragma unroll
                for (int col = 0; col < 8; ++col) {
                    float temp = aug[pivot][col];
                    aug[pivot][col] = aug[row][col];
                    aug[row][col] = temp;
                }
            }
        }
        const float pivot_value = aug[pivot][pivot];
        if (fabsf(pivot_value) > 1e-10f) {
            #pragma unroll
            for (int col = 0; col < 8; ++col) aug[pivot][col] /= pivot_value;
            #pragma unroll
            for (int row = 0; row < 4; ++row) {
                if (row != pivot) {
                    const float factor = aug[row][pivot];
                    #pragma unroll
                    for (int col = 0; col < 8; ++col) {
                        aug[row][col] -= factor * aug[pivot][col];
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        #pragma unroll
        for (int col = 0; col < 4; ++col) result[row * 4 + col] = aug[row][col + 4];
    }
}
}
