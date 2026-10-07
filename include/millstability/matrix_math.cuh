#pragma once

#include <cuda_runtime.h>
#include <math.h>

namespace millstability {
namespace detail {

// Gauss-Jordan inverse with partial pivoting; matrices use row-major storage.
__device__ inline void matrix_inverse_4x4(const float* A, float* result) {

    float aug[32];
    const int n = 4;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * 8 + j] = A[i * n + j];
            aug[i * 8 + j + n] = (i == j) ? 1.0f : 0.0f;
        }
    }

    for (int pivot = 0; pivot < n; pivot++) {

        int max_row = pivot;
        for (int i = pivot + 1; i < n; i++) {
            if (fabsf(aug[i * 8 + pivot]) > fabsf(aug[max_row * 8 + pivot])) {
                max_row = i;
            }
        }

        if (max_row != pivot) {
            for (int j = 0; j < 8; j++) {
                float temp = aug[pivot * 8 + j];
                aug[pivot * 8 + j] = aug[max_row * 8 + j];
                aug[max_row * 8 + j] = temp;
            }
        }

        float pivot_val = aug[pivot * 8 + pivot];
        if (fabsf(pivot_val) > 1e-10f) {
            for (int j = 0; j < 8; j++) {
                aug[pivot * 8 + j] /= pivot_val;
            }

            for (int i = 0; i < n; i++) {
                if (i != pivot) {
                    float factor = aug[i * 8 + pivot];
                    for (int j = 0; j < 8; j++) {
                        aug[i * 8 + j] -= factor * aug[pivot * 8 + j];
                    }
                }
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] = aug[i * 8 + j + n];
        }
    }
}

// Taylor approximation through order 10, stopping when the term norm is below 1e-10.
// A must already include the time-step factor; dt is unused.
__device__ inline void matrix_expm_4x4(const float* A, float dt, float* result) {
    const int n = 4;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    float A_power[16];
    float temp[16];

    for (int i = 0; i < 16; i++) {
        A_power[i] = A[i];
    }

    for (int i = 0; i < 16; i++) {
        result[i] += A_power[i];
    }

    float factorial = 1.0f;
    for (int k = 2; k <= 10; k++) {
        factorial *= k;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                temp[i * n + j] = 0.0f;
                for (int l = 0; l < n; l++) {
                    temp[i * n + j] += A_power[i * n + l] * A[l * n + j];
                }
            }
        }

        for (int i = 0; i < 16; i++) {
            A_power[i] = temp[i];
        }

        for (int i = 0; i < 16; i++) {
            result[i] += A_power[i] / factorial;
        }

        float term_norm = 0.0f;
        for (int i = 0; i < 16; i++) {
            term_norm += fabsf(A_power[i] / factorial);
        }
        if (term_norm < 1e-10f) {
            break;
        }
    }
}

}
}
