#pragma once

#include <ATen/ATen.h>

namespace millstability::detail {

// CUDA FP32 output: [stx+1, sty+1, 2*m+4, 2*m+4].
at::Tensor grid_matrices(
    int N, float Kt, float Kn, float w0x, float w0y, float zetax, float zetay,
    float m_tx, float m_ty, float aD, int up_or_down, int stx, int sty,
    float w_st, float w_fi, float o_st, float o_fi, int m, int device_id);

// Construct the principal block of active columns; its nonzero eigenvalues equal F's.
at::Tensor grid_reduced_matrices(
    int N, float Kt, float Kn, float w0x, float w0y, float zetax, float zetay,
    float m_tx, float m_ty, float aD, int up_or_down, int stx, int sty,
    float w_st, float w_fi, float o_st, float o_fi, int m, int device_id);

// Each parameter row produces one CUDA FP32 [2*m+4, 2*m+4] matrix.
at::Tensor parameter_matrices(
    at::Tensor parameters, int N, float aD, int up_or_down, int m,
    float w, float o, int device_id);

}
