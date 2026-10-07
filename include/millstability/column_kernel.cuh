#pragma once

#include <cuda_runtime.h>

namespace millstability::detail {

// Requires 2 <= m <= 40. Workspace: (stx + 1) * 64 floats for integration matrices.
void launch_column_grid(
    float* result, float* phi_workspace,
    const float* A0, const float* invA0,
    const float* hxx, const float* hxy, const float* hyx, const float* hyy,
    int stx, int sty, float o_st, float o_fi, float w_st, float w_fi,
    int N, int m, cudaStream_t stream);

}
