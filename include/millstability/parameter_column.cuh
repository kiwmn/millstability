#pragma once
#include <cuda_runtime.h>

namespace millstability::detail {
inline constexpr int kParameterGeometryMaxTeeth = 64;

// Requires 2 <= m <= 40. For N <= 64, workspace holds (m + 1) * N * 4 floats;
// for larger N, pass nullptr and compute geometry within each block.
void launch_parameter_columns(
    float* result, float* geometry_workspace,
    const float* parameters, int N, float aD, int up_or_down,
    int m, float w, float o, int n_params, cudaStream_t stream);
}
