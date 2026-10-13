#pragma once
#include <cuda_runtime.h>

namespace millstability::detail {
inline constexpr int kParameterGeometryMaxTeeth = 64;

// Shared-memory construction for 2 <= m <= 62.
// Geometry workspace: [m+1, N, 4] when N <= kParameterGeometryMaxTeeth.
void launch_parameter_columns(
    float* result, float* geometry_workspace,
    const float* parameters, int N, float aD, int up_or_down,
    int m, float w, float o, int n_params, cudaStream_t stream);
}
