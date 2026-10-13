#pragma once

#include <cuda_runtime.h>

namespace millstability::detail {

// Global workspaces: phi[stx+1,64], steps[points,m,36].
// A null active_map constructs the complete matrix.
void launch_global_column_grid(
    float* result, float* phi_workspace, float* step_workspace,
    const float* A0, const float* invA0,
    const float* hxx, const float* hxy, const float* hyx, const float* hyy,
    int stx, int sty, float o_st, float o_fi, float w_st, float w_fi,
    int N, int m, const int* active_map, int reduced_order,
    cudaStream_t stream);

void launch_global_active_columns(
    int* map, int* active, const float* hxx, const float* hxy,
    const float* hyx, const float* hyy, int m, cudaStream_t stream);

// Output: [n_params,2*m+4,2*m+4]; the caller owns the step workspace.
void launch_global_parameter_matrices(
    float* result, float* step_workspace, const float* parameters,
    int N, float aD, int up_or_down, int m, float w, float o,
    int n_params, cudaStream_t stream);

}
