#pragma once

#include <cuda_runtime.h>

namespace millstability::detail {

// Shared-memory construction for 2 <= m <= 62; phi has (stx+1)*64 floats.
void launch_column_grid(
    float* result, float* phi_workspace,
    const float* A0, const float* invA0,
    const float* hxx, const float* hxy, const float* hyx, const float* hyy,
    int stx, int sty, float o_st, float o_fi, float w_st, float w_fi,
    int N, int m, cudaStream_t stream);

void launch_phi_workspace(
    float* phi_workspace, const float* A0, const float* invA0,
    int stx, float o_st, float o_fi, int N, int m, cudaStream_t stream);

// map[0] is the count; map[1..count] contains sorted original column indices.
// Allocate 2*m+5 integer entries.
void launch_active_columns(
    int* map, const float* hxx, const float* hxy,
    const float* hyx, const float* hyy, int m, cudaStream_t stream);

// Construct F[S,S], where active_map stores the nonzero-column support S.
void launch_reduced_column_grid(
    float* result, float* phi_workspace,
    const float* A0, const float* invA0,
    const float* hxx, const float* hxy, const float* hyx, const float* hyy,
    int stx, int sty, float o_st, float o_fi, float w_st, float w_fi,
    int N, int m, const int* active_map, int reduced_order,
    cudaStream_t stream);

}
