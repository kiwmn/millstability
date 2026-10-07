#pragma once

#include <cuda_runtime.h>

namespace millstability {
namespace detail {

__global__ void stability_analysis_kernel(
    float* result_ptr, float* d_A0, float* d_invA0,
    float* d_hxx, float* d_hxy, float* d_hyx, float* d_hyy,
    int stx, int sty, float o_st, float o_fi, float w_st, float w_fi,
    int N, int m);

__global__ void compute_cutting_coefficients_kernel(
    float* hxx, float* hxy, float* hyx, float* hyy,
    float fist, float fiex, float Kt, float Kn,
    int N, int m, float dtr);

__global__ void multi_parameter_single_point_kernel(
    float* result_ptr, float* parameters, int N, float aD, int up_or_down,
    int m, float w, float o, int n_params);

}
}
