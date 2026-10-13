#pragma once

#include <ATen/ATen.h>

namespace millstability {

// Output axes: spindle speed, axial cutting depth.
at::Tensor milling_stability_ei_cuda(
    int N, float Kt, float Kn, float w0x, float w0y, float zetax, float zetay,
    float m_tx, float m_ty, float aD, int up_or_down, int stx, int sty,
    float w_st, float w_fi, float o_st, float o_fi, int m, int device_id);

// Compute the fixed 201x101 process grid.
at::Tensor test_ei(int device_id);

// Return one CUDA FP32 EI for each row of eight physical parameters.
at::Tensor multi_parameter_single_point_ei(
    at::Tensor parameters, int N, float aD, int up_or_down, int m,
    float w, float o, int device_id);

}
