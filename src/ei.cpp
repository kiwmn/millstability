#include "millstability/ei.h"
#include "millstability/eigensolver.h"
#include "millstability/matrices.h"

#include <cmath>

namespace millstability {

at::Tensor milling_stability_ei_cuda(
    int N, float Kt, float Kn, float w0x, float w0y, float zetax, float zetay,
    float m_tx, float m_ty, float aD, int up_or_down, int stx, int sty,
    float w_st, float w_fi, float o_st, float o_fi, int m, int device_id) {

    return detail::spectral_radius(detail::grid_matrices(
        N, Kt, Kn, w0x, w0y, zetax, zetay, m_tx, m_ty, aD, up_or_down,
        stx, sty, w_st, w_fi, o_st, o_fi, m, device_id));
}

at::Tensor test_ei(int device_id) {

    // Reference process: two teeth, 5,000-25,000 rpm and 0-10 mm axial depth.
    const float frequency = 922.0f * 2.0f * M_PI;
    return milling_stability_ei_cuda(
        2, 6e8f, 2e8f, frequency, frequency, 0.011f, 0.011f,
        0.03993f, 0.03993f, 0.05f, 1, 200, 100,
        0e-3f, 10e-3f, 5e3f, 25e3f, 40, device_id);
}

at::Tensor multi_parameter_single_point_ei(
    at::Tensor parameters, int N, float aD, int up_or_down, int m,
    float w, float o, int device_id) {

    return detail::spectral_radius(detail::parameter_matrices(
        parameters, N, aD, up_or_down, m, w, o, device_id));
}

}
