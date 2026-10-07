#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace millstability {
namespace detail {

inline void check_device_id(int device_id) {
    TORCH_CHECK(device_id >= 0 && device_id < c10::cuda::device_count(),
                "device_id must identify an available CUDA device");
}

inline void check_cuda_float_tensor(const at::Tensor& tensor,
                                    const char* name, int device_id) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must have dtype float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.get_device() == device_id,
                name, " must be on CUDA device ", device_id);
}

inline void check_finite(float value, const char* name) {
    TORCH_CHECK(std::isfinite(value), name, " must be finite");
}

// The fixed-size device arrays support at most 40 time steps.
inline void check_discretization(int N, float aD, int up_or_down, int m) {
    TORCH_CHECK(N > 0 && N < std::numeric_limits<int>::max(), "N must be positive");
    TORCH_CHECK(m >= 1 && m <= 40,
                "m must be in [1, 40] for the fixed-size matrix workspace");
    TORCH_CHECK(std::isfinite(aD) && aD >= 0.0f && aD <= 1.0f,
                "aD must be finite and in [0, 1]");
    TORCH_CHECK(up_or_down == 1 || up_or_down == -1,
                "up_or_down must be 1 (up-milling) or -1 (down-milling)");
}

inline void check_process_point(float w, float o) {
    TORCH_CHECK(std::isfinite(w) && w >= 0.0f,
                "cutting depth must be finite and nonnegative");
    TORCH_CHECK(std::isfinite(o) && o > 0.0f,
                "spindle speed must be finite and positive");
}

inline void check_physical_parameters(float Kt, float Kn, float w0x, float w0y,
                                     float zetax, float zetay,
                                     float m_tx, float m_ty) {
    check_finite(Kt, "Kt");
    check_finite(Kn, "Kn");
    check_finite(zetax, "zetax");
    check_finite(zetay, "zetay");
    TORCH_CHECK(std::isfinite(w0x) && w0x > 0.0f &&
                    std::isfinite(w0y) && w0y > 0.0f,
                "natural frequencies must be finite and positive");
    TORCH_CHECK(std::isfinite(m_tx) && m_tx > 0.0f &&
                    std::isfinite(m_ty) && m_ty > 0.0f,
                "masses must be finite and positive");
}

// Matrix offsets use 32-bit integers inside the CUDA kernels.
inline void check_matrix_index_capacity(int64_t batch_size, int m) {
    const int64_t order = 2 * m + 4;
    TORCH_CHECK(batch_size <= std::numeric_limits<int>::max() / (order * order),
                "matrix batch is too large for the 32-bit kernel indices");
}

}
}
