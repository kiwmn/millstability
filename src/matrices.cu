#include "millstability/matrices.h"
#include "millstability/matrix_kernels.cuh"
#include "millstability/column_kernel.cuh"
#include "millstability/parameter_column.cuh"
#include "millstability/global_column_kernel.cuh"
#include "millstability/validation.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace millstability::detail {
namespace {

void validate_grid_parameters(
    int N, float Kt, float Kn, float w0x, float w0y, float zetax, float zetay,
    float m_tx, float m_ty, float aD, int up_or_down, int stx, int sty,
    float w_st, float w_fi, float o_st, float o_fi, int m, int device_id) {
    check_device_id(device_id);
    check_discretization(N, aD, up_or_down, m);
    check_physical_parameters(Kt, Kn, w0x, w0y, zetax, zetay, m_tx, m_ty);
    check_process_point(w_st, o_st);
    check_process_point(w_fi, o_fi);
    TORCH_CHECK(stx > 0 && sty > 0, "stx and sty must be positive grid step counts");
    const auto* properties = at::cuda::getDeviceProperties(device_id);
    TORCH_CHECK(int64_t(sty) + 1 <= properties->maxThreadsPerBlock,
                "sty + 1 exceeds the CUDA device maximum threads per block");
    TORCH_CHECK(int64_t(stx) + 1 <= properties->maxGridSize[0],
                "stx + 1 exceeds the CUDA device maximum grid size");
    const int64_t batch_size = (int64_t(stx) + 1) * (int64_t(sty) + 1);
    check_matrix_index_capacity(batch_size, m);
}

// A0 and invA0 are two row-major 4x4 system matrices.
at::Tensor make_system_matrices(float w0x, float w0y, float zetax, float zetay,
                                float m_tx, float m_ty, const at::Device& device) {
    auto host = at::zeros({2, 16}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));

    float* A0 = host.data_ptr<float>();
    A0[0*4 + 0] = -zetax * w0x;
    A0[0*4 + 2] = 1.0f/m_tx;
    A0[2*4 + 0] = (zetax*zetax - 1)*m_tx * w0x * w0x;
    A0[1*4 + 1] = -zetay * w0y;
    A0[2*4 + 2] = -zetax * w0x;
    A0[1*4 + 3] = 1.0f/m_ty;
    A0[3*4 + 1] = (zetay*zetay - 1)*m_ty * w0y * w0y;
    A0[3*4 + 3] = -zetay * w0y;

    float* invA0 = A0 + 16;

    // Gauss-Jordan elimination on [A0 | I] with partial pivoting.
    float augmented[32];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            augmented[i * 8 + j] = A0[i * 4 + j];
            augmented[i * 8 + j + 4] = (i == j) ? 1.0f : 0.0f;
        }
    }

    for (int pivot = 0; pivot < 4; pivot++) {

        int max_row = pivot;
        for (int i = pivot + 1; i < 4; i++) {
            if (fabsf(augmented[i * 8 + pivot]) > fabsf(augmented[max_row * 8 + pivot])) {
                max_row = i;
            }
        }

        if (max_row != pivot) {
            for (int j = 0; j < 8; j++) {
                float temp = augmented[pivot * 8 + j];
                augmented[pivot * 8 + j] = augmented[max_row * 8 + j];
                augmented[max_row * 8 + j] = temp;
            }
        }

        float pivot_val = augmented[pivot * 8 + pivot];
        if (fabsf(pivot_val) > 1e-10f) {
            for (int j = 0; j < 8; j++) {
                augmented[pivot * 8 + j] /= pivot_val;
            }

            for (int i = 0; i < 4; i++) {
                if (i != pivot) {
                    float factor = augmented[i * 8 + pivot];
                    for (int j = 0; j < 8; j++) {
                        augmented[i * 8 + j] -= factor * augmented[pivot * 8 + j];
                    }
                }
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            invA0[i * 4 + j] = augmented[i * 8 + j + 4];
        }
    }

    // The blocking copy keeps the temporary host buffer alive through transfer.
    return host.to(device);
}

}

static at::Tensor grid_matrices_impl(
    int N, float Kt, float Kn, float w0x, float w0y, float zetax, float zetay,
    float m_tx, float m_ty, float aD, int up_or_down, int stx, int sty,
    float w_st, float w_fi, float o_st, float o_fi, int m, int device_id,
    bool reduce) {
    validate_grid_parameters(N, Kt, Kn, w0x, w0y, zetax, zetay, m_tx, m_ty,
                            aD, up_or_down, stx, sty, w_st, w_fi, o_st, o_fi, m, device_id);

    const c10::cuda::CUDAGuard device_guard(at::Device(at::kCUDA, device_id));
    const int order = 2 * m + 4;
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_id);

    auto system = make_system_matrices(w0x, w0y, zetax, zetay, m_tx, m_ty,
                                       at::Device(at::kCUDA, device_id));
    auto coefficients = at::empty({4, m + 1}, options);
    float* A0 = system.data_ptr<float>();
    float* invA0 = A0 + 16;
    float* hxx = coefficients.data_ptr<float>();
    float* hxy = hxx + m + 1;
    float* hyx = hxy + m + 1;
    float* hyy = hyx + m + 1;

    // Construct on the caller's current PyTorch stream.
    const auto stream = at::cuda::getCurrentCUDAStream(device_id).stream();

    float fist, fiex;
    if (up_or_down == 1) {
        fist = 0.0f;
        fiex = acosf(1.0f - 2.0f * aD);
    } else {
        fist = acosf(2.0f * aD - 1.0f);
        fiex = M_PI;
    }
    const float dtr = 2.0f * M_PI / N / m;

    constexpr int coefficient_threads = 256;
    const int coefficient_blocks = (m + coefficient_threads) / coefficient_threads;
    compute_cutting_coefficients_kernel<<<coefficient_blocks, coefficient_threads, 0, stream>>>(
        hxx, hxy, hyx, hyy, fist, fiex, Kt, Kn, N, m, dtr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    at::Tensor active_map;
    at::Tensor active_flags;
    int output_order = order;
    const bool global_path = m > 62;
    if (reduce && m >= 2) {
        active_map = at::empty({order + 1}, options.dtype(at::kInt));
        if (global_path) {
            active_flags = at::empty({order}, options.dtype(at::kInt));
            launch_global_active_columns(active_map.data_ptr<int>(), active_flags.data_ptr<int>(),
                hxx, hxy, hyx, hyy, m, stream);
        } else {
            launch_active_columns(active_map.data_ptr<int>(), hxx, hxy, hyx, hyy, m, stream);
        }
        output_order = active_map[0].item<int>();
        TORCH_CHECK(output_order >= 4 && output_order <= order,
                    "invalid active column count");
    }
    auto result = at::empty({stx + 1, sty + 1, output_order, output_order}, options);

    if (global_path) {
        const int64_t points = int64_t(stx + 1) * (sty + 1);
        TORCH_CHECK(points <= std::numeric_limits<int>::max(),
                    "global matrix constructor batch is too large");
        auto phi = at::empty({stx + 1, 64}, result.options());
        const int64_t steps_elements = points * int64_t(m) * 36;
        TORCH_CHECK(steps_elements <= std::numeric_limits<int64_t>::max(),
                    "global step workspace size overflow");
        auto steps = at::empty({steps_elements}, result.options());
        launch_global_column_grid(result.data_ptr<float>(), phi.data_ptr<float>(), steps.data_ptr<float>(),
            A0, invA0, hxx, hxy, hyx, hyy,
            stx, sty, o_st, o_fi, w_st, w_fi, N, m,
            active_map.defined() ? active_map.data_ptr<int>() : nullptr,
            output_order, stream);
    } else if (m >= 2) {
        auto phi = at::empty({stx + 1, 64}, result.options());
        if (reduce) {
            launch_reduced_column_grid(result.data_ptr<float>(), phi.data_ptr<float>(),
                A0, invA0, hxx, hxy, hyx, hyy,
                stx, sty, o_st, o_fi, w_st, w_fi, N, m,
                active_map.const_data_ptr<int>(), output_order, stream);
        } else {
            launch_column_grid(result.data_ptr<float>(), phi.data_ptr<float>(),
                A0, invA0, hxx, hxy, hyx, hyy,
                stx, sty, o_st, o_fi, w_st, w_fi, N, m, stream);
        }
    // At m=1, current and delayed columns overlap and must accumulate.
    } else {
        const size_t shared_memory_bytes = 4 * (m + 1) * sizeof(float);
        stability_analysis_kernel<<<stx + 1, sty + 1, shared_memory_bytes, stream>>>(
            result.data_ptr<float>(), A0, invA0, hxx, hxy, hyx, hyy,
            stx, sty, o_st, o_fi, w_st, w_fi, N, m);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return result;
}

at::Tensor grid_matrices(
    int N, float Kt, float Kn, float w0x, float w0y, float zetax, float zetay,
    float m_tx, float m_ty, float aD, int up_or_down, int stx, int sty,
    float w_st, float w_fi, float o_st, float o_fi, int m, int device_id) {
    return grid_matrices_impl(N, Kt, Kn, w0x, w0y, zetax, zetay, m_tx, m_ty,
        aD, up_or_down, stx, sty, w_st, w_fi, o_st, o_fi, m, device_id, false);
}

at::Tensor grid_reduced_matrices(
    int N, float Kt, float Kn, float w0x, float w0y, float zetax, float zetay,
    float m_tx, float m_ty, float aD, int up_or_down, int stx, int sty,
    float w_st, float w_fi, float o_st, float o_fi, int m, int device_id) {
    return grid_matrices_impl(N, Kt, Kn, w0x, w0y, zetax, zetay, m_tx, m_ty,
        aD, up_or_down, stx, sty, w_st, w_fi, o_st, o_fi, m, device_id, true);
}

// Rows are [Kt,Kn,w0x,w0y,zetax,zetay,m_tx,m_ty]; preserve their output order.
at::Tensor parameter_matrices(
    at::Tensor parameters, int N, float aD, int up_or_down, int m,
    float w, float o, int device_id) {
    check_device_id(device_id);
    check_discretization(N, aD, up_or_down, m);
    check_process_point(w, o);
    check_cuda_float_tensor(parameters, "parameters", device_id);
    TORCH_CHECK(parameters.dim() == 2 && parameters.size(1) == 8,
                "parameters must have shape [n_params, 8]");
    const int64_t batch_size = parameters.size(0);
    check_matrix_index_capacity(batch_size, m);
    const c10::cuda::CUDAGuard device_guard(parameters.device());

    TORCH_CHECK(at::isfinite(parameters).all().item<bool>(),
                "parameters must contain only finite values");
    TORCH_CHECK(parameters.slice(1, 2, 4).gt(0).all().item<bool>(),
                "natural frequencies must be positive");
    TORCH_CHECK(parameters.slice(1, 6, 8).gt(0).all().item<bool>(),
                "masses must be positive");

    const int order = 2 * m + 4;
    auto result = at::empty({batch_size, order, order}, parameters.options());
    // Preserve empty output shape without launching a CUDA block.
    if (batch_size == 0) {

        return result;
    }
    constexpr int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    // Construct on the caller's current PyTorch stream.
    const auto stream = at::cuda::getCurrentCUDAStream(device_id).stream();
    if (m > 62) {
        const int64_t h_stride = int64_t(4) * (m + 1);
        const int64_t workspace_elements = batch_size * (64 + h_stride + int64_t(36) * m);
        TORCH_CHECK(workspace_elements <= std::numeric_limits<int64_t>::max(),
                    "global parameter workspace size overflow");
        auto workspace = at::empty({workspace_elements}, parameters.options());
        TORCH_CHECK(batch_size <= std::numeric_limits<int>::max(),
                    "parameter batch is too large for global constructor");
        launch_global_parameter_matrices(
            result.data_ptr<float>(), workspace.data_ptr<float>(), parameters.data_ptr<float>(),
            N, aD, up_or_down, m, w, o, static_cast<int>(batch_size), stream);
    } else if (m >= 2) {
        auto geometry = N <= kParameterGeometryMaxTeeth
            ? at::empty({m + 1, N, 4}, parameters.options()) : at::Tensor();
        launch_parameter_columns(
            result.data_ptr<float>(),
            geometry.defined() ? geometry.data_ptr<float>() : nullptr,
            parameters.data_ptr<float>(), N, aD, up_or_down,
            m, w, o, static_cast<int>(batch_size), stream);
    } else {

        multi_parameter_single_point_kernel<<<blocks, threads, 0, stream>>>(
            result.data_ptr<float>(), parameters.data_ptr<float>(), N, aD,
            up_or_down, m, w, o, static_cast<int>(batch_size));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return result;
}

}
