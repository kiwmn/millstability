#include "millstability/eigensolver.h"
#include "millstability/ei_reduction.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cusolverDn.h>
#include <algorithm>
#include <limits>

#if CUDART_VERSION < 12080
#error "millstability requires CUDA Toolkit 12.8 or newer for cusolverDnXgeev"
#endif

namespace millstability::detail {
namespace {

void check_solver(cusolverStatus_t status, const char* operation) {

    TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS,
                operation, " failed (cuSOLVER status ", int(status),
                "). Check CUDA/cuSOLVER >= 12.8 and available workspace memory.");
}

// Own the cuSOLVER handle across normal returns and exceptions.
class SolverHandle {
public:

    SolverHandle() { check_solver(cusolverDnCreate(&handle_), "cusolverDnCreate"); }
    ~SolverHandle() { if (handle_) cusolverDnDestroy(handle_); }
    SolverHandle(const SolverHandle&) = delete;
    SolverHandle& operator=(const SolverHandle&) = delete;
    cusolverDnHandle_t get() const { return handle_; }
private:
    cusolverDnHandle_t handle_ = nullptr;
};

constexpr auto kNoVectors = CUSOLVER_EIG_MODE_NOVECTOR;
constexpr auto kReal = CUDA_R_32F;

// Limit the extra column-major copy to this many matrices.
constexpr int64_t kSolverBatch = 256;

}

// These nonsymmetric matrices may have complex eigenvalues.
at::Tensor spectral_radius(const at::Tensor& matrices) {

    TORCH_CHECK(matrices.is_cuda(), "matrices must be a CUDA tensor");
    TORCH_CHECK(matrices.scalar_type() == at::kFloat, "matrices must have dtype float32");
    TORCH_CHECK(matrices.dim() >= 2 && matrices.size(-2) == matrices.size(-1),
                "matrices must have shape [..., n, n] (square matrices)");
    const int64_t n = matrices.size(-1);
    TORCH_CHECK(n > 0 && n <= 46340, "matrix order n must be in [1, 46340]");
    TORCH_CHECK(!matrices.requires_grad(), "spectral_radius does not support autograd");
    const c10::cuda::CUDAGuard guard(matrices.device());

    auto shape = matrices.sizes().vec();
    shape.resize(shape.size() - 2);
    auto ei = at::empty(shape, matrices.options());
    const int64_t count = ei.numel();
    if (count == 0) return ei;
    TORCH_CHECK(at::isfinite(matrices).all().item<bool>(),
                "matrices contain NaN or Inf before eigenvalue computation");

    auto flat = matrices.reshape({count, n, n});
    auto eigenvalues = at::empty({std::min(count, kSolverBatch), 2 * n}, matrices.options());
    auto info = at::empty({std::min(count, kSolverBatch)}, matrices.options().dtype(at::kInt));
    const auto stream = at::cuda::getCurrentCUDAStream(matrices.get_device());

    SolverHandle solver;
    check_solver(cusolverDnSetStream(solver.get(), stream.stream()), "cusolverDnSetStream");

    // Xgeev uses CPU and GPU workspaces, reused for every matrix in this call.
    size_t device_bytes = 0, host_bytes = 0;
    check_solver(cusolverDnXgeev_bufferSize(
        solver.get(), nullptr, kNoVectors, kNoVectors, n,
        kReal, flat.const_data_ptr<float>(), n,
        kReal, eigenvalues.data_ptr<float>(),
        kReal, nullptr, 1, kReal, nullptr, 1, kReal,
        &device_bytes, &host_bytes), "cusolverDnXgeev_bufferSize");
    TORCH_CHECK(device_bytes <= size_t(std::numeric_limits<int64_t>::max()) &&
                host_bytes <= size_t(std::numeric_limits<int64_t>::max()),
                "cuSOLVER workspace size exceeds tensor capacity");

    auto device_work = at::empty({static_cast<int64_t>(device_bytes)}, matrices.options().dtype(at::kByte));
    auto host_work = at::empty({static_cast<int64_t>(host_bytes)},
        at::TensorOptions().device(at::kCPU).dtype(at::kByte).pinned_memory(true));

    for (int64_t start = 0; start < count; start += kSolverBatch) {
        const int64_t batch = std::min(kSolverBatch, count - start);

        // cuSOLVER overwrites its column-major input; transpose into separate storage.
        auto work = at::empty({batch, n, n}, matrices.options());
        work.copy_(flat.narrow(0, start, batch).transpose(-2, -1));

        // Each eigenvalue row stores real parts followed by imaginary parts.
        for (int64_t i = 0; i < batch; ++i) {
            check_solver(cusolverDnXgeev(
                solver.get(), nullptr, kNoVectors, kNoVectors, n,
                kReal, work.data_ptr<float>() + i * n * n, n,
                kReal, eigenvalues.data_ptr<float>() + i * 2 * n,
                kReal, nullptr, 1, kReal, nullptr, 1, kReal,
                device_work.data_ptr(), device_bytes,
                host_work.data_ptr(), host_bytes, info.data_ptr<int>() + i),
                "cusolverDnXgeev");
        }

        // Only solver status is copied to the CPU; eigenvalues and EI stay on the GPU.
        const auto host_info = info.narrow(0, 0, batch).cpu();
        const int* status = host_info.const_data_ptr<int>();
        // Each eigenvalue row stores real parts followed by imaginary parts.
        for (int64_t i = 0; i < batch; ++i) {
            TORCH_CHECK(status[i] >= 0, "cuSOLVER rejected argument ", -status[i],
                        " for matrix ", start + i);
            TORCH_CHECK(status[i] == 0, "cuSOLVER eigenvalue solver did not converge for matrix ",
                        start + i, " (info=", status[i], ")");
        }
        launch_ei_reduction(eigenvalues.const_data_ptr<float>(),
                            ei.data_ptr<float>() + start, batch, n, stream.stream());
    }
    TORCH_CHECK(at::isfinite(ei).all().item<bool>(),
                "cuSOLVER produced NaN or Inf eigenvalues/EI");
    return ei;
}

}
