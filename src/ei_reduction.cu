#include "millstability/ei_reduction.cuh"

#include <c10/cuda/CUDAException.h>
#include <math_constants.h>
#include <cmath>

namespace millstability {
namespace {

constexpr int kThreads = 128;

// One block computes the maximum eigenvalue magnitude for one matrix.
__global__ void eigenvalue_max_modulus_kernel(
    const float* eigenvalues, float* ei, int64_t n) {

    const float* real = eigenvalues + int64_t(blockIdx.x) * 2 * n;
    const float* imag = real + n;
    float maximum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float modulus = hypotf(real[i], imag[i]);

        // fmaxf can discard NaNs, so propagate nonfinite results as infinity.
        maximum = isfinite(modulus) ? fmaxf(maximum, modulus) : CUDART_INF_F;
    }

    __shared__ float partial[kThreads];
    partial[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ei[blockIdx.x] = partial[0];
    }
}

}

void launch_ei_reduction(const float* eigenvalues, float* ei,
                         int64_t count, int64_t n, cudaStream_t stream) {
    if (count == 0) return;
    eigenvalue_max_modulus_kernel<<<count, kThreads, 0, stream>>>(eigenvalues, ei, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
