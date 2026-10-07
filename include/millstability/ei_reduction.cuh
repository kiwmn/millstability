#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace millstability {

// Input: count rows of n real parts followed by n imaginary parts. Output: count radii.
void launch_ei_reduction(const float* eigenvalues, float* ei,
                         int64_t count, int64_t n, cudaStream_t stream);

}
