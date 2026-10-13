#pragma once

#include <ATen/ATen.h>

namespace millstability::detail {

// CUDA FP32 [...,n,n] -> CUDA FP32 [...] spectral radii.
// Trim exact zero columns and solve in FP64 with hardware-selected host concurrency.
at::Tensor spectral_radius(const at::Tensor& matrices);

}
