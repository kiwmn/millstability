#pragma once

#include <ATen/ATen.h>

namespace millstability::detail {

// Return CUDA float32 radii of [..., n, n] matrices without modifying them.
at::Tensor spectral_radius(const at::Tensor& matrices);

}
