#pragma once

#include "../simd/double4d.hpp"

#include <span>

namespace aks {

auto simd_transform(auto const f, std::span<double> out, auto const... inputs) {
  size_t const size = out.size();
  size_t const simd_size = dbl4::dim;

  size_t i = 0;
  for (; i + simd_size <= size; i += simd_size) {
    dbl4 result = f(dbl4::loadu(&inputs[i])...);
    result.storeu(&out[i]);
  }
  for (; i < size; ++i) {
    out[i] = f((inputs[i])...);
  }
}

}  // namespace aks
