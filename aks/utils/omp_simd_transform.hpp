#pragma once

#include "../simd/double4d.hpp"
#include <cstddef>
#include <omp.h>
#include <span>

namespace aks {

auto omp_simd_transform(auto const f, std::span<double> out,
                        auto const... inputs) {
  size_t const size = out.size();
  size_t const simd_size = dbl4::dim;

  int count = static_cast<int>(size / simd_size);

#pragma omp parallel for
  for (int i = 0; i < count; ++i) {
    dbl4 result = f(dbl4::loadu(&inputs[i * simd_size])...);
    result.storeu(&out[i * simd_size]);
  }

  for (int i = count * simd_size; i < size; ++i) {
    out[i] = f((inputs[i])...);
  }
}

} // namespace aks