#pragma once

#include <cstddef>
#include <omp.h>
#include <span>

namespace aks {

template <typename T>
auto omp_transform(auto const f, std::span<T> out, auto const... inputs) {
  size_t const size = out.size();
#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    out[i] = f((inputs[i])...);
  }
}

} // namespace aks