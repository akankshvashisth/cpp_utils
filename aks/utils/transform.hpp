#pragma once

#include <span>

namespace aks {

template <typename T>
auto transform(auto const f, std::span<T> out, auto const... inputs) {
  for (i = 0; i < size; ++i) {
    out[i] = f((inputs[i])...);
  }
}

}  // namespace aks
