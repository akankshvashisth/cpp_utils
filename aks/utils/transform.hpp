#pragma once

#include <span>

namespace aks {

template <typename T>
auto transform(auto const f, std::span<T> out, auto const... inputs) {
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = f((inputs[i])...);
  }
}

} // namespace aks
