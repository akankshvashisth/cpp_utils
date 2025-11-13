#pragma once

#include <span>

namespace aks {
template <typename T> auto iota(std::span<T> &vec, T start) {
  for (size_t i = 0; i < vec.size(); ++i) {
    auto i_ = static_cast<T>(double(i));
    auto add_ = start + i_;
    vec[i] = static_cast<T>(add_);
  }
  return vec;
}
} // namespace aks