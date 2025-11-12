#ifndef ALIGNED_VECTOR_HPP
#define ALIGNED_VECTOR_HPP

#include <ostream>

#include "aligned_allocator.hpp"

namespace aks {

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T, alignof(T)> >;

template <typename T, size_t N>
using aligned_vector_with_alignment_size =
    std::vector<T, aligned_allocator<T, alignof(T) <= N ? N : alignof(T)> >;

template <typename T>
std::ostream& operator<<(std::ostream& os, aligned_vector<T> const& v) {
  for (auto const& e : v) {
    os << e << '\n';
  }
  return os;
}

}  // namespace aks

#endif  // ALIGNED_VECTOR_HPP
