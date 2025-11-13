#pragma once
#include <memory_resource>
#include <vector>

namespace aks {
auto linspace(auto start, auto end, size_t num, bool endpoint = true,
              std::pmr::polymorphic_allocator<> alloc = {}) {
  using real_t = decltype(start);
  std::pmr::vector<real_t> ret(num, alloc);
  real_t step = (end - start) / (num - (endpoint ? 1 : 0));
  for (size_t i = 0; i < num; ++i) {
    ret[i] = start + step * i;
  }
  return ret;
}
} // namespace aks