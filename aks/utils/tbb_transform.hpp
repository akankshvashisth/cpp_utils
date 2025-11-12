
#pragma once

#include <cstddef>
#include <span>
#include <tbb/parallel_for.h>

namespace aks {

template <typename T>
auto tbb_transform(auto const f, std::span<T> out, auto const... inputs) {
  size_t const size = out.size();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, size),
                    [&](tbb::blocked_range<size_t> const &r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        out[i] = f((inputs[i])...);
                      }
                    });
}

} // namespace aks