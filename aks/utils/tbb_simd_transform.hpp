

#pragma once

#include <cstddef>
#include <span>
#include <tbb/parallel_for.h>

namespace aks {

auto tbb_simd_transform(auto const f, std::span<double> out,
                        auto const... inputs) {
  size_t const size = out.size();
  size_t const simd_size = dbl4::dim;

  int count = static_cast<int>(size / simd_size);

  tbb::parallel_for(tbb::blocked_range<int>(0, count),
                    [&](tbb::blocked_range<int> const &r) {
                      for (int i = r.begin(); i != r.end(); ++i) {
                        dbl4 result = f(dbl4::loadu(&inputs[i * simd_size])...);
                        result.storeu(&out[i * simd_size]);
                      }
                    });

  for (int i = count * simd_size; i < size; ++i) {
    out[i] = f((inputs[i])...);
  }
}

} // namespace aks