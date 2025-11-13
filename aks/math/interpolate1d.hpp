#pragma once
#include "../simd/double4d.hpp"
#include <algorithm>
#include <concepts>
#include <memory_resource>
#include <ranges>
#include <vector>

namespace aks {
using real = double;

auto interplote1d(std::ranges::random_access_range auto const &x,
                  std::ranges::random_access_range auto const &y,
                  std::floating_point auto const xnew,
                  std::pmr::polymorphic_allocator<> alloc = {}) {
  using v_type = std::remove_cvref_t<decltype(x[0])>;
  // assume x is sorted
  size_t n = x.size();
  if (xnew <= x[0]) {
    return y[0];
  }
  if (xnew >= x[n - 1]) {
    return y[n - 1];
  }
  // size_t i = 0;
  // while (i < n - 1 && x[i + 1] < xnew) {
  //   ++i;
  // }

  // Replace linear scan with binary search to find the interval index.
  // Find the first element >= xnew, then use the previous index as the lower
  // bracket to match the original loop behavior.
  auto it = std::lower_bound(x.begin(), x.end(), xnew);
  size_t j = static_cast<size_t>(std::distance(x.begin(), it));
  size_t i = (j == 0) ? 0 : (j - 1);

  v_type x0(x[i]);
  v_type x1(x[i + 1]);
  v_type y0(y[i]);
  v_type y1(y[i + 1]);
  v_type slope = (y1 - y0) / (x1 - x0);
  return y0 + slope * (xnew - x0);
}

auto interplote1d(std::pmr::vector<real> const &x,
                  std::pmr::vector<dbl4> const &y, dbl4 const xnew,
                  std::pmr::polymorphic_allocator<> alloc = {}) {
  std::pmr::vector<std::pmr::vector<real>> y_chunks(
      4, std::pmr::vector<real>(y.size(), 0.0, alloc), alloc);

  auto y_c_0_itr = y_chunks[0].begin();
  auto y_c_1_itr = y_chunks[1].begin();
  auto y_c_2_itr = y_chunks[2].begin();
  auto y_c_3_itr = y_chunks[3].begin();

  for (auto it = y.cbegin(); it != y.cend(); ++it) {
    auto const y_i = *it;
    *(y_c_0_itr++) = it->_0;
    *(y_c_1_itr++) = it->_1;
    *(y_c_2_itr++) = it->_2;
    *(y_c_3_itr++) = it->_3;
  }

  return dbl4(interplote1d(x, y_chunks[0], xnew[0], alloc),
              interplote1d(x, y_chunks[1], xnew[1], alloc),
              interplote1d(x, y_chunks[2], xnew[2], alloc),
              interplote1d(x, y_chunks[3], xnew[3], alloc));
  // assume x is sorted
}
} // namespace aks