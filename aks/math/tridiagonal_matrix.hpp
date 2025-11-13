
#pragma once

#include <iomanip>
#include <iostream>
#include <memory_resource>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace aks {
template <typename T> struct tridiagonalmatrix {
  using v_type = T;
  std::pmr::vector<v_type> lower; // sub-diagonal (a_ij where i = j+1)
  std::pmr::vector<v_type> diag;  // main diagonal (a_ij where i = j)
  std::pmr::vector<v_type> upper; // super-diagonal (a_ij where i = j-1)

  mutable std::pmr::vector<v_type> c_prime;
  mutable std::pmr::vector<v_type> d_prime;

  tridiagonalmatrix(size_t size, std::pmr::polymorphic_allocator<> mr = {})
      : lower(size - 1, v_type(0.0), mr), diag(size, v_type(0.0), mr),
        upper(size - 1, v_type(0.0), mr), c_prime(size - 1, v_type(0.0), mr),
        d_prime(size, v_type(0.0), mr) {}

  size_t size() const { return diag.size(); }

  auto &operator()(size_t const row, size_t const col) {
    if (row == col)
      return diag[row];
    else if (row == col + 1)
      return lower[col];
    else if (row + 1 == col)
      return upper[row];
    else
      throw std::out_of_range("Index out of range for tridiagonal matrix");
  }

  auto const &operator()(size_t const row, size_t const col) const {
    if (row == col)
      return diag[row];
    else if (row == col + 1)
      return lower[col];
    else if (row + 1 == col)
      return upper[row];
    else
      throw std::out_of_range("Index out of range for tridiagonal matrix");
  }

  void solve(std::span<v_type const> rhs, std::span<v_type> solution) const {
    size_t n = size();
    if (rhs.size() != n) {
      throw std::invalid_argument(
          "RHS vector size does not match matrix dimensions");
    }

    if (solution.size() != n) {
      throw std::invalid_argument(
          "Solution vector size does not match matrix dimensions");
    }

    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for (size_t i = 1; i < n - 1; ++i) {
      auto m = diag[i] - lower[i - 1] * c_prime[i - 1];
      c_prime[i] = upper[i] / m;
      d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / m;
    }
    d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) /
                     (diag[n - 1] - lower[n - 2] * c_prime[n - 2]);

    solution[n - 1] = d_prime[n - 1];
    for (size_t i = n - 2; i < n; --i) {
      solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, tridiagonalmatrix<T> const &mat) {
  size_t n = mat.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i == j)
        os << std::setw(8) << std::setprecision(5) << mat.diag[i] << " ";
      else if (i == j + 1)
        os << std::setw(8) << std::setprecision(5) << mat.lower[j] << " ";
      else if (i + 1 == j)
        os << std::setw(8) << std::setprecision(5) << mat.upper[i] << " ";
      else
        os << std::setw(8) << std::setprecision(5) << 0.0 << " ";
    }
    os << "\n";
  }
  return os;
}

template <typename T>
auto matmul(tridiagonalmatrix<T> const &A,
            std::ranges::random_access_range auto const &x,
            std::span<T> result) {
  using v_type = std::remove_cvref_t<decltype(x[0] + A(0, 0))>;

  size_t const n = A.diag.size();
  // std::pmr::vector<v_type> result(n, alloc);

  if (result.size() != n) {
    throw std::invalid_argument(
        "Result vector size does not match matrix dimensions");
  }

  for (size_t i = 0; i < n; ++i) {
    v_type sum = A.diag[i] * x[i];
    if (i > 0) {
      sum += A.lower[i - 1] * x[i - 1];
    }
    if (i < n - 1) {
      sum += A.upper[i] * x[i + 1];
    }
    result[i] = sum;
  }
  return result;
}

} // namespace aks