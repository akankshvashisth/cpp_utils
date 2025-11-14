#pragma once

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory_resource>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

namespace aks {
using real = double;

template <typename T> struct grid_2d {
  using v_type = T;

  grid_2d(size_t const rows, size_t const cols,
          std::pmr::polymorphic_allocator<> alloc = {})
      : rows_(rows), cols_(cols), data_(rows * cols, T{}, alloc) {}

  grid_2d(size_t const rows, size_t const cols, v_type const &initial_value,
          std::pmr::polymorphic_allocator<> alloc = {})
      : rows_(rows), cols_(cols), data_(rows * cols, initial_value, alloc) {}

  grid_2d(grid_2d const &other, std::pmr::polymorphic_allocator<> alloc = {})
      : rows_(other.rows_), cols_(other.cols_),
        data_(other.data_.begin(), other.data_.end(), alloc) {}

  grid_2d(grid_2d &&other) noexcept
      : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)) {}

  grid_2d &operator=(grid_2d const &other) {
    if (this != &other) {
      rows_ = other.rows_;
      cols_ = other.cols_;
      data_ = std::pmr::vector<T>(other.data_.begin(), other.data_.end(),
                                  data_.get_allocator().resource());
    }
    return *this;
  }

  auto get_allocator() const { return data_.get_allocator(); }

  grid_2d &operator=(grid_2d &&other) noexcept {
    if (this != &other) {
      rows_ = other.rows_;
      cols_ = other.cols_;
      data_ = std::move(other.data_);
    }
    return *this;
  }

  grid_2d &fill(T const &value) {
    std::fill(data_.begin(), data_.end(), value);
    return *this;
  }

  T &at(size_t const row, size_t const col) {
    return data_.at(toIdx(row, col));
  }

  T const &at(size_t const row, size_t const col) const {
    return data_.at(toIdx(row, col));
  }

  T &operator()(size_t const row, size_t const col) {
    return data_[toIdx(row, col)];
  }

  T const &operator()(size_t const row, size_t const col) const {
    return data_[toIdx(row, col)];
  }

  template <size_t N> size_t size() const {
    if constexpr (N == 0)
      return rows_;
    else if constexpr (N == 1)
      return cols_;
    else
      throw std::out_of_range("grid_2d only has 2 dimensions (0 and 1)");
  }

  grid_2d &fill_column(size_t const column, T const &value) {
    for (size_t row = 0; row < rows_; ++row) {
      (*this)(row, column) = value;
    }
    return *this;
  }

  grid_2d &fill_row(size_t const row, T const &value) {
    for (size_t col = 0; col < cols_; ++col) {
      (*this)(row, col) = value;
    }
    return *this;
  }

  std::span<T> row_span(size_t const row) {
    return std::span<T>(&data_[toIdx(row, 0)], cols_);
  }

  std::span<T const> row_span(size_t const row) const {
    return std::span<T const>(&data_[toIdx(row, 0)], cols_);
  }

  std::span<T> first_row_span() {
    return std::span<T>(&data_[toIdx(0, 0)], cols_);
  }

  std::span<T const> first_row_span() const {
    return std::span<T const>(&data_[toIdx(0, 0)], cols_);
  }

  std::span<T> last_row_span() {
    return std::span<T>(&data_[toIdx(rows_ - 1, 0)], cols_);
  }

  std::span<T const> last_row_span() const {
    return std::span<T const>(&data_[toIdx(rows_ - 1, 0)], cols_);
  }

  auto col_view(size_t const col) {
    return data_ | std::views::drop(col) | std::views::stride(cols_);
  }

  auto col_view(size_t const col) const {
    return data_ | std::views::drop(col) | std::views::stride(cols_);
  }

  auto first_col_view() { return col_view(0); }
  auto first_col_view() const { return col_view(0); }

  auto last_col_view() { return col_view(cols_ - 1); }
  auto last_col_view() const { return col_view(cols_ - 1); }

  grid_2d slice(size_t const row_start, size_t const row_end,
                size_t const col_start, size_t const col_end) const {
    size_t new_rows = row_end - row_start;
    size_t new_cols = col_end - col_start;
    grid_2d<T> result(new_rows, new_cols, data_.get_allocator());
    for (size_t i = 0; i < new_rows; ++i) {
      for (size_t j = 0; j < new_cols; ++j) {
        result(i, j) = (*this)(row_start + i, col_start + j);
      }
    }
    return result;
  }

  grid_2d transpose() const {
    grid_2d<T> result(cols_, rows_, data_.get_allocator());
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result(j, i) = (*this)(i, j);
      }
    }
    return result;
  }

  std::span<T> data_span() { return std::span<T>(data_.data(), data_.size()); }
  std::span<T const> data_span() const {
    return std::span<T const>(data_.data(), data_.size());
  }

private:
  size_t rows_;
  size_t cols_;
  std::pmr::vector<T> data_;

  size_t toIdx(size_t row, size_t col) const { return row * cols_ + col; }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, grid_2d<T> const &grid) {
  for (size_t i = 0; i < grid.template size<0>(); ++i) {
    for (size_t j = 0; j < grid.template size<1>(); ++j) {
      if constexpr (std::is_floating_point_v<T>) {
        os << std::setw(8) << std::setprecision(6) << grid(i, j) << " ";
      } else if constexpr (std::is_same_v<T, uint8_t>) {
        os << std::setw(3) << static_cast<int>(grid(i, j)) << " ";
      } else {
        os << std::setw(6) << grid(i, j) << " ";
      }
    }
    os << "\n";
  }
  return os;
}

template <typename T>
std::ostream &to_csv(std::ostream &os, grid_2d<T> const &grid) {
  os << std::fixed << std::setprecision(10);
  for (size_t i = 0; i < grid.template size<0>(); ++i) {
    for (size_t j = 0; j < grid.template size<1>(); ++j) {
      os << grid(i, j);
      if (j < grid.template size<1>() - 1) {
        os << ",";
      }
    }
    os << "\n";
  }
  return os;
}

} // namespace aks