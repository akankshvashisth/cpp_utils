#pragma once

#include <array>
#include <iostream>
#include <optional>
#include <print>
#include <span>
#include <sstream>
#include <vector>

#include "../../utils/simd_transform.hpp"
#include "quaternion.hpp"
#include "vec3.hpp"

namespace aks {

struct matrix_3x3 {
 private:
  //clang-format off
  std::array<double, 9> _data = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
  matrix_3x3 transform(auto f) const {
    matrix_3x3 result;
    simd_transform(f, result._data, _data);
    return result;
  }
  matrix_3x3 transform(auto f, matrix_3x3 const& other) const {
    matrix_3x3 result;
    simd_transform(f, result._data, _data, other._data);
    return result;
  }
  //clang-format on
 public:
  std::span<double const, 9> data() const {
    return std::span<double const, 9>(_data);
  }
  matrix_3x3() = default;

  void set_diagonal(double v0, double v1, double v2) {
    _data[0] = v0;
    _data[4] = v1;
    _data[8] = v2;
  }

  void set_diagonal(double v) { set_diagonal(v, v, v); }

  vec3 get_column(int col) const {
    return vec3(_data[col * 3 + 0], _data[col * 3 + 1], _data[col * 3 + 2]);
  }

  vec3 get_row(int row) const {
    return vec3{_data[row + 0], _data[row + 3], _data[row + 6]};
  }

  matrix_3x3 operator+(matrix_3x3 const& other) const {
    return transform([](auto x, auto y) { return x + y; }, other);
  }
  matrix_3x3 operator-(matrix_3x3 const& other) const {
    return transform([](auto x, auto y) { return x - y; }, other);
  }
  matrix_3x3 operator+(double scalar) const {
    return transform([scalar](auto x) { return x + scalar; });
  }
  matrix_3x3 operator-(double scalar) const {
    return transform([scalar](auto x) { return x - scalar; });
  }

  matrix_3x3 operator*(double scalar) const {
    return transform([scalar](auto x) { return x * scalar; });
  }

  matrix_3x3 operator/(double scalar) const {
    return transform([scalar](auto x) { return x / scalar; });
  }

  auto lerp(matrix_3x3 const& other, double t) const {
    return transform([t](auto a, auto b) { return a * (1. - t) + b * t; },
                     other);
  }

  auto determinant() const {
    return _data[0] * (_data[4] * _data[8] - _data[5] * _data[7]) -
           _data[1] * (_data[3] * _data[8] - _data[5] * _data[6]) +
           _data[2] * (_data[3] * _data[7] - _data[4] * _data[6]);
  }

  matrix_3x3 transpose() const {
    matrix_3x3 r;
    auto& rd = r._data;
    auto const& td = _data;
    rd[0] = td[0];
    rd[1] = td[3];
    rd[2] = td[6];
    rd[3] = td[1];
    rd[4] = td[4];
    rd[5] = td[7];
    rd[6] = td[2];
    rd[7] = td[5];
    rd[8] = td[8];
    return r;
  }

  std::optional<matrix_3x3> inverse() const {
    double det = determinant();
    if (det == 0)
      return std::nullopt;
    det = ((double)1.0) / det;
    matrix_3x3 result;
    auto& rd = result._data;
    auto const& td = _data;
    rd[0] = (td[4] * td[8] - td[5] * td[7]) * det;
    rd[1] = (td[2] * td[7] - td[1] * td[8]) * det;
    rd[2] = (td[1] * td[5] - td[2] * td[4]) * det;
    rd[3] = (td[5] * td[6] - td[3] * td[8]) * det;
    rd[4] = (td[0] * td[8] - td[2] * td[6]) * det;
    rd[5] = (td[2] * td[3] - td[0] * td[5]) * det;
    rd[6] = (td[3] * td[7] - td[4] * td[6]) * det;
    rd[7] = (td[1] * td[6] - td[0] * td[7]) * det;
    rd[8] = (td[0] * td[4] - td[1] * td[3]) * det;
    return result;
  }

  vec3 operator*(vec3 const vector) const {
    auto const x = vector.x();
    auto const y = vector.y();
    auto const z = vector.z();
    auto const& td = _data;
    return vec3(x * td[0] + y * td[1] + z * td[2],
                x * td[3] + y * td[4] + z * td[5],
                x * td[6] + y * td[7] + z * td[8]);
  }

  vec3 transform(vec3 const vector) const { return (*this) * vector; }

  vec3 transform_transpose(const vec3 vector) const {
    return vec3(
        vector.x() * _data[0] + vector.y() * _data[3] + vector.z() * _data[6],
        vector.x() * _data[1] + vector.y() * _data[4] + vector.z() * _data[7],
        vector.x() * _data[2] + vector.y() * _data[5] + vector.z() * _data[8]);
  }

  vec3 transform_inverse_direction(vec3 const vector) const {
    auto const x = vector.x();
    auto const y = vector.y();
    auto const z = vector.z();
    auto const& td = _data;
    return vec3(x * td[0] + y * td[3] + z * td[6],
                x * td[1] + y * td[4] + z * td[7],
                x * td[2] + y * td[5] + z * td[8]);
  }

  vec3 transform_inverse(vec3 vector) const {
    auto& x = vector.x();
    auto& y = vector.y();
    auto& z = vector.z();
    return vec3(x * _data[0] + y * _data[3] + z * _data[6],
                x * _data[1] + y * _data[4] + z * _data[7],
                x * _data[2] + y * _data[5] + z * _data[8]);
  }

  vec3 axis_vector(int i) const {
    return vec3(_data[i], _data[i + 3], _data[i + 6]);
  }

  void set_axis_vector(int i, vec3 const& v) {
    _data[i] = v.x();
    _data[i + 3] = v.y();
    _data[i + 6] = v.z();
  }

  void set_column(int col, vec3 const& v) {
    _data[col * 3 + 0] = v.x();
    _data[col * 3 + 1] = v.y();
    _data[col * 3 + 2] = v.z();
  }

  void set_row(int row, vec3 const r) {
    _data[row + 0] = r[0];
    _data[row + 3] = r[1];
    _data[row + 6] = r[2];
  }

  void set_orientation(quaternion const q) {
    auto i = q.i();
    auto j = q.j();
    auto k = q.k();
    auto r = q.r();
    _data[0] = 1 - (2 * j * j + 2 * k * k);
    _data[1] = 2 * i * j + 2 * k * r;
    _data[2] = 2 * i * k - 2 * j * r;
    _data[3] = 2 * i * j - 2 * k * r;
    _data[4] = 1 - (2 * i * i + 2 * k * k);
    _data[5] = 2 * j * k + 2 * i * r;
    _data[6] = 2 * i * k + 2 * j * r;
    _data[7] = 2 * j * k - 2 * i * r;
    _data[8] = 1 - (2 * i * i + 2 * j * j);
  }

  void set_inertia_tensor_coefficients(double ix,
                                       double iy,
                                       double iz,
                                       double ixy = 0,
                                       double ixz = 0,
                                       double iyz = 0) {
    _data[0] = ix;
    _data[1] = _data[3] = -ixy;
    _data[2] = _data[6] = -ixz;
    _data[4] = iy;
    _data[5] = _data[7] = -iyz;
    _data[8] = iz;
  }

  void set_block_inertia_tensor(const vec3 halfSizes, double mass) {
    vec3 squares = halfSizes.component_product(halfSizes);
    set_inertia_tensor_coefficients(0.3f * mass * (squares.y() + squares.z()),
                                    0.3f * mass * (squares.x() + squares.z()),
                                    0.3f * mass * (squares.x() + squares.y()));
  }

  void set_skew_symmetric(const vec3 vector) {
    set_diagonal(0.);
    _data[1] = -vector.z();
    _data[2] = vector.y();
    _data[3] = vector.z();
    _data[5] = -vector.x();
    _data[6] = -vector.y();
    _data[7] = vector.x();
  }

  void setComponents(const vec3 compOne,
                     const vec3 compTwo,
                     const vec3 compThree) {
    _data[0] = compOne.x();
    _data[1] = compTwo.x();
    _data[2] = compThree.x();
    _data[3] = compOne.y();
    _data[4] = compTwo.y();
    _data[5] = compThree.y();
    _data[6] = compOne.z();
    _data[7] = compTwo.z();
    _data[8] = compThree.z();
  }

  friend std::ostream& operator<<(std::ostream& os, matrix_3x3 const& m) {
    os << "matrix_3x3(\n";
    for (int r = 0; r < 3; ++r) {
      os << "  ";
      os << m._data[r * 3 + 0] << ", " << m._data[r * 3 + 1] << ", "
         << m._data[r * 3 + 2];
      if (r < 2) {
        os << ",";
      }
      os << "\n";
    }
    os << ")";
    return os;
  }
};

}  // namespace aks
