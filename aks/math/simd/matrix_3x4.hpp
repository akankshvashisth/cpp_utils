
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

struct matrix_3x4 {
 private:
  //clang-format off
  std::array<double, 12> _data = {1., 0., 0., 0., 0., 1.,
                                  0., 0., 0., 0., 1., 0.};

  matrix_3x4 transform(auto f) const {
    matrix_3x4 result;
    simd_transform(f, result._data, _data);
    return result;
  }

  matrix_3x4 transform(auto f, matrix_3x4 const& other) const {
    matrix_3x4 result;
    simd_transform(f, result._data, _data, other._data);
    return result;
  }

  //clang-format on
 public:
  std::span<double const, 12> data() const {
    return std::span<double const, 12>(_data);
  }

  matrix_3x4() = default;

  matrix_3x4(quaternion const q, vec3 const pos) {
    set_orientation_and_position(q, pos);
  }

  void set_diagonal(double v0, double v1, double v2) {
    _data[0] = v0;
    _data[5] = v1;
    _data[10] = v2;
  }
  void set_diagonal(double v) { set_diagonal(v, v, v); }

  vec3 get_column(int col) const {
    return vec3(_data[col * 3 + 0], _data[col * 3 + 1], _data[col * 3 + 2]);
  }

  std::array<double, 4> get_row(int row) const {
    return {_data[row + 0], _data[row + 3], _data[row + 6], _data[row + 9]};
  }

  auto operator+(matrix_3x4 const& other) const {
    return transform([](auto x, auto y) { return x + y; }, other);
  }

  auto operator-(matrix_3x4 const& other) const {
    return transform([](auto x, auto y) { return x - y; }, other);
  }

  auto operator+(double scalar) const {
    return transform([scalar](auto x) { return x + scalar; });
  }

  auto operator-(double scalar) const {
    return transform([scalar](auto x) { return x - scalar; });
  }

  auto operator*(double scalar) const {
    return transform([scalar](auto x) { return x * scalar; });
  }

  auto operator/(double scalar) const {
    return transform([scalar](auto x) { return x / scalar; });
  }

  auto lerp(matrix_3x4 const& other, double t) const {
    return transform([t](auto a, auto b) { return a * (1. - t) + b * t; },
                     other);
  }

  matrix_3x4 operator*(matrix_3x4 const& o) const {
    matrix_3x4 r;
    auto& rd = r._data;
    auto const& od = o._data;
    auto const& td = _data;
    rd[0] = (od[0] * td[0]) + (od[4] * td[1]) + (od[8] * td[2]);
    rd[4] = (od[0] * td[4]) + (od[4] * td[5]) + (od[8] * td[6]);
    rd[8] = (od[0] * td[8]) + (od[4] * td[9]) + (od[8] * td[10]);

    rd[1] = (od[1] * td[0]) + (od[5] * td[1]) + (od[9] * td[2]);
    rd[5] = (od[1] * td[4]) + (od[5] * td[5]) + (od[9] * td[6]);
    rd[9] = (od[1] * td[8]) + (od[5] * td[9]) + (od[9] * td[10]);

    rd[2] = (od[2] * td[0]) + (od[6] * td[1]) + (od[10] * td[2]);
    rd[6] = (od[2] * td[4]) + (od[6] * td[5]) + (od[10] * td[6]);
    rd[10] = (od[2] * td[8]) + (od[6] * td[9]) + (od[10] * td[10]);

    rd[3] = (od[3] * td[0]) + (od[7] * td[1]) + (od[11] * td[2]) + td[3];
    rd[7] = (od[3] * td[4]) + (od[7] * td[5]) + (od[11] * td[6]) + td[7];
    rd[11] = (od[3] * td[8]) + (od[7] * td[9]) + (od[11] * td[10]) + td[11];
    return r;
  }

  matrix_3x4& operator*=(matrix_3x4 const& other) {
    *this = (*this) * other;
    return *this;
  }

  vec3 operator*(vec3 const vector) const {
    auto const x = vector.x();
    auto const y = vector.y();
    auto const z = vector.z();
    auto const& td = _data;

    return vec3(x * td[0] + y * td[1] + z * td[2] + td[3],
                x * td[4] + y * td[5] + z * td[6] + td[7],
                x * td[8] + y * td[9] + z * td[10] + td[11]);
  }

  double determinant() const {
    return _data[8] * _data[5] * _data[2] + _data[4] * _data[9] * _data[2] +
           _data[8] * _data[1] * _data[6] - _data[0] * _data[9] * _data[6] -
           _data[4] * _data[1] * _data[10] + _data[0] * _data[5] * _data[10];
  }

  matrix_3x4 transpose() const {
    matrix_3x4 r;
    auto& rd = r._data;
    auto const& td = _data;
    rd[0] = td[0];
    rd[1] = td[3];
    rd[2] = td[6];
    rd[3] = td[9];
    rd[4] = td[1];
    rd[5] = td[4];
    rd[6] = td[7];
    rd[7] = td[10];
    rd[8] = td[2];
    rd[9] = td[5];
    rd[10] = td[8];
    rd[11] = td[11];
    return r;
  }

  std::optional<matrix_3x4> inverse() const {
    double det = determinant();
    if (det == 0)
      return std::nullopt;

    det = ((double)1.0) / det;

    matrix_3x4 result;
    auto& rd = result._data;
    auto const& td = _data;

    rd[0] = (-td[9] * td[6] + td[5] * td[10]) * det;
    rd[4] = (td[8] * td[6] - td[4] * td[10]) * det;
    rd[8] = (-td[8] * td[5] + td[4] * td[9] * td[15]) * det;

    rd[1] = (td[9] * td[2] - td[1] * td[10]) * det;
    rd[5] = (-td[8] * td[2] + td[0] * td[10]) * det;
    rd[9] = (td[8] * td[1] - td[0] * td[9] * td[15]) * det;

    rd[2] = (-td[5] * td[2] + td[1] * td[6] * td[15]) * det;
    rd[6] = (+td[4] * td[2] - td[0] * td[6] * td[15]) * det;
    rd[10] = (-td[4] * td[1] + td[0] * td[5] * td[15]) * det;

    rd[3] = (td[9] * td[6] * td[3] - td[5] * td[10] * td[3] -
             td[9] * td[2] * td[7] + td[1] * td[10] * td[7] +
             td[5] * td[2] * td[11] - td[1] * td[6] * td[11]) *
            det;
    rd[7] = (-td[8] * td[6] * td[3] + td[4] * td[10] * td[3] +
             td[8] * td[2] * td[7] - td[0] * td[10] * td[7] -
             td[4] * td[2] * td[11] + td[0] * td[6] * td[11]) *
            det;
    rd[11] = (td[8] * td[5] * td[3] - td[4] * td[9] * td[3] -
              td[8] * td[1] * td[7] + td[0] * td[9] * td[7] +
              td[4] * td[1] * td[11] - td[0] * td[5] * td[11]) *
             det;

    return result;
  }

  vec3 transform(vec3 const vector) const {
    auto const x = vector.x();
    auto const y = vector.y();
    auto const z = vector.z();
    auto const& td = _data;

    return vec3(x * td[0] + y * td[1] + z * td[2],
                x * td[4] + y * td[5] + z * td[6],
                x * td[8] + y * td[9] + z * td[10]);
  }

  vec3 transform_inverse_direction(vec3 const vector) const {
    auto const x = vector.x();
    auto const y = vector.y();
    auto const z = vector.z();
    auto const& td = _data;

    return vec3(x * td[0] + y * td[4] + z * td[8],
                x * td[1] + y * td[5] + z * td[9],
                x * td[2] + y * td[6] + z * td[10]);
  }

  vec3 transform_inverse(vec3 vector) const {
    auto& x = vector.x();
    auto& y = vector.y();
    auto& z = vector.z();

    x -= _data[3];
    y -= _data[7];
    z -= _data[11];
    return vec3(x * _data[0] + y * _data[4] + z * _data[8],
                x * _data[1] + y * _data[5] + z * _data[9],
                x * _data[2] + y * _data[6] + z * _data[10]);
  }

  vec3 axis_vector(int i) const {
    return vec3(_data[i], _data[i + 4], _data[i + 8]);
  }

  friend std::ostream& operator<<(std::ostream& os, matrix_3x4 const& m) {
    os << "matrix_3x4(\n";
    for (int r = 0; r < 4; ++r) {
      os << "  ";
      auto row = m.get_row(r);
      os << row[0] << ", " << row[1] << ", " << row[2] << ", " << row[3];
      if (r < 3) {
        os << ",";
      }
      os << "\n";
    }
    os << ")";
    return os;
  }

 private:
  void set_orientation_and_position(quaternion const q, vec3 const pos) {
    auto i = q.i();
    auto j = q.j();
    auto k = q.k();
    auto r = q.r();
    auto x = pos.x();
    auto y = pos.y();
    auto z = pos.z();

    _data[0] = 1 - (2 * j * j + 2 * k * k);
    _data[1] = 2 * i * j + 2 * k * r;
    _data[2] = 2 * i * k - 2 * j * r;
    _data[3] = x;

    _data[4] = 2 * i * j - 2 * k * r;
    _data[5] = 1 - (2 * i * i + 2 * k * k);
    _data[6] = 2 * j * k + 2 * i * r;
    _data[7] = y;

    _data[8] = 2 * i * k + 2 * j * r;
    _data[9] = 2 * j * k - 2 * i * r;
    _data[10] = 1 - (2 * i * i + 2 * j * j);
    _data[11] = z;
  }
};

void fill_gl_array(matrix_3x4 const& m, float* out_array) {
  auto const& data = m.data();
  out_array[0] = static_cast<float>(data[0]);
  out_array[1] = static_cast<float>(data[4]);
  out_array[2] = static_cast<float>(data[8]);
  out_array[3] = 0.0f;
  out_array[4] = static_cast<float>(data[1]);
  out_array[5] = static_cast<float>(data[5]);
  out_array[6] = static_cast<float>(data[9]);
  out_array[7] = 0.0f;
  out_array[8] = static_cast<float>(data[2]);
  out_array[9] = static_cast<float>(data[6]);
  out_array[10] = static_cast<float>(data[10]);
  out_array[11] = 0.0f;
  out_array[12] = static_cast<float>(data[3]);
  out_array[13] = static_cast<float>(data[7]);
  out_array[14] = static_cast<float>(data[11]);
  out_array[15] = 1.0f;
}

}  // namespace aks
