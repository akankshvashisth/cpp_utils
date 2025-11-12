#pragma once

#include "../../simd/double4d.hpp"
#include "../../simd/formatters.hpp"

#include <ostream>
#include <print>

namespace aks {
struct vec3 {
  vec3(double x, double y, double z) : _data(x, y, z, 0.0) {}
  vec3() : _data(0.0, 0.0, 0.0, 0.0) {}
  explicit vec3(double v) : _data(v, v, v, 0.0) {}
  vec3(vec3 const &other) = default;
  vec3 &operator=(vec3 const &other) = default;
  vec3(vec3 &&other) = default;
  vec3 &operator=(vec3 &&other) = default;

  double &operator[](int i) { return _data[i]; }
  double operator[](int i) const { return _data[i]; }

  double &x() { return _data.x; }
  double x() const { return _data.x; }
  double &y() { return _data.y; }
  double y() const { return _data.y; }
  double &z() { return _data.z; }
  double z() const { return _data.z; }

  vec3 operator+(vec3 const &other) const { return vec3{_data + other._data}; }
  vec3 operator-(vec3 const &other) const { return vec3{_data - other._data}; }
  vec3 operator*(double scalar) const { return vec3{_data * scalar}; }
  vec3 operator/(double scalar) const { return vec3{_data / scalar}; }

  friend std::ostream &operator<<(std::ostream &os, vec3 const &v) {
    os << "vec3( " << v.x() << ", " << v.y() << ", " << v.z() << " )";
    return os;
  }

  vec3 normalized() const {
    double mag = magnitude();
    if (mag == 0.0) {
      return vec3(0.0, 0.0, 0.0);
    }
    return vec3(_data / mag);
  }

  vec3 trim(double new_magnitude) const {
    double mag_sq = magnitude_squared();
    if (mag_sq <= new_magnitude * new_magnitude) {
      return *this;
    }
    return normalized() * new_magnitude;
  }

  vec3 cross(vec3 const &o) const {
    vec3 lhs(_data.y * o._data.z, _data.z * o._data.x, _data.x * o._data.y);
    vec3 rhs(_data.z * o._data.y, _data.x * o._data.z, _data.y * o._data.x);
    return lhs - rhs;
  }

  double dot(vec3 const &o) const { return (_data * o._data).horizontal_add(); }
  double magnitude() const { return std::sqrt(magnitude_squared()); }
  double magnitude_squared() const { return (_data * _data).horizontal_add(); }

private:
  explicit vec3(aks::dbl4 const data) : _data(data) {}
  aks::dbl4 _data;
};

} // namespace aks

template <> struct std::formatter<aks::vec3, char> {
  bool full = false;

  template <class ParseContext>
  constexpr ParseContext::iterator parse(ParseContext &ctx) {
    auto it = ctx.begin();
    if (it == ctx.end())
      return it;

    if (*it == '#') {
      full = true;
      ++it;
    }
    if (it != ctx.end() && *it != '}')
      throw std::format_error("Invalid format args for dbl4.");

    return it;
  }

  template <class FmtContext>
  FmtContext::iterator format(aks::vec3 s, FmtContext &ctx) const {
    std::string out;
    if (!full) {
      std::stringstream ss;
      ss << s;
      out = ss.str();
    } else
      out = std::format("[{}, {}, {}]", s[0], s[1], s[2]);

    return std::ranges::copy(std::move(out), ctx.out()).out;
  }
};
