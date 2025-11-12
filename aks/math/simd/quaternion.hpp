#pragma once

#include "../../simd/double4d.hpp"
#include "../../simd/formatters.hpp"

#include <ostream>
#include <print>

namespace aks {
struct quaternion {
  quaternion(double r, double i, double j, double k) : _data(r, i, j, k) {}
  quaternion() : _data(1., 0., 0., 0.) {}

  double &operator[](int i) { return _data[i]; }
  double operator[](int i) const { return _data[i]; }

  double r() const { return _data[0]; }
  double i() const { return _data[1]; }
  double j() const { return _data[2]; }
  double k() const { return _data[3]; }

  double &r() { return _data[0]; }
  double &i() { return _data[1]; }
  double &j() { return _data[2]; }
  double &k() { return _data[3]; }

  quaternion normalized() const {
    double d = (_data * _data).horizontal_add();
    if (d == 0) {
      return quaternion();
    }
    return quaternion(_data / std::sqrt(d));
  }

  friend std::ostream &operator<<(std::ostream &os, quaternion const &v) {
    os << "quaternion( " << v.r() << ", " << v.i() << ", " << v.j() << ", "
       << v.k() << " )";
    return os;
  }

private:
  quaternion(aks::dbl4 const data) : _data(data) {}
  aks::dbl4 _data;

public:
  quaternion operator*(quaternion o) const {
    double r_ = r() * o.r() - i() * o.i() - j() * o.j() - k() * o.k();
    double i_ = r() * o.i() + i() * o.r() + j() * o.k() - k() * o.j();
    double j_ = r() * o.j() + j() * o.r() + k() * o.i() - i() * o.k();
    double k_ = r() * o.k() + k() * o.r() + i() * o.j() - j() * o.i();
    return quaternion(r_, i_, j_, k_);
  }

  template <typename vec3_type> quaternion operator+(vec3_type o) const {
    quaternion q(0., o.x(), o.y(), o.z());
    q = q * (*this);
    return quaternion(_data + (q._data * 0.5));
  }

  template <typename vec3_type> quaternion rotate_by(vec3_type o) const {
    quaternion q(0., o.x(), o.y(), o.z());
    return (*this) * q;
  }
};
} // namespace aks

template <> struct std::formatter<aks::quaternion, char> {
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
  FmtContext::iterator format(aks::quaternion s, FmtContext &ctx) const {
    std::string out;
    if (!full) {
      std::stringstream ss;
      ss << s;
      out = ss.str();
    } else
      out = std::format("[{}, {}, {}, {}]", s[0], s[1], s[2], s[3]);
    return std::ranges::copy(std::move(out), ctx.out()).out;
  }
};
