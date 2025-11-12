#ifndef POINT3D_HPP
#define POINT3D_HPP

#include "double4d.hpp"

namespace aks {

struct alignas(32) pt3d {
 private:
  dbl4 _data;
  explicit pt3d(const dbl4 data) : _data(data) {}
  pt3d tidy() {
    _data.w = 0.0;
    return *this;
  }

 public:
  pt3d(double x, double y, double z) : _data(x, y, z, 0.0) {}
  explicit pt3d(double x) : _data(x, x, x, 0.0) {}

  pt3d operator+(const pt3d other) const { return pt3d(_data + other._data); }
  pt3d operator-(const pt3d other) const { return pt3d(_data - other._data); }
  pt3d operator*(const pt3d other) const { return pt3d(_data * other._data); }
  pt3d operator/(pt3d other) const { return pt3d(_data / other._data).tidy(); }

  pt3d operator-() const { return pt3d(-_data); }

  pt3d operator+(double other) const { return (*this) + pt3d(other); }
  pt3d operator-(double other) const { return (*this) - pt3d(other); }
  pt3d operator*(double other) const { return (*this) * pt3d(other); }
  pt3d operator/(double other) const { return (*this) / pt3d(other); }

  pt3d& operator+=(const pt3d other) {
    (*this) = (*this) + other;
    return *this;
  }

  pt3d& operator-=(const pt3d other) {
    (*this) = (*this) - other;
    return *this;
  }

  pt3d& operator*=(const pt3d other) {
    (*this) = (*this) * other;
    return *this;
  }

  pt3d& operator/=(const pt3d other) {
    (*this) = (*this) / other;
    return *this;
  }

  pt3d& operator+=(double other) {
    (*this) += pt3d(other);
    return *this;
  }

  pt3d& operator-=(double other) {
    (*this) -= pt3d(other);
    return *this;
  }

  pt3d& operator*=(double other) {
    (*this) *= pt3d(other);
    return *this;
  }

  pt3d& operator/=(double other) {
    (*this) /= pt3d(other);
    return *this;
  }

  double length_square() const { return _data.square().horizontal_add(); }

  double length() const { return std::sqrt(length_square()); }

  pt3d normalized() const { return *this / length(); }

  double dot(const pt3d other) const {
    return (_data * other._data).horizontal_add();
  }

  pt3d cross(const pt3d v) const {
    return pt3d(_data.y * v._data.z - v._data.y * _data.z,
                _data.z * v._data.x - v._data.z * _data.x,
                _data.x * v._data.y - v._data.x * _data.y);
  }

  double distance(const pt3d other) const { return (*this - other).length(); }

  double distance_square(const pt3d other) const {
    return (*this - other).length_square();
  }

  pt3d projected(const pt3d normal) const {
    return (*this) - (normal * dot(normal));
  }

  pt3d reflected(const pt3d normal) const {
    return (*this) - (normal * 2.0 * dot(normal));
  }

  std::tuple<pt3d, pt3d> tangential() const {
    bool const not_parallel_to_x_axis =
        (std::fabs(_data.y) > 0 || std::fabs(_data.z) > 0);
    pt3d const a =
        (not_parallel_to_x_axis ? pt3d(1.0, 0.0, 0.0) : pt3d(0.0, 1.0, 0.0))
            .cross(*this)
            .normalized();
    pt3d const b = cross(a);
    return std::make_tuple(a, b);
  }

  dbl4 data() const { return _data; }

  pt3d sin() const { return pt3d(_data.sin()); }
  pt3d cos() const { return pt3d(_data.cos()).tidy(); }
  pt3d tan() const { return pt3d(_data.tan()).tidy(); }
};

pt3d operator*(double s, const pt3d p) {
  return p * s;
}

pt3d operator+(double s, const pt3d p) {
  return p + s;
}

pt3d operator-(double s, const pt3d p) {
  return pt3d(s) - p;
}

pt3d operator/(double s, const pt3d p) {
  return pt3d(s) / p;
}

std::ostream& operator<<(std::ostream& os, const pt3d p) {
  os << "pt3d( " << p.data() << " )";
  return os;
}

}  // namespace aks

#endif  // POINT3D_HPP
