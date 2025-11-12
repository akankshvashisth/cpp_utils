#ifndef POINT4D_HPP
#define POINT4D_HPP

#include "double4d.hpp"

namespace aks {

struct alignas(32) pt4d {
 private:
  dbl4 _data;

 public:
  pt4d(double x, double y, double z, double w) : _data(x, y, z, w) {}
  explicit pt4d(const dbl4 data) : _data(data) {}

  pt4d operator+(const pt4d other) const { return pt4d(_data + other._data); }
  pt4d operator-(const pt4d other) const { return pt4d(_data - other._data); }
  pt4d operator*(const pt4d other) const { return pt4d(_data * other._data); }
  pt4d operator/(const pt4d other) const { return pt4d(_data / other._data); }

  pt4d operator-() const { return pt4d(-_data); }

  pt4d operator+(double other) const { return pt4d(_data + other); }
  pt4d operator-(double other) const { return pt4d(_data - other); }
  pt4d operator*(double other) const { return pt4d(_data * other); }
  pt4d operator/(double other) const { return pt4d(_data / other); }

  pt4d& operator+=(const pt4d other) {
    _data += other._data;
    return *this;
  }

  pt4d& operator-=(const pt4d other) {
    _data -= other._data;
    return *this;
  }

  pt4d& operator*=(const pt4d other) {
    _data *= other._data;
    return *this;
  }

  pt4d& operator/=(const pt4d other) {
    _data /= other._data;
    return *this;
  }

  pt4d& operator+=(double other) {
    _data += other;
    return *this;
  }

  pt4d& operator-=(double other) {
    _data -= other;
    return *this;
  }

  pt4d& operator*=(double other) {
    _data *= other;
    return *this;
  }

  pt4d& operator/=(double other) {
    _data /= other;
    return *this;
  }

  double length_square() const { return _data.square().horizontal_add(); }

  double length() const { return std::sqrt(length_square()); }

  pt4d normalized() const { return *this / length(); }

  double dot(const pt4d other) const {
    return (this->_data * other._data).horizontal_add();
  }

  double distance(const pt4d other) const { return (*this - other).length(); }

  double distance_square(const pt4d other) const {
    return (*this - other).length_square();
  }

  dbl4 data() const { return _data; }
};

std::ostream& operator<<(std::ostream& os, const pt4d p) {
  os << "pt4d( " << p.data() << " )";
  return os;
}
}  // namespace aks

#endif  // POINT4D_HPP
