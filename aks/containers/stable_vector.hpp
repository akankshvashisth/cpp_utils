#pragma once
#include <limits>
#include <ostream>
#include <vector>

namespace aks {

template <typename T> struct stable_vector {
  using value_type = T;

  constexpr static std::size_t sentinel =
      std::numeric_limits<std::size_t>::max();

  std::vector<T> _data;
  std::vector<std::size_t> _index;
  std::vector<std::size_t> _actual_index;

  stable_vector() = default;
  stable_vector(stable_vector &&) = default;
  stable_vector &operator=(stable_vector &&) = default;
  stable_vector(stable_vector const &) = default;
  stable_vector &operator=(stable_vector const &) = default;

  std::size_t push_back(T const &t) {
    std::size_t const sz = _data.size();
    std::size_t const asz = _actual_index.size();
    _data.push_back(t);
    _index.push_back(asz);
    _actual_index.push_back(sz);
    return asz;
  }

  T const &at(std::size_t i) const { return _data[_actual_index[i]]; }

  T &at(std::size_t i) { return _data[_actual_index[i]]; }

  void remove(std::size_t i) {
    std::size_t const idx = _actual_index[i];

    _index[idx] = _index.back();
    _actual_index[_index[idx]] = idx;
    _actual_index[i] = sentinel;

    _index.pop_back();

    _data[idx] = std::move(_data.back());
    _data.pop_back();
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, stable_vector<T> const &v) {
  std::vector<std::size_t> xs;
  for (std::size_t i = 0; i < v._actual_index.size(); ++i) {
    xs.push_back(i);
  }
  os << "stable_vector:\n"
     << "  _data        : " << v._data << "\n"
     << "  _index       : " << v._index << "\n"
     << "  _actual_index: " << v._actual_index << "\n"
     << "        _counts: " << xs << "\n";
  return os;
}

} // namespace aks
