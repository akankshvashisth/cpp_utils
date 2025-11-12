// simd_02.cpp :

#ifndef AKS_SIMD_COMMON_HPP__
#define AKS_SIMD_COMMON_HPP__

#include <concepts>
#include <type_traits>

namespace aks {
template <typename T>
concept is_number = std::is_integral_v<T> || std::is_floating_point_v<T>;
namespace simd {

struct zero_t {};

constexpr zero_t const zero_v{};

template <typename simd_specification_type_> struct vec {
  using spec = simd_specification_type_;
  using simd_type = typename spec::simd_type;
  using scalar_type = typename spec::scalar_type;
  constexpr static std::size_t size_ = spec::size;

  typedef union {
    simd_type simd_;
    scalar_type scalar_[size_];
  } data_type;

  data_type data_;

  vec() = default;

  template <typename... Args> vec(Args... args) : data_({args...}) {
    static_assert(sizeof...(Args) == size_, "invalid number of values");
  }

  explicit vec(simd_type v) : data_({v}) {}

  template <typename scalar_type_>
  explicit vec(scalar_type_ v) : data_({spec::set1(v)}) {}

  explicit vec(zero_t) : data_({spec::setzero()}) {}

  operator simd_type() const { return data_.simd_; }

  scalar_type const &at(std::size_t i) const { return data_.scalar_[i]; }
  scalar_type &at(std::size_t i) { return data_.scalar_[i]; }

  // auto begin() { return data_.scalar_; }
  // auto end() { return data_.scalar_ + size_; }
  //
  // scalar_type const* begin() const { return data_.scalar_; }
  // scalar_type const* end() const { return data_.scalar_ + size_; }
  //
  // auto cbegin() const { return data_.scalar_; }
  // auto cend() const { return data_.scalar_ + size_; }

  constexpr auto size() const { return size_; }
};

template <typename simd_specification_type_>
struct is_aks_simd_vec_t : std::false_type {};

template <typename simd_specification_type_>
struct is_aks_simd_vec_t<vec<simd_specification_type_>> : std::true_type {};

template <typename simd_specification_type_>
constexpr bool is_aks_simd_vec_v =
    is_aks_simd_vec_t<simd_specification_type_>::value;

template <typename simd_specification_type_>
concept is_vec_c = is_aks_simd_vec_v<simd_specification_type_>;

template <typename simd_specification_type_> struct mask {
  using spec = simd_specification_type_;
  using mask_type = typename spec::mask_type;
  using scalar_type = typename spec::scalar_type;
  constexpr static std::size_t size_ = spec::size;

  typedef union {
    mask_type mask_;
    scalar_type scalar_[size_];
  } data_type;

  data_type data_;

  mask() = default;

  explicit mask(mask_type v) : data_({v}) {}

  operator mask_type() const { return data_.mask_; }

  constexpr auto size() const { return size_; }
};

template <typename simd_specification_type_>
struct is_aks_simd_mask_t : std::false_type {};

template <typename simd_specification_type_>
struct is_aks_simd_mask_t<mask<simd_specification_type_>> : std::true_type {};

template <typename simd_specification_type_>
constexpr bool is_aks_simd_mask_v =
    is_aks_simd_mask_t<simd_specification_type_>::value;

template <typename simd_specification_type_>
concept is_mask_c = is_aks_simd_mask_v<simd_specification_type_>;

template <typename simd_specification_type_>
concept is_simd_c =
    is_vec_c<simd_specification_type_> || is_mask_c<simd_specification_type_>;

} // namespace simd

template <typename T>
concept is_value = is_number<T> || simd::is_simd_c<T> || std::same_as<T, bool>;

} // namespace aks

#endif // !AKS_COMMON_HPP

#ifndef AKS_SIMD_ONE_VALUE_HPP__
#define AKS_SIMD_ONE_VALUE_HPP__

#include <array>
#include <cmath>
#include <type_traits>

namespace aks {
namespace simd {

template <typename T> struct one_value_spec {
  using simd_type = T;
  using mask_type = bool;
  using scalar_type = T;
  static constexpr std::size_t size = 1;

  static simd_type set1(scalar_type v) { return v; }
  static simd_type setzero() { return scalar_type(0); }

  static simd_type add(simd_type a, simd_type b) { return a + b; }
  static simd_type sub(simd_type a, simd_type b) { return a - b; }
  static simd_type mul(simd_type a, simd_type b) { return a * b; }
  static simd_type div(simd_type a, simd_type b) { return a / b; }

  static simd_type neg(simd_type x) { return -x; }

  static simd_type sqrt(simd_type a) { return std::sqrt(a); }

  static simd_type abs(simd_type a) { return std::abs(a); }

  static mask_type eq(simd_type a, simd_type b) { return a == b; }

  static mask_type neq(simd_type a, simd_type b) { return a != b; }

  static mask_type lt(simd_type a, simd_type b) { return a < b; }

  static mask_type lte(simd_type a, simd_type b) { return a <= b; }

  static mask_type gt(simd_type a, simd_type b) { return a > b; }

  static mask_type gte(simd_type a, simd_type b) { return a >= b; }

  static mask_type and_(mask_type a, mask_type b) { return a && b; }

  static mask_type or_(mask_type a, mask_type b) { return a || b; }

  static mask_type xor_(mask_type a, mask_type b) { return a != b; }

  static mask_type not_(mask_type a) { return !a; }

  static auto simd_as_array(simd_type a) {
    std::array<scalar_type, size> result;
    result[0] = a;
    return result;
  }

  static auto mask_as_array(mask_type a) {
    std::array<bool, size> result;
    result[0] = a;
    return result;
  }

  static simd_type select(mask_type a, simd_type b, simd_type c) {
    return a ? b : c;
  }

  static bool any(mask_type a) { return a; }

  static bool all(mask_type a) { return a; }
};
using vec1f = simd::vec<one_value_spec<float>>;
using vec1f_mask = simd::mask<one_value_spec<float>>;
} // namespace simd
} // namespace aks

#endif // !AKS_SIMD_ONE_VALUE_HPP__

#ifndef AKS_SIMD_VEC4D_HPP__
#define AKS_SIMD_VEC4D_HPP__

#include <array>
#include <immintrin.h>
#include <type_traits>

namespace aks {
namespace simd {
namespace avx {
namespace detail_vec4d {
static auto const zeros{_mm256_setzero_pd()};
static auto const neg_zeros{_mm256_set1_pd(-0.0)};
} // namespace detail_vec4d

struct vec4d_spec {
  using simd_type = __m256d;
  using mask_type = __m256d;
  using scalar_type = double;
  static constexpr std::size_t size = 4;

  static simd_type set1(scalar_type v) { return _mm256_set1_pd(v); }
  static simd_type setzero() { return _mm256_setzero_pd(); }

  static simd_type add(simd_type a, simd_type b) { return _mm256_add_pd(a, b); }
  static simd_type sub(simd_type a, simd_type b) { return _mm256_sub_pd(a, b); }
  static simd_type mul(simd_type a, simd_type b) { return _mm256_mul_pd(a, b); }
  static simd_type div(simd_type a, simd_type b) { return _mm256_div_pd(a, b); }

  static simd_type neg(simd_type x) {
    return _mm256_xor_pd(x, detail_vec4d::neg_zeros);
  }

  static simd_type sqrt(simd_type a) { return _mm256_sqrt_pd(a); }

  static simd_type abs(simd_type a) {
    return _mm256_andnot_pd(detail_vec4d::neg_zeros, a);
  }

  static mask_type eq(simd_type a, simd_type b) {
    return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
  }

  static mask_type neq(simd_type a, simd_type b) {
    return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ);
  }

  static mask_type lt(simd_type a, simd_type b) {
    return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
  }

  static mask_type lte(simd_type a, simd_type b) {
    return _mm256_cmp_pd(a, b, _CMP_LE_OQ);
  }

  static mask_type gt(simd_type a, simd_type b) {
    return _mm256_cmp_pd(a, b, _CMP_GT_OQ);
  }

  static mask_type gte(simd_type a, simd_type b) {
    return _mm256_cmp_pd(a, b, _CMP_GE_OQ);
  }

  static mask_type and_(mask_type a, mask_type b) {
    return _mm256_and_pd(a, b);
  }

  static mask_type or_(mask_type a, mask_type b) { return _mm256_or_pd(a, b); }

  static mask_type xor_(mask_type a, mask_type b) {
    return _mm256_xor_pd(a, b);
  }

  static mask_type not_(mask_type a) {
    return _mm256_xor_pd(a, detail_vec4d::zeros);
  }

  static auto simd_as_array(simd_type a) {
    std::array<scalar_type, size> result;
    _mm256_storeu_pd(result.data(), a);
    return result;
  }

  static auto mask_as_array(mask_type a) {
    std::array<bool, size> result;
    auto mvmsk = _mm256_movemask_pd(a);
    for (int i = 0; i < size; i++) {
      result[i] = mvmsk & (1 << i);
    }
    return result;
  }

  static simd_type select(mask_type a, simd_type b, simd_type c) {
    return _mm256_blendv_pd(c, b, a);
  }

  static bool any(mask_type a) { return _mm256_movemask_pd(a) != 0; }

  static bool all(mask_type a) { return _mm256_movemask_pd(a) == 0xf; }
};
} // namespace avx
using vec4d = simd::vec<avx::vec4d_spec>;
using vec4d_mask = simd::mask<avx::vec4d_spec>;
} // namespace simd
} // namespace aks

#endif // !AKS_SIMD_VEC4D_HPP__

#ifndef AKS_SIMD_COMMON_OPS_HPP
#define AKS_SIMD_COMMON_OPS_HPP

// #include "aks_simd_common.hpp"
#include <type_traits>

namespace aks {
namespace simd {

namespace detail_ops {

template <typename T> struct spec {
  using type = std::remove_cvref_t<T>::spec;
};

template <typename T> using spec_t = typename spec<T>::type;

template <typename F>
inline auto check_apply(is_vec_c auto const a, is_vec_c auto const b, F f) {
  static_assert(std::is_same_v<decltype(a), decltype(b)>, "mismatched types");
  using simd_vec_type = std::remove_cvref_t<decltype(a)>;
  return simd_vec_type{f(a, b)};
}

} // namespace detail_ops

auto operator+(is_vec_c auto const a, is_vec_c auto const b) {
  return detail_ops::check_apply(a, b, detail_ops::spec_t<decltype(a)>::add);
}

auto operator-(is_vec_c auto const a, is_vec_c auto const b) {
  return detail_ops::check_apply(a, b, detail_ops::spec_t<decltype(a)>::sub);
}

auto operator*(is_vec_c auto const a, is_vec_c auto const b) {
  return detail_ops::check_apply(a, b, detail_ops::spec_t<decltype(a)>::mul);
}

auto operator/(is_vec_c auto const a, is_vec_c auto const b) {
  return detail_ops::check_apply(a, b, detail_ops::spec_t<decltype(a)>::div);
}

auto operator-(is_vec_c auto const a) {
  return std::remove_cvref_t<decltype(a)>{
      detail_ops::spec_t<decltype(a)>::neg(a)};
}

auto sqrt(is_vec_c auto const a) {
  return std::remove_cvref_t<decltype(a)>{
      detail_ops::spec_t<decltype(a)>::sqrt(a)};
}

auto abs(is_vec_c auto const a) {
  return std::remove_cvref_t<decltype(a)>{
      detail_ops::spec_t<decltype(a)>::abs(a)};
}

auto as_array(is_vec_c auto const a) {
  return detail_ops::spec_t<decltype(a)>::simd_as_array(a);
}

} // namespace simd
} // namespace aks

#endif // !AKS_SIMD_COMMON_OPS_HPP

#ifndef AKS_SIMD_COMMON_COMPARE_OPS_HPP
#define AKS_SIMD_COMMON_COMPARE_OPS_HPP

// #include "aks_simd_common.hpp"
#include <type_traits>

namespace aks {
namespace simd {

namespace detail_cops {

template <typename T> struct spec {
  using type = std::remove_cvref_t<T>::spec;
};

template <typename T> using spec_t = typename spec<T>::type;

template <typename F>
inline auto check_apply(is_vec_c auto const a, is_vec_c auto const b, F f) {
  static_assert(std::is_same_v<decltype(a), decltype(b)>, "mismatched types");
  using mask_type = mask<spec_t<decltype(a)>>;
  return mask_type{f(a, b)};
}

} // namespace detail_cops

auto operator<(is_vec_c auto const a, is_vec_c auto const b) {
  return detail_cops::check_apply(a, b, detail_cops::spec_t<decltype(a)>::lt);
}

auto operator<=(is_vec_c auto const a, is_vec_c auto const b) {
  return detail_cops::check_apply(a, b, detail_cops::spec_t<decltype(a)>::lte);
}

auto operator>(is_vec_c auto const a, is_vec_c auto const b) {
  return detail_cops::check_apply(a, b, detail_cops::spec_t<decltype(a)>::gt);
}

auto operator>=(is_vec_c auto const a, is_vec_c auto const b) {
  return detail_cops::check_apply(a, b, detail_cops::spec_t<decltype(a)>::gte);
}

auto operator==(is_vec_c auto const a, is_vec_c auto const b) {
  return detail_cops::check_apply(a, b, detail_cops::spec_t<decltype(a)>::eq);
}

auto operator!=(is_vec_c auto const a, is_vec_c auto const b) {
  return detail_cops::check_apply(a, b, detail_cops::spec_t<decltype(a)>::neq);
}

} // namespace simd
} // namespace aks

#endif // !AKS_SIMD_COMMON_COMPARE_OPS_HPP

#ifndef AKS_SIMD_COMMON_LOGICAL_OPS_HPP
#define AKS_SIMD_COMMON_LOGICAL_OPS_HPP

// #include "aks_simd_common.hpp"

namespace aks {
namespace simd {

namespace detail_lops {
template <typename T> struct spec {
  using type = std::remove_cvref_t<T>::spec;
};

template <typename T> using spec_t = typename spec<T>::type;

template <typename F>
inline auto check_apply(is_mask_c auto const a, is_mask_c auto const b, F f) {
  static_assert(std::is_same_v<decltype(a), decltype(b)>, "mismatched types");
  using mask_vec_type = std::remove_cvref_t<decltype(a)>;
  return mask_vec_type{f(a, b)};
}

} // namespace detail_lops

auto operator&&(is_mask_c auto const a, is_mask_c auto const b) {
  return detail_lops::check_apply(a, b, detail_lops::spec_t<decltype(a)>::and_);
}

auto operator||(is_mask_c auto const a, is_mask_c auto const b) {
  return detail_lops::check_apply(a, b, detail_lops::spec_t<decltype(a)>::or_);
}

auto operator^(is_mask_c auto const a, is_mask_c auto const b) {
  return detail_lops::check_apply(a, b, detail_lops::spec_t<decltype(a)>::xor_);
}

auto operator!(is_mask_c auto const a) {
  return std::remove_cvref_t<decltype(a)>{
      detail_lops::spec_t<decltype(a)>::not_(a)};
}

auto all(is_mask_c auto const a) {
  return detail_lops::spec_t<decltype(a)>::all(a);
}

auto any(is_mask_c auto const a) {
  return detail_lops::spec_t<decltype(a)>::any(a);
}

auto as_array(is_mask_c auto const a) {
  return detail_lops::spec_t<decltype(a)>::mask_as_array(a);
}

} // namespace simd
} // namespace aks

#endif // !AKS_SIMD_COMMON_LOGICAL_OPS_HPP

#ifndef AKS_SIMD_COMMON_SELECT_OPS_HPP
#define AKS_SIMD_COMMON_SELECT_OPS_HPP

// #include "aks_simd_common.hpp"

namespace aks {
namespace simd {
namespace detail_sops {

template <typename T> struct spec {
  using type = std::remove_cvref_t<T>::spec;
};

template <typename T> using spec_t = typename spec<T>::type;

} // namespace detail_sops

auto select(is_mask_c auto const a, is_vec_c auto const b,
            is_vec_c auto const c) {
  static_assert(std::is_same_v<decltype(b), decltype(c)>, "mismatched types");
  static_assert(std::is_same_v<detail_sops::spec_t<decltype(a)>,
                               detail_sops::spec_t<decltype(b)>>,
                "mismatched types between mask and vec");

  using mask_vec_type = std::remove_cvref_t<decltype(a)>;
  using simd_vec_type = std::remove_cvref_t<decltype(b)>;

  return simd_vec_type{detail_sops::spec_t<decltype(a)>::select(a, b, c)};
}

} // namespace simd
} // namespace aks

#endif // !AKS_SIMD_COMMON_SELECT_OPS_HPP

#ifndef AKS_SIMD_COMMON_FORMAT_OPS_HPP
#define AKS_SIMD_COMMON_FORMAT_OPS_HPP

#define FMT_HEADER_ONLY
#include <print>

template <aks::simd::is_simd_c simd_> struct std::formatter<simd_> {
  using custom_range_t = simd_;

  // Parse format specifiers if needed
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  // Format the CustomRange
  template <typename format_ctx>
  auto format(const custom_range_t &range, format_ctx &ctx) const {
    // Format the range as a list
    std::format_to(ctx.out(), "[");
    auto data = as_array(range);
    std::size_t size = range.size();

    for (std::size_t i = 0; i < size; ++i) {
      if (i > 0) {
        std::format_to(ctx.out(), ", ");
      }
      std::format_to(ctx.out(), "{}", data[i]);
    }

    return std::format_to(ctx.out(), "]");
  }
};

#endif // !AKS_SIMD_COMMON_FORMAT_OPS_HPP

#include <cassert>
#include <span>
#include <tuple>
#include <vector>

namespace aks {
namespace semd {
namespace ops {
template <typename T> struct semd_custom_op {
  using oper_t = T;

  template <typename... Args> auto operator()(Args... args) const {
    return op_(*args...);
  }

  oper_t op_;
};

template <typename T> struct semd_op_constant {
  T data;
  auto operator()(auto) const { return data; }
};

struct semd_op_add {
  auto operator()(auto a, auto b) const { return *a + *b; }
};

struct semd_op_mul {
  auto operator()(auto a, auto b) const { return *a * *b; }
};

struct semd_op_sub {
  auto operator()(auto a, auto b) const { return *a - *b; }
};

struct semd_op_div {
  auto operator()(auto a, auto b) const { return *a / *b; }
};

struct semd_op_neg {
  auto operator()(auto a) const { return -(*a); }
};

struct semd_op_abs {
  auto operator()(auto a) const {
    using std::abs;
    return abs(*a);
  }
};

struct semd_op_lt {
  auto operator()(auto a, auto b) const { return *a < *b; }
};

struct semd_op_gt {
  auto operator()(auto a, auto b) const { return *a > *b; }
};

struct semd_op_le {
  auto operator()(auto a, auto b) const { return *a <= *b; }
};

struct semd_op_ge {
  auto operator()(auto a, auto b) const { return *a >= *b; }
};

struct semd_op_eq {
  auto operator()(auto a, auto b) const { return *a == *b; }
};

struct semd_op_ne {
  auto operator()(auto a, auto b) const { return *a != *b; }
};

struct semd_op_and {
  auto operator()(auto a, auto b) const { return *a && *b; }
};

struct semd_op_or {
  auto operator()(auto a, auto b) const { return *a || *b; }
};

struct semd_op_not {
  auto operator()(auto a) const { return !(*a); }
};

template <typename T> auto select(bool a, T const &b, T const &c) {
  return a ? b : c;
}

struct semd_op_select {
  auto operator()(auto a, auto b, auto c) const { return select(*a, *b, *c); }
};
} // namespace ops

template <typename Op, typename... iters_> struct semd_iterator {
  using iters_t = std::tuple<iters_...>;
  using oper_t = Op;

  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = std::remove_cvref_t<decltype(std::apply(
      std::declval<oper_t>(), std::declval<iters_t>()))>;
  using pointer = value_type *;
  using reference = value_type &;

  oper_t op_;
  iters_t ts_;

  auto operator*() const { return std::apply(op_, ts_); }

  auto operator++() {
    std::apply([&](auto &...vs) { (++vs, ...); }, ts_);
    return *this;
  }

  auto operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  auto operator==(semd_iterator const &other) const { return other.ts_ == ts_; }

  auto operator!=(semd_iterator const &other) const {
    return !(*this == other);
  }

  auto apply() const { return std::apply(op_, ts_); }
};

template <typename Op, typename... iters_>
std::size_t distance(semd_iterator<Op, iters_...> const &a,
                     semd_iterator<Op, iters_...> const &b) {
  using std::distance;
  return distance(std::get<0>(a.ts_), std::get<0>(b.ts_));
}

template <typename Op_, typename... iters_> struct expr {
  using oper_t = Op_;
  using iters_t = std::tuple<iters_...>;
  using iterator = semd_iterator<oper_t, iters_...>;
  using value_type = std::remove_cvref_t<decltype(*(std::declval<iterator>()))>;

  expr(oper_t op, iters_t begin, iters_t end)
      : op_(op), begin_(begin), end_(end) {}

  auto begin() const { return iterator{op_, begin_}; }
  auto end() const { return iterator{op_, end_}; }

  auto size() const {
    using std::distance;
    return distance(std::get<0>(begin_), std::get<0>(end_));
  }

  oper_t op_;
  iters_t begin_;
  iters_t end_;
};

template <typename T> struct is_aks_simd_expr_t : std::false_type {};

template <typename op_, typename... iters_>
struct is_aks_simd_expr_t<expr<op_, iters_...>> : std::true_type {};

template <typename T>
constexpr bool const is_aks_simd_expr_v = is_aks_simd_expr_t<T>::value;

template <typename T>
concept is_expr_c = is_aks_simd_expr_v<T>;

// template <typename T>
// struct semd;

template <typename T>
  requires requires(T t) {
    { t.size() } -> std::convertible_to<std::size_t>;
    { t.begin() } -> std::convertible_to<typename T::iterator>;
    { t.end() } -> std::convertible_to<typename T::iterator>;
  }
struct semd {
  using type = typename T::value_type;
  using iterator = typename T::iterator;

  iterator begin_, end_;
  size_t size_;

  std::size_t size() const { return size_; }

  explicit semd(T &data)
      : begin_(data.begin()), end_(data.end()), size_(data.size()) {}

  explicit semd(T const &data)
      : begin_(data.begin()), end_(data.end()), size_(data.size()) {}

  semd(semd &&other) = default;
  semd(const semd &other) = default;

  semd &operator=(const semd &other) {
    if (this != &other) {
      assert(size_ == other.size_);
      for (auto bt = begin_, bf = other.begin_; bt != end_; ++bt, ++bf) {
        *bt = *bf;
      }
    }
  }

  template <typename expr_> semd &operator=(expr_ expr) {
    assert(size_ == expr.size());

    auto bt = begin_;
    auto bf = expr.begin();
    auto be = expr.end();

    for (; bf != be; ++bt, ++bf) {
      *bt = *bf;
    }

    return *this;
  }

  semd &operator=(semd &&other) = default;

  auto begin() const { return begin_; }
  auto end() const { return end_; }
};

template <typename T> struct is_aks_simd_semd_t : std::false_type {};

template <typename T> struct is_aks_simd_semd_t<semd<T>> : std::true_type {};

template <typename T>
constexpr bool const is_aks_simd_semd_v = is_aks_simd_semd_t<T>::value;

template <typename T>
concept is_aks_simd_semd = is_aks_simd_semd_v<T>;

template <typename T>
concept is_semd_c = is_aks_simd_semd_v<T> || is_aks_simd_expr_v<T>;

namespace detail_semd {
auto begin(is_semd_c auto const a, is_semd_c auto const b,
           is_semd_c auto const c) {
  assert(a.size() == b.size());
  assert(a.size() == c.size());
  return std::make_tuple(a.begin(), b.begin(), c.begin());
}

auto end(is_semd_c auto const a, is_semd_c auto const b,
         is_semd_c auto const c) {
  assert(a.size() == b.size());
  assert(a.size() == c.size());
  return std::make_tuple(a.end(), b.end(), c.end());
}

auto begin(is_semd_c auto const a, is_semd_c auto const b) {
  assert(a.size() == b.size());
  return std::make_tuple(a.begin(), b.begin());
}

auto end(is_semd_c auto const a, is_semd_c auto const b) {
  assert(a.size() == b.size());
  return std::make_tuple(a.end(), b.end());
}

auto begin(is_semd_c auto const a) { return std::make_tuple(a.begin()); }

auto end(is_semd_c auto const a) { return std::make_tuple(a.end()); }

auto cexpr(aks::is_value auto const a, is_semd_c auto const b) {
  return expr{ops::semd_op_constant{a}, detail_semd::begin(b),
              detail_semd::end(b)};
}

template <is_semd_c... Args> auto begin(Args const... args) {
  return std::make_tuple(args.begin()...);
}

template <is_semd_c... Args> auto end(Args const... args) {
  return std::make_tuple(args.end()...);
}

} // namespace detail_semd

auto operator+(is_semd_c auto const a, is_semd_c auto const b) {
  return expr{ops::semd_op_add{}, detail_semd::begin(a, b),
              detail_semd::end(a, b)};
}

auto operator*(is_semd_c auto const a, is_semd_c auto const b) {
  return expr{ops::semd_op_mul{}, detail_semd::begin(a, b),
              detail_semd::end(a, b)};
}

auto operator-(is_semd_c auto const a, is_semd_c auto const b) {
  return expr{ops::semd_op_sub{}, detail_semd::begin(a, b),
              detail_semd::end(a, b)};
}

auto operator/(is_semd_c auto const a, is_semd_c auto const b) {
  return expr{ops::semd_op_div{}, detail_semd::begin(a, b),
              detail_semd::end(a, b)};
}

auto operator-(is_semd_c auto const a) {
  return expr{ops::semd_op_neg{}, detail_semd::begin(a), detail_semd::end(a)};
}

auto operator+(is_semd_c auto const a, aks::is_value auto const b) {
  return a + detail_semd::cexpr(b, a);
}

auto operator-(is_semd_c auto const a, aks::is_value auto const b) {
  return a - detail_semd::cexpr(b, a);
}

auto operator*(is_semd_c auto const a, aks::is_value auto const b) {
  return a * detail_semd::cexpr(b, a);
}

auto operator/(is_semd_c auto const a, aks::is_value auto const b) {
  return a / detail_semd::cexpr(b, a);
}

auto operator+(aks::is_value auto const a, is_semd_c auto const b) {
  return detail_semd::cexpr(a, b) + b;
}

auto operator*(aks::is_value auto const a, is_semd_c auto const b) {
  return detail_semd::cexpr(a, b) * b;
}

auto operator-(aks::is_value auto const a, is_semd_c auto const b) {
  return detail_semd::cexpr(a, b) - b;
}

auto operator/(aks::is_value auto const a, is_semd_c auto const b) {
  return detail_semd::cexpr(a, b) / b;
}

auto abs(is_semd_c auto const a) {
  return expr{ops::semd_op_abs{}, detail_semd::begin(a), detail_semd::end(a)};
}

auto operator<(is_semd_c auto const a, is_semd_c auto const b) {
  return expr{ops::semd_op_lt{}, detail_semd::begin(a, b),
              detail_semd::end(a, b)};
}

auto operator<=(is_semd_c auto const a, is_semd_c auto const b) {
  return expr{ops::semd_op_le{}, detail_semd::begin(a, b),
              detail_semd::end(a, b)};
}

auto operator>(is_semd_c auto const a, is_semd_c auto const b) {
  return expr{ops::semd_op_gt{}, detail_semd::begin(a, b),
              detail_semd::end(a, b)};
}

auto operator>=(is_semd_c auto const a, is_semd_c auto const b) {
  return expr{ops::semd_op_ge{}, detail_semd::begin(a, b),
              detail_semd::end(a, b)};
}

auto operator==(is_semd_c auto const a, is_semd_c auto const b) {
  return expr{ops::semd_op_eq{}, detail_semd::begin(a, b),
              detail_semd::end(a, b)};
}

auto operator!=(is_semd_c auto const a, is_semd_c auto const b) {
  return expr{ops::semd_op_ne{}, detail_semd::begin(a, b),
              detail_semd::end(a, b)};
}

auto operator&&(is_semd_c auto const a, is_semd_c auto const b) {
  return expr{ops::semd_op_and{}, detail_semd::begin(a, b),
              detail_semd::end(a, b)};
}

auto operator||(is_semd_c auto const a, is_semd_c auto const b) {
  return expr{ops::semd_op_or{}, detail_semd::begin(a, b),
              detail_semd::end(a, b)};
}

auto operator!(is_semd_c auto const a) {
  return expr{ops::semd_op_not{}, detail_semd::begin(a), detail_semd::end(a)};
}

auto operator<(aks::is_value auto const a, is_semd_c auto const b) {
  return detail_semd::cexpr(a, b) < b;
}

auto operator<=(aks::is_value auto const a, is_semd_c auto const b) {
  return detail_semd::cexpr(a, b) <= b;
}

auto operator>(aks::is_value auto const a, is_semd_c auto const b) {
  return detail_semd::cexpr(a, b) > b;
}

auto operator>=(aks::is_value auto const a, is_semd_c auto const b) {
  return detail_semd::cexpr(a, b) >= b;
}

auto operator==(aks::is_value auto const a, is_semd_c auto const b) {
  return detail_semd::cexpr(a, b) == b;
}

auto operator!=(aks::is_value auto const a, is_semd_c auto const b) {
  return detail_semd::cexpr(a, b) != b;
}

auto operator&&(aks::is_value auto const a, is_semd_c auto const b) {
  return detail_semd::cexpr(a, b) && b;
}

auto operator||(aks::is_value auto const a, is_semd_c auto const b) {
  return detail_semd::cexpr(a, b) || b;
}

auto operator<(is_semd_c auto const a, aks::is_value auto const b) {
  return a < detail_semd::cexpr(b, a);
}

auto operator<=(is_semd_c auto const a, aks::is_value auto const b) {
  return a <= detail_semd::cexpr(b, a);
}

auto operator>(is_semd_c auto const a, aks::is_value auto const b) {
  return a > detail_semd::cexpr(b, a);
}

auto operator>=(is_semd_c auto const a, aks::is_value auto const b) {
  return a >= detail_semd::cexpr(b, a);
}

auto operator==(is_semd_c auto const a, aks::is_value auto const b) {
  return a == detail_semd::cexpr(b, a);
}

auto operator!=(is_semd_c auto const a, aks::is_value auto const b) {
  return a != detail_semd::cexpr(b, a);
}

auto operator&&(is_semd_c auto const a, aks::is_value auto const b) {
  return a && detail_semd::cexpr(b, a);
}

auto operator||(is_semd_c auto const a, aks::is_value auto const b) {
  return a || detail_semd::cexpr(b, a);
}

auto select(is_semd_c auto const a, is_semd_c auto const b,
            is_semd_c auto const c) {
  return expr{ops::semd_op_select{}, detail_semd::begin(a, b, c),
              detail_semd::end(a, b, c)};
}

auto select(is_semd_c auto const a, aks::is_value auto const b,
            is_semd_c auto const c) {
  return select(a, detail_semd::cexpr(b, a), c);
}

auto select(is_semd_c auto const a, is_semd_c auto const b,
            aks::is_value auto const c) {
  return select(a, b, detail_semd::cexpr(c, a));
}

auto select(is_semd_c auto const a, aks::is_value auto const b,
            aks::is_value auto const c) {
  return select(a, detail_semd::cexpr(b, a), detail_semd::cexpr(c, a));
}

auto select(aks::is_value auto const a, is_semd_c auto const b,
            is_semd_c auto const c) {
  return select(detail_semd::cexpr(a, b), b, c);
}

template <typename F, is_semd_c... Args>
auto custom_op(F f, Args const... args) {
  return expr{ops::semd_custom_op{f}, detail_semd::begin(args...),
              detail_semd::end(args...)};
}

auto size(is_semd_c auto const a) { return a.size(); }

auto begin(is_semd_c auto const a) { return a.begin(); }

auto end(is_semd_c auto const a) { return a.end(); }

auto any(is_semd_c auto const a) {
  return std::any_of(a.begin(), a.end(), [](auto const &v) {
    if constexpr (aks::simd::is_mask_c<std::remove_cvref_t<decltype(v)>>) {
      return aks::simd::any(v);
    } else {
      return v;
    }
  });
}

auto all(is_semd_c auto const a) {
  return std::all_of(a.begin(), a.end(), [](auto const &v) {
    if constexpr (aks::simd::is_mask_c<std::remove_cvref_t<decltype(v)>>) {
      return aks::simd::all(v);
    } else {
      return v;
    }
  });
}

} // namespace semd
} // namespace aks

#if 0

template <size_t N>
auto func(aks::semd::is_semd_c auto const a, aks::semd::is_semd_c auto const b,
          aks::semd::is_semd_c auto const c, aks::is_value auto const d) {
  if constexpr (N > 0) {
    return select(d < decltype(d)(2.0),
                  a * func<N - 1>(a + c, b, c * d, d + decltype(d)(1.0)),
                  b * func<N - 1>(a * c, b, c * d, d * decltype(d)(0.2)));
  } else {
    return a * b;
  }
}

void check() {
  // using namespace aks::semd;
  using namespace aks::simd;
  using namespace aks::semd;
  using dvec = std::vector<double>;
  using avec = std::array<double, 4>;
  using vdvec = std::vector<vec4d>;
  using vavec = std::array<vec4d, 4>;
  {
    std::vector<double> xs{1., 2., 5., 6., 8., 10.};
    std::array<float, 6> ys{6., 1., 2., 8., 9., 10.};

    semd x{xs};
    semd y{ys};

    std::print("x = {}\n", x);
    std::print("y = {}\n", y);
    std::print("any(x < y) = {}\n", any(x < y));
    std::print("all(x < y) = {}\n", all(x < y));
    std::print("x < y = {}\n", x < y);
    std::print("x == y = {}\n", x == y);
  }
  {
    dvec xs{1., 2., 5., 6.};
    avec ys{6., 1., 2., 8.};

    semd x{xs};
    semd y{ys};

    std::print("any(x < y) = {}\n", any(x < y));
    std::print("all(x < y) = {}\n", all(x < y));
  }
  {
    vdvec xs{vec4d{1.0}, vec4d{2.0}, vec4d{5.0}, vec4d{6.0}};
    vavec ys{vec4d{6.0}, vec4d{1.0}, vec4d{2.0}, vec4d{8.0}};

    semd x{xs};
    semd y{ys};

    std::print("any(x < y) = {}\n", any(x < y));
    std::print("all(x < y) = {}\n", all(x < y));
  }
  {
    vdvec xs{vec4d{1.0}, vec4d{2.0}, vec4d{5.0}, vec4d{6.0}};
    vavec ys{vec4d{6.0}, vec4d{1.0}, vec4d{2.0}, vec4d{8.0}};

    semd x{xs};
    semd y{ys};

    auto f = custom_op(
        [](auto x, auto y, auto z, auto w) {
          std::print("{} + {} + {} + {}\n", x, y, z, w);
          return x + y + z + w;
        },
        x, y, x, y);

    std::print("f = {}\n", f);
  }

  {
    vdvec xs{vec4d{1.0}, vec4d{2.0}, vec4d{5.0}, vec4d{6.0}};
    vavec ys{vec4d{6.0}, vec4d{1.0}, vec4d{2.0}, vec4d{8.0}};

    semd x{xs};
    semd y{ys};

    std::print("func_0 = {}\n", func<1>(x, y, x, vec4d{1.0}));
    std::print("func_2 = {}\n", func<1>(x, y, x, vec4d{2.0}));
    std::print("func_3 = {}\n", func<1>(x, y, x, vec4d{3.0}));
  }
  {
    dvec xs{1.0, 2.0, 5.0, 6.0};
    avec ys{6.0, 1.0, 2.0, 8.0};

    semd x{xs};
    semd y{ys};

    std::print("s_func_0 = {}\n", func<1>(x, y, x, 1.0));
    std::print("s_func_2 = {}\n", func<1>(x, y, x, 2.0));
    std::print("s_func_3 = {}\n", func<1>(x, y, x, 4.0));
  }
  {
    vdvec xs{vec4d{1.0}, vec4d{2.0}, vec4d{5.0}, vec4d{6.0}};
    vavec ys{vec4d{6.0}, vec4d{1.0}, vec4d{2.0}, vec4d{8.0}};

    semd x{xs};
    semd y{ys};

    std::print("select(x < 5, 8, 9) = {}\n",
               select(x < vec4d{5.}, vec4d{11}, vec4d{9}) / vec4d{2});
    std::print(" x > 5  && y > 5= {}\n", (x >= vec4d{5}) || (y >= vec4d{5}));
    std::print("yay! => x + 5.2 = {}\n", x + vec4d{5.2});
    std::print("yay! => select(x < y, x * x, y * y) = {}\n",
               select(x < y, x * x, y * y));
  }
  {
    dvec xs{1.0, 2.0, 5.0, 6.0};
    avec ys{6.0, 1.0, 2.0, 8.0};

    semd x{xs};
    semd y{ys};

    std::print("select(x < 5, 8, 9) = {}\n", select(x < 5., 11, 9) / 2);
    std::print(" x > 5  && y > 5= {}\n", (x >= 5) || (y >= 5));
    std::print("yay! => x + 5.2 = {}\n", x + 5.2);
    std::print("yay! => select(x < y, x * x, y * y) = {}\n",
               select(x < y, x * x, y + y));
  }
  {
    dvec xs{1.0, 2.0, 5.0, 6.0};
    dvec ys{5.0, 1.0, 2.0, 8.0};
    dvec zs{-19., -18., -17., -16.};
    dvec ws{0.0, 0.0, 0.0, 0.0};

    semd x{xs};
    semd y{ys};
    semd z{zs};
    semd w{ws};

    std::print("x < y = {}\n", select(x < y, x, y));
  }
  {
    vdvec xs{{1.0, 2.0, 5.0, 6.0}};
    vdvec ys{{5.0, 1.0, 2.0, 8.0}};
    vdvec zs{{-19., -18., -17., -16.}};
    vdvec ws{{0.0, 0.0, 0.0, 0.0}};

    semd x{xs};
    semd y{ys};
    semd z{zs};
    semd w{ws};

    std::print("x < y = {}\n", select(x < y, x, y));
  }
  {
    using avec = std::array<double, 4>;
    avec xs{1.0, 2.0, 5.0, 6.0};
    avec ys{5.0, 6.0, 7.0, 8.0};
    avec zs{-19., -18., -17., -16.};
    avec ws{0.0, 0.0, 0.0, 0.0};

    semd x{xs};
    semd y{ys};
    semd z{zs};
    semd w{ws};

    w = x + y + z;

    std::print("x = {}\ny = {}\nz = {}\n", x, y, z);
    std::print("w = {}\nx + y + z = {}\n", w, x + y + z);
    std::print("x = {}, -x = {}, -(-x) = {}\n", x, -x, -(-x));
    std::print("x * (y / z) = {}\n", x * (y / z));
    std::print("abs(z) = {}\n", abs(z));
  }
  {
    dvec xs{1.0, 2.0, 5.0, 6.0};
    dvec ys{5.0, 6.0, 7.0, 8.0};
    dvec zs{-19., -18., -17., -16.};
    dvec ws{0.0, 0.0, 0.0, 0.0};

    semd x{xs};
    semd y{ys};
    semd z{zs};
    semd w{ws};

    auto e = x + y + z;
    semd ee{e};

    w = e;

    std::print("x = {}\ny = {}\nz = {}\n", x, y, z);
    std::print("w = {}\nx + y + z = {}\n", w, x + y + z);
    std::print("x = {}, -x = {}, -(-x) = {}\n", x, -x, -(-x));
    std::print("x * (y / z) = {}\n", x * (y / z));
    std::print("abs(z) = {}\n", abs(z));
  }
}

int main() {
  check();
  using namespace aks::simd;
  auto x = vec4d{1.0, 2.0, 5.0, 6.0};
  auto y = vec4d{5.0};
  auto z = vec4d{};
  auto w = vec4d{zero_v};
  std::print(
      "x = {}\ny = {}\nz = {}\nw = {}\nx + y = {}\nx < y = {}\nx == y = "
      "{}\nx "
      "<= y = {}\nx < y || x == y = {}\nany(x < y && x == y) = {}\nany(x "
      "== "
      "y) = {}\n",
      x, y, z, w, x + y, x < y, x == y, x <= y, (x < y) || (x == y),
      any(x < y && x == y), any(x == y));

  std::print("x = {}\ny = {}\nx < y = {}\nselect(x<y, x, y) = {}\n", x, y,
             x < y, select(x < y, x, y));
  std::print("{}\n{}\n", -x, -w);

  auto xf = vec1f{1.2f};
  auto yf = vec1f{5.1f};
  std::print("xf = {}\nyf = {}\nxf < yf = {}\n", xf, yf, xf < yf);
  std::print("xf < yf ? yf : xf = {}\n", select(xf < yf, yf, xf));
  std::print("xf > yf ? yf : xf = {}\n", select(xf > yf, yf, xf));
}

#endif