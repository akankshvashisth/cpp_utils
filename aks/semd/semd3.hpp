

#ifndef AKS_SEMD_HPP_01
#define AKS_SEMD_HPP_01

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <iterator>
#include <tuple>
#include <type_traits>

#define AKS_SEMD_OP_FUNC_PREFIX [[nodiscard]] inline constexpr

namespace aks {
namespace semd3 {
namespace detail_semd3 {
template <typename... ts> using iters_type_t = std::tuple<ts...>;

template <typename... t_iters>
AKS_SEMD_OP_FUNC_PREFIX auto make_iters(t_iters... iters)
    -> iters_type_t<t_iters...> {
  return std::make_tuple(iters...);
}
} // namespace detail_semd3

template <typename t_oper, typename... t_iters>
struct semd_random_access_iterator {
  // check random access iterator
  static_assert(
      (std::is_same_v<
           typename std::random_access_iterator_tag,
           typename std::iterator_traits<t_iters>::iterator_category> &&
       ...),
      "only support random access iterators");

  static auto applyf(auto const &op, auto const &ts) {
    auto op_wrap = [&](auto &...vs) { return op((*vs)...); };
    return std::apply(op_wrap, ts);
  }

  using oper_t = t_oper;
  using iters_t = detail_semd3::iters_type_t<t_iters...>;

  oper_t op_;
  iters_t ts_;

  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;

  using value_type = std::remove_cvref_t<decltype(applyf(
      std::declval<oper_t>(), std::declval<iters_t>()))>;
  using pointer = value_type;
  using reference = value_type;

  AKS_SEMD_OP_FUNC_PREFIX reference operator*() const {
    return applyf(op_, ts_);
  }

  semd_random_access_iterator &operator++() {
    std::apply([&](auto &...vs) { (++vs, ...); }, ts_);
    return *this;
  }
  semd_random_access_iterator operator++(int) {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }
  semd_random_access_iterator &operator--() {
    std::apply([&](auto &...vs) { (--vs, ...); }, ts_);
    return *this;
  }
  semd_random_access_iterator operator--(int) {
    auto tmp = *this;
    --(*this);
    return tmp;
  }

  semd_random_access_iterator &operator+=(difference_type n) {
    std::apply([&](auto &...vs) { ((vs += n), ...); }, ts_);
    return *this;
  }
  semd_random_access_iterator &operator-=(difference_type n) {
    std::apply([&](auto &...vs) { ((vs -= n), ...); }, ts_);
    return *this;
  }

  AKS_SEMD_OP_FUNC_PREFIX reference operator[](difference_type n) const {
    auto tmp = *this;
    tmp += n;
    return *(tmp);
  }

  AKS_SEMD_OP_FUNC_PREFIX bool
  operator==(const semd_random_access_iterator &other) const {
    return ts_ == other.ts_;
  }
  AKS_SEMD_OP_FUNC_PREFIX bool
  operator!=(const semd_random_access_iterator &other) const {
    return !(*this == other);
  }
  AKS_SEMD_OP_FUNC_PREFIX auto
  operator<=>(const semd_random_access_iterator &other) const {
    return ts_ <=> other.ts_;
  }
};

namespace concepts {
namespace concepts_detail {
template <typename T>
struct is_semd_random_access_iterator_type : std::false_type {};

template <typename Op, typename... iters_>
struct is_semd_random_access_iterator_type<
    semd_random_access_iterator<Op, iters_...>> : std::true_type {};

template <typename T>
constexpr static bool is_semd_random_access_iterator_v =
    is_semd_random_access_iterator_type<T>::value;
} // namespace concepts_detail
template <typename T>
concept is_semd_random_access_iterator =
    concepts_detail::is_semd_random_access_iterator_v<T>;
} // namespace concepts

AKS_SEMD_OP_FUNC_PREFIX std::ptrdiff_t
operator-(concepts::is_semd_random_access_iterator auto const &x,
          concepts::is_semd_random_access_iterator auto const &y) {
  return std::get<0>(x.ts_) - std::get<0>(y.ts_);
}

template <typename t_oper, typename... t_iters> struct semd_expression {
  using oper_t = t_oper;
  using iters_t = detail_semd3::iters_type_t<t_iters...>;
  using iterator = semd_random_access_iterator<oper_t, t_iters...>;
  using const_iterator = semd_random_access_iterator<oper_t, t_iters...>;
  using value_type = std::remove_cvref_t<decltype(*(std::declval<iterator>()))>;

  // semd_expression(oper_t op, iters_t begin, iters_t end)
  //     : op_(op), begin_(begin), end_(end) {}
  //
  // semd_expression(semd_expression const& other) = default;
  // semd_expression(semd_expression&& other) = default;
  // semd_expression& operator=(semd_expression const& other) = default;
  // semd_expression& operator=(semd_expression&& other) = default;

  AKS_SEMD_OP_FUNC_PREFIX auto begin() const { return iterator{op_, begin_}; }
  AKS_SEMD_OP_FUNC_PREFIX auto end() const { return iterator{op_, end_}; }

  AKS_SEMD_OP_FUNC_PREFIX auto cbegin() const {
    return const_iterator{op_, begin_};
  }
  AKS_SEMD_OP_FUNC_PREFIX auto cend() const {
    return const_iterator{op_, end_};
  }

  AKS_SEMD_OP_FUNC_PREFIX auto size() const {
    using std::distance;
    return distance(std::get<0>(begin_), std::get<0>(end_));
  }

  oper_t op_{};
  iters_t begin_{};
  iters_t end_{};
};

namespace concepts {
namespace concepts_detail {
template <typename T> struct is_semd_expression_type : std::false_type {};

template <typename Op, typename... iters_>
struct is_semd_expression_type<semd_expression<Op, iters_...>>
    : std::true_type {};

template <typename T>
constexpr static bool is_semd_expression_v = is_semd_expression_type<T>::value;
} // namespace concepts_detail
template <typename T>
concept is_semd_expression = concepts_detail::is_semd_expression_v<T>;
} // namespace concepts

template <typename T>
  requires requires(T t) {
    { size(t) } -> std::convertible_to<std::size_t>;
    { begin(t) } -> std::convertible_to<typename T::const_iterator>;
    { end(t) } -> std::convertible_to<typename T::const_iterator>;
  }
struct semdvec {
  using value_type = typename T::value_type;
  using iterator = typename T::iterator;

  iterator begin_{}, end_{};
  std::size_t size_{};

  AKS_SEMD_OP_FUNC_PREFIX std::size_t size() const { return size_; }

  explicit semdvec(T &data) {
    using std::begin;
    using std::end;
    using std::size;

    begin_ = begin(data);
    end_ = end(data);
    size_ = end_ - begin_; // size(data);
  }

  explicit semdvec(T const &data) {
    using std::begin;
    using std::end;
    using std::size;

    begin_ = begin(data);
    end_ = end(data);
    size_ = end_ - begin_;
  }

  semdvec(semdvec &&other) = default;
  semdvec(const semdvec &other) = default;

  template <typename U> semdvec &operator=(const semdvec<U> &other) {
    assert(size_ == other.size_);
    for (size_t i = 0; i < size_; ++i) {
      begin_[i] = other.begin_[i];
    }
    return *this;
  }

  semdvec &operator=(const semdvec &other) {
    if (this == &other) {
      return *this;
    }
    assert(size_ == other.size_);
    for (size_t i = 0; i < size_; ++i) {
      begin_[i] = other.begin_[i];
    }
    return *this;
  }

  semdvec &operator=(concepts::is_semd_expression auto expr) {
    assert(size_ == expr.size());

    for (size_t i = 0; i < size_; ++i) {
      begin_[i] = expr.begin()[i];
    }

    return *this;
  }

  semdvec &operator=(semdvec &&other) = default;

  AKS_SEMD_OP_FUNC_PREFIX decltype(auto) begin() const { return begin_; }
  AKS_SEMD_OP_FUNC_PREFIX decltype(auto) end() const { return end_; }

  AKS_SEMD_OP_FUNC_PREFIX decltype(auto) cbegin() const { return begin_; }
  AKS_SEMD_OP_FUNC_PREFIX decltype(auto) cend() const { return end_; }
};

namespace concepts {
namespace concepts_detail {
template <typename T> struct is_semdvec_type : std::false_type {};

template <typename T> struct is_semdvec_type<semdvec<T>> : std::true_type {};

template <typename T>
constexpr static bool is_semd_v = is_semdvec_type<T>::value;
} // namespace concepts_detail
template <typename T>
concept is_semdvec = concepts_detail::is_semd_v<T>;

template <typename T>
concept is_semd = is_semdvec<T> || is_semd_expression<T>;

} // namespace concepts

namespace concepts {
template <typename T>
concept is_number = std::is_integral_v<T> || std::is_floating_point_v<T>;

template <typename T>
concept is_boolean = std::same_as<T, bool>;

template <typename T>
concept is_number_or_boolean = is_number<T> || is_boolean<T>;

template <typename T> struct is_semd_supported_value_type : std::false_type {};

template <is_number_or_boolean T>
struct is_semd_supported_value_type<T> : std::true_type {};

template <typename T>
concept is_semd_supported_value = is_semd_supported_value_type<T>::value;
}; // namespace concepts

namespace concepts {

template <typename T>
concept has_plus = requires(T x, T y) {
  { x + y };
};

template <typename T>
concept has_minus = requires(T x, T y) {
  { x - y };
};

template <typename T>
concept has_mul = requires(T x, T y) {
  { x *y };
};

template <typename T>
concept has_div = requires(T x, T y) {
  { x / y };
};

template <typename T>
concept has_neg = requires(T x) {
  { -x };
};

template <typename T>
concept has_eq = requires(T x, T y) {
  { x == y };
};

template <typename T>
concept has_ne = requires(T x, T y) {
  { x != y };
};

template <typename T>
concept has_gt = requires(T x, T y) {
  { x > y };
};

template <typename T>
concept has_lt = requires(T x, T y) {
  { x < y };
};

template <typename T>
concept has_ge = requires(T x, T y) {
  { x >= y };
};

template <typename T>
concept has_le = requires(T x, T y) {
  { x <= y };
};

template <typename T>
concept has_or = requires(T x, T y) {
  { x || y };
};

template <typename T>
concept has_and = requires(T x, T y) {
  { x &&y };
};

template <typename T>
concept has_not = requires(T x) {
  { !x };
};

template <typename T>
concept has_mod = requires(T x, T y) {
  { x % y };
};

template <typename T>
concept has_bitwise_and = requires(T x, T y) {
  { x &y };
};

template <typename T>
concept has_bitwise_or = requires(T x, T y) {
  { x | y };
};

template <typename T>
concept has_bitwise_xor = requires(T x, T y) {
  { x ^ y };
};

template <typename T>
concept has_abs = requires(T x) {
  { abs(x) };
};

template <typename T>
concept has_sqrt = requires(T x) {
  { sqrt(x) };
};

template <typename T>
concept has_select = requires(T x, T y, T z) {
  { select(x, y, z) };
};

} // namespace concepts

namespace ops {

AKS_SEMD_OP_FUNC_PREFIX auto op_add_(concepts::has_plus auto x,
                                     concepts::has_plus auto y) {
  return x + y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_sub_(concepts::has_minus auto x,
                                     concepts::has_minus auto y) {
  return x - y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_mul_(concepts::has_mul auto x,
                                     concepts::has_mul auto y) {
  return x * y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_div_(concepts::has_div auto x,
                                     concepts::has_div auto y) {
  return x / y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_neg_(concepts::has_neg auto x) { return -x; }

AKS_SEMD_OP_FUNC_PREFIX auto op_eq_(concepts::has_eq auto x,
                                    concepts::has_eq auto y) {
  return x == y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_ne_(concepts::has_ne auto x,
                                    concepts::has_ne auto y) {
  return x != y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_gt_(concepts::has_gt auto x,
                                    concepts::has_gt auto y) {
  return x > y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_lt_(concepts::has_lt auto x,
                                    concepts::has_lt auto y) {
  return x < y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_ge_(concepts::has_ge auto x,
                                    concepts::has_ge auto y) {
  return x >= y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_le_(concepts::has_le auto x,
                                    concepts::has_le auto y) {
  return x <= y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_or_(concepts::has_or auto x,
                                    concepts::has_or auto y) {
  return x || y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_and_(concepts::has_and auto x,
                                     concepts::has_and auto y) {
  return x && y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_not_(concepts::has_not auto x) { return !x; }

AKS_SEMD_OP_FUNC_PREFIX auto op_mod_(concepts::has_mod auto x,
                                     concepts::has_mod auto y) {
  return x % y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_bitwise_and_(concepts::has_bitwise_and auto x,
                                             concepts::has_bitwise_and auto y) {
  return x & y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_bitwise_or_(concepts::has_bitwise_or auto x,
                                            concepts::has_bitwise_or auto y) {
  return x | y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_bitwise_xor_(concepts::has_bitwise_xor auto x,
                                             concepts::has_bitwise_xor auto y) {
  return x ^ y;
}

AKS_SEMD_OP_FUNC_PREFIX auto op_abs_(concepts::has_abs auto x) {
  using std::abs;
  return abs(x);
}

AKS_SEMD_OP_FUNC_PREFIX auto op_sqrt_(concepts::has_sqrt auto x) {
  using std::sqrt;
  return sqrt(x);
}

AKS_SEMD_OP_FUNC_PREFIX auto op_id_(auto x) { return x; }

AKS_SEMD_OP_FUNC_PREFIX auto op_select_(concepts::is_number_or_boolean auto x,
                                        concepts::is_number_or_boolean auto y,
                                        concepts::is_number_or_boolean auto z) {
  return x ? y : z;
}

} // namespace ops

namespace detail_semd3 {
AKS_SEMD_OP_FUNC_PREFIX auto begins(concepts::is_semd auto const x,
                                    concepts::is_semd auto const... xs) {
  auto const check = ((x.size() == xs.size()) && ...);
  assert(check);
  return make_iters(x.begin(), xs.begin()...);
}
AKS_SEMD_OP_FUNC_PREFIX auto ends(concepts::is_semd auto const x,
                                  concepts::is_semd auto const... xs) {
  auto const check = ((x.size() == xs.size()) && ...);
  assert(check);
  return make_iters(x.end(), xs.end()...);
}

AKS_SEMD_OP_FUNC_PREFIX auto
cexpr(concepts::is_semd_supported_value auto const a,
      concepts::is_semd auto const b) {
  return semd_expression{[a](auto) { return a; }, begins(b), ends(b)};
}

} // namespace detail_semd3

#define AKS_SEMD_DETAIL_UN_OP_FUNC_DEFINE(NAME, OP)                            \
  AKS_SEMD_OP_FUNC_PREFIX auto OP(concepts::is_semd auto const a) {            \
    return semd_expression{[](auto x) { return ops::op_##NAME(x); },           \
                           detail_semd3::begins(a), detail_semd3::ends(a)};    \
  }

#define AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(NAME, OP)                           \
  AKS_SEMD_OP_FUNC_PREFIX auto OP(concepts::is_semd auto const a,              \
                                  concepts::is_semd auto const b) {            \
    assert(a.size() == b.size());                                              \
    return semd_expression{                                                    \
        [](auto x, auto y) { return ops::op_##NAME(x, y); },                   \
        detail_semd3::begins(a, b), detail_semd3::ends(a, b)};                 \
  }                                                                            \
  AKS_SEMD_OP_FUNC_PREFIX auto OP(                                             \
      concepts::is_semd auto const a,                                          \
      concepts::is_semd_supported_value auto const b) {                        \
    return OP(a, detail_semd3::cexpr(b, a));                                   \
  }                                                                            \
  AKS_SEMD_OP_FUNC_PREFIX auto OP(                                             \
      concepts::is_semd_supported_value auto const a,                          \
      concepts::is_semd auto const b) {                                        \
    return OP(detail_semd3::cexpr(a, b), b);                                   \
  }

AKS_SEMD_DETAIL_UN_OP_FUNC_DEFINE(id_, identity);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(add_, operator+);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(sub_, operator-);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(mul_, operator*);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(div_, operator/);
AKS_SEMD_DETAIL_UN_OP_FUNC_DEFINE(neg_, operator-);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(eq_, operator==);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(ne_, operator!=);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(gt_, operator>);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(lt_, operator<);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(ge_, operator>=);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(le_, operator<=);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(or_, operator||);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(and_, operator&&);
AKS_SEMD_DETAIL_UN_OP_FUNC_DEFINE(not_, operator!);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(mod_, operator%);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(bitwise_and_, operator&);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(bitwise_or_, operator|);
AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE(bitwise_xor_, operator^);
AKS_SEMD_DETAIL_UN_OP_FUNC_DEFINE(abs_, abs);
AKS_SEMD_DETAIL_UN_OP_FUNC_DEFINE(sqrt_, sqrt);

template <typename T, typename U, typename V> struct select_f;

template <concepts::is_number_or_boolean T, concepts::is_number_or_boolean U,
          concepts::is_number_or_boolean V>
struct select_f<T, U, V> {
  auto operator()(T x, U y, V z) const { return ops::op_select_(x, y, z); }
};

template <typename T, typename U, typename V>
auto make_select_f(T x, U y, V z) {
  return select_f<typename T::value_type, typename U::value_type,
                  typename V::value_type>{};
}

AKS_SEMD_OP_FUNC_PREFIX auto select(concepts::is_semd auto const a,
                                    concepts::is_semd auto const b,
                                    concepts::is_semd auto const c) {
  assert(a.size() == b.size() && b.size() == c.size());
  return semd_expression{make_select_f(a, b, c), detail_semd3::begins(a, b, c),
                         detail_semd3::ends(a, b, c)};
}

template <typename T> struct any_f;

template <typename T> struct all_f;

template <concepts::is_number_or_boolean T> struct any_f<T> {
  bool operator()(T x) const { return bool(x); }
};

template <concepts::is_number_or_boolean T> struct all_f<T> {
  bool operator()(T x) const { return bool(x); }
};

AKS_SEMD_OP_FUNC_PREFIX auto any(concepts::is_semd auto const xs) {
  for (auto const &x : xs) {
    if (any_f<std::remove_cvref_t<decltype(x)>>{}(x)) {
      return true;
    }
  }
  return false;
}

AKS_SEMD_OP_FUNC_PREFIX auto all(concepts::is_semd auto const xs) {
  for (auto const &x : xs) {
    if (!all_f<std::remove_cvref_t<decltype(x)>>{}(x)) {
      return false;
    }
  }
  return true;
}

AKS_SEMD_OP_FUNC_PREFIX auto vec_apply(auto f, concepts::is_semd auto const x,
                                       concepts::is_semd auto const... xs) {
  assert(((x.size() == xs.size()) && ...));
  return semd_expression{f, detail_semd3::begins(x, xs...),
                         detail_semd3::ends(x, xs...)};
}

#undef AKS_SEMD_DETAIL_UN_OP_FUNC_DEFINE
#undef AKS_SEMD_DETAIL_BIN_OP_FUNC_DEFINE
#undef AKS_SEMD_DETAIL_TERNARY_OP_FUNC_DEFINE

} // namespace semd3
} // namespace aks

#undef AKS_SEMD_OP_FUNC_PREFIX

#endif // !AKS_SEMD_HPP_01

#if 0

#define FMT_HEADER_ONLY

#include <array>
#include <vector>

#include <fmt/format.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

#include "simd.hpp"

namespace aks::semd3 {
namespace concepts {

template <>
struct is_semd_supported_value_type<aks::simd::vec4d> : std::true_type {};

template <>
struct is_semd_supported_value_type<aks::simd::vec4d_mask> : std::true_type {};
}  // namespace concepts

template <aks::simd::is_simd_c T,
          aks::simd::is_simd_c U,
          aks::simd::is_simd_c V>
struct select_f<T, U, V> {
  auto operator()(T x, U y, V z) const { return aks::simd::select(x, y, z); }
};

template <aks::simd::is_simd_c T>
struct any_f<T> {
  bool operator()(T x) const { return aks::simd::any(x); }
};

template <aks::simd::is_simd_c T>
struct all_f<T> {
  bool operator()(T x) const { return aks::simd::all(x); }
};
}  // namespace aks::semd3

#include <string>

void examples() {
  {
    std::vector<std::string> xs{"a", "b", "c"};
    std::vector<std::string> ys{"d", "e", "f"};

    aks::semd::semdvec x{xs};
    aks::semd::semdvec y{ys};

    fmt::println("x + y = {}", x + y + x + y);
  }
  {
    using aks::simd::vec4d;

    std::vector<vec4d> xs{vec4d{-1., 2., 5., 6.}, vec4d{8., 10., 1.2, 1.5}};
    std::array<vec4d, 2> ys{vec4d{6., 1., 2., 8.}, vec4d{9., 10., 1.1, 1.6}};
    std::array<vec4d, 2> zs{vec4d{6., 1., 2., 8.}, vec4d{9., 10., 1.2, 1.6}};

    aks::semd::semdvec x{xs};
    aks::semd::semdvec y{ys};
    aks::semd::semdvec z{zs};

    z = y;

    fmt::println("z = {}", z);

    fmt::print("x = {}\n", x);
    fmt::print("y = {}\n", y);
    fmt::println("id(y) = {}", identity(y));
    // fmt::println("pow = {}", vec_apply(std::pow<double, double>, x, y));
    fmt::print("sqrt(y) = {}\n", sqrt(y));
    fmt::print("select(x < y, x, y) = {}\n", select(x < y, x, y));
    fmt::print("x*y = {}\n", x * y);
    fmt::print("2.0*y = {}\n", vec4d{2.0} * y);
    fmt::print("y*2 = {}\n", y * vec4d{2.0});
    fmt::print("abs(x) = {}\n", abs(x));
    fmt::print("any(x < y) = {}\n", any(x < y));
    fmt::print("all(x < y) = {}\n", all(x < y));
    fmt::print("x < y = {}\n", x < y);
    fmt::print("x == y = {}\n", x == y);
  }

  {
    std::vector<aks::simd::vec1f> xs{-1.f, 2.f, 5.f, 6.f, 8.f, 10.f};
    std::array<aks::simd::vec1f, 6> ys{6.f, 1.f, 2.f, 8.f, 9.f, 10.f};
    std::array<aks::simd::vec1f, 6> zs{6.f, 1.f, 2.f, 8.f, 9.f, 10.f};

    aks::semd::semdvec x{xs};
    aks::semd::semdvec y{ys};
    aks::semd::semdvec z{zs};

    z = y;

    fmt::println("z = {}", z);

    fmt::print("x = {}\n", x);
    fmt::print("y = {}\n", y);
    fmt::println("id(y) = {}", identity(y));
    fmt::println("pow = {}", vec_apply(std::pow<double, double>, x, y));
    fmt::print("sqrt(y) = {}\n", sqrt(y));
    fmt::print("select(x < y, x, y) = {}\n", select(x < y, x, y));
    fmt::print("x*y = {}\n", x * y);
    fmt::print("2.0*y = {}\n", 2.0 * y);
    fmt::print("y*2 = {}\n", y * 2);
    fmt::print("abs(x) = {}\n", abs(x));
    fmt::print("any(x < y) = {}\n", any(x < y));
    fmt::print("all(x < y) = {}\n", all(x < y));
    fmt::print("x < y = {}\n", x < y);
    fmt::print("x == y = {}\n", x == y);
  }

  {
    std::array<float, 6> xs{-1., 2., 5., 6., 8., 10.};
    std::array<float, 6> ys{6., 1., 2., 8., 9., 10.};
    std::array<float, 6> zs{6., 1., 2., 8., 9., 10.};

    aks::semd::semdvec x{xs};
    aks::semd::semdvec y{ys};
    aks::semd::semdvec z{zs};

    auto f0 = x + y;
    auto f1 = f0 + x;
    auto f2 = f1 + y;
    z = f2;

    fmt::println("z = {}", z);

    fmt::print("x = {}\n", x);
    fmt::print("y = {}\n", y);
    fmt::print("foo = {}\n", x + y + x + y);
    fmt::println("id(y) = {}", identity(y));
    fmt::println("pow = {}", vec_apply(std::pow<double, double>, x, y));
    fmt::print("sqrt(y) = {}\n", sqrt(y));
    fmt::print("select(x < y, x, y) = {}\n", select(x < y, x, y));
    fmt::print("x*y = {}\n", x * y);
    fmt::print("2.0*y = {}\n", 2.0 * y);
    fmt::print("y*2 = {}\n", y * 2);
    fmt::print("abs(x) = {}\n", abs(x));
    fmt::print("any(x < y) = {}\n", any(x < y));
    fmt::print("all(x < y) = {}\n", all(x < y));
    fmt::print("x < y = {}\n", x < y);
    fmt::print("x == y = {}\n", x == y);
  }
}

int main() {
  examples();
  return 0;
}

#endif