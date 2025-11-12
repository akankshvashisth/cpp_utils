
#ifndef AKS_SEMD_HPP
#define AKS_SEMD_HPP

#include <cassert>
#include <ranges>
#include <type_traits>

#include <cmath>
#include <string_view>

namespace aks {
namespace semd {
#define AKS_SEMD_OP_FUNC_PREFIX [[nodiscard]] inline constexpr

template <std::move_constructible Op_, std::ranges::input_range... Ts_>
struct expression
    : public std::ranges::view_interface<expression<Op_, Ts_...>> {
  auto begin() const { return rng_.begin(); }
  auto end() const { return rng_.end(); }

  expression(Op_ op, Ts_... ts) : rng_(op, ts...) {
    // check sizes equal
  }

  decltype(auto) operator[](std::size_t i) { return rng_[i]; }
  decltype(auto) operator[](std::size_t i) const { return rng_[i]; }

  std::ranges::zip_transform_view<Op_, Ts_...> rng_;
};

template <std::ranges::range T>
struct semd : public std::ranges::view_interface<semd<T>> {
  auto begin() const { return rng_.begin(); }
  auto end() const { return rng_.end(); }
  semd(T& t) : rng_(t) {}

  semd(semd const&) = default;
  semd(semd&&) = default;

  semd& operator=(semd const&) = default;

  template <std::ranges::range U>
  semd& operator=(semd<U> const& u) {
    // check sizes equal
    auto& me = *this;
    for (auto i = 0; i < this->size(); ++i) {
      me[i] = u[i];
    }
    return *this;
  }

  template <typename Op_, typename... Ts_>
  semd& operator=(expression<Op_, Ts_...> const& u) {
    // check sizes equal
    auto& me = *this;
    for (auto i = 0; i < this->size(); ++i) {
      me[i] = u[i];
    }
    return *this;
  }

  decltype(auto) operator[](std::size_t i) { return rng_[i]; }
  decltype(auto) operator[](std::size_t i) const { return rng_[i]; }

  std::ranges::ref_view<T> rng_;
};

namespace concepts {
template <typename T>
struct is_expr_type : std::false_type {};

template <std::move_constructible Op_, std::ranges::input_range... Ts_>
struct is_expr_type<expression<Op_, Ts_...>> : std::true_type {};

template <typename T>
inline constexpr bool is_expr_type_v = is_expr_type<T>::value;

template <typename T>
concept is_expr_c = is_expr_type_v<T>;

template <typename T>
struct is_semd_type : std::false_type {};

template <std::ranges::range T>
struct is_semd_type<semd<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_semd_type_v = is_semd_type<T>::value;

template <typename T>
concept is_semd_c = is_semd_type_v<T>;

template <typename T>
concept is_semd = is_semd_c<T> || is_expr_c<T>;
}  // namespace concepts

#define BIN_INFIX_OPERATOR(NAME, OP)                                         \
  namespace ops {                                                            \
  template <typename T, typename U>                                          \
  struct NAME##_f {                                                          \
    static_assert(false, "not implemented for type");                        \
  };                                                                         \
  template <typename T, typename U>                                          \
    requires requires(T a, U b) {                                            \
      { a OP b };                                                            \
    }                                                                        \
  struct NAME##_f<T, U> {                                                    \
    AKS_SEMD_OP_FUNC_PREFIX auto operator()(T const& a, U const& b) const {  \
      return a OP b;                                                         \
    }                                                                        \
  };                                                                         \
  }                                                                          \
  AKS_SEMD_OP_FUNC_PREFIX auto operator OP(concepts::is_semd auto const x,   \
                                           concepts::is_semd auto const y) { \
    /*assert(x.size() == y.size());*/                                        \
    using x_val_t = std::ranges::range_value_t<decltype(x)>;                 \
    using y_val_t = std::ranges::range_value_t<decltype(y)>;                 \
    return expression{ops::NAME##_f<x_val_t, y_val_t>{}, x, y};              \
  }                                                                          \
  AKS_SEMD_OP_FUNC_PREFIX auto operator OP(auto const x,                     \
                                           concepts::is_semd auto const y) { \
    using x_val_t = std::remove_cvref_t<decltype(x)>;                        \
    using y_val_t = std::ranges::range_value_t<decltype(y)>;                 \
    auto const rep = std::ranges::repeat_view{x};                            \
    return expression{ops::NAME##_f<x_val_t, y_val_t>{}, rep, y};            \
  }                                                                          \
  AKS_SEMD_OP_FUNC_PREFIX auto operator OP(concepts::is_semd auto const x,   \
                                           auto const y) {                   \
    using x_val_t = std::ranges::range_value_t<decltype(x)>;                 \
    using y_val_t = std::remove_cvref_t<decltype(y)>;                        \
    auto const rep = std::ranges::repeat_view{y};                            \
    return expression{ops::NAME##_f<x_val_t, y_val_t>{}, x, rep};            \
  }

BIN_INFIX_OPERATOR(plus, +);
BIN_INFIX_OPERATOR(minus, -);
BIN_INFIX_OPERATOR(mul, *);
BIN_INFIX_OPERATOR(div, /);
BIN_INFIX_OPERATOR(mod, %);
BIN_INFIX_OPERATOR(bwand, &);
BIN_INFIX_OPERATOR(bwor, |);
BIN_INFIX_OPERATOR(xor, ^);
BIN_INFIX_OPERATOR(and, &&);
BIN_INFIX_OPERATOR(or, ||);
BIN_INFIX_OPERATOR(eq, ==);
BIN_INFIX_OPERATOR(neq, !=);
BIN_INFIX_OPERATOR(lt, <);
BIN_INFIX_OPERATOR(lte, <=);
BIN_INFIX_OPERATOR(gt, >);
BIN_INFIX_OPERATOR(gte, >=);

#undef BIN_INFIX_OPERATOR

#define BIN_FUNC_OPERATOR(NAME, OP_NAME, OP_APPLY)                          \
  namespace ops {                                                           \
  template <typename T, typename U>                                         \
  struct NAME##_f {                                                         \
    static_assert(false, "not implemented for type");                       \
  };                                                                        \
  template <typename T, typename U>                                         \
    requires requires(T a, U b) {                                           \
      { OP_APPLY(a, b) };                                                   \
    }                                                                       \
  struct NAME##_f<T, U> {                                                   \
    AKS_SEMD_OP_FUNC_PREFIX auto operator()(T const& a, U const& b) const { \
      return OP_APPLY(a, b);                                                \
    }                                                                       \
  };                                                                        \
  }                                                                         \
  AKS_SEMD_OP_FUNC_PREFIX auto OP_NAME(concepts::is_semd auto const x,      \
                                       concepts::is_semd auto const y) {    \
    /*assert(x.size() == y.size());*/                                       \
    using x_val_t = std::ranges::range_value_t<decltype(x)>;                \
    using y_val_t = std::ranges::range_value_t<decltype(y)>;                \
    return expression{ops::NAME##_f<x_val_t, y_val_t>{}, x, y};             \
  }                                                                         \
  AKS_SEMD_OP_FUNC_PREFIX auto OP_NAME(auto const x,                        \
                                       concepts::is_semd auto const y) {    \
    using x_val_t = std::remove_cvref_t<decltype(x)>;                       \
    using y_val_t = std::ranges::range_value_t<decltype(y)>;                \
    auto const rep = std::ranges::repeat_view{x};                           \
    return expression{ops::NAME##_f<x_val_t, y_val_t>{}, rep, y};           \
  }                                                                         \
  AKS_SEMD_OP_FUNC_PREFIX auto OP_NAME(concepts::is_semd auto const x,      \
                                       auto const y) {                      \
    using x_val_t = std::ranges::range_value_t<decltype(x)>;                \
    using y_val_t = std::remove_cvref_t<decltype(y)>;                       \
    auto const rep = std::ranges::repeat_view{y};                           \
    return expression{ops::NAME##_f<x_val_t, y_val_t>{}, x, rep};           \
  }

BIN_FUNC_OPERATOR(pow, pow, std::pow);
BIN_FUNC_OPERATOR(atan2, atan2, std::atan2);

#undef BIN_FUNC_OPERATOR

#define UNARY_PREFIX_OPERATOR(NAME, OP)                                      \
  namespace ops {                                                            \
  template <typename T>                                                      \
  struct NAME##_f {                                                          \
    static_assert(false, "not implemented");                                 \
  };                                                                         \
  template <typename T>                                                      \
    requires requires(T a) {                                                 \
      { OP a };                                                              \
    }                                                                        \
  struct NAME##_f<T> {                                                       \
    auto operator()(T const& a) const {                                      \
      return OP a;                                                           \
    }                                                                        \
  };                                                                         \
  }                                                                          \
  AKS_SEMD_OP_FUNC_PREFIX auto operator OP(concepts::is_semd auto const x) { \
    using x_val_t = std::ranges::range_value_t<decltype(x)>;                 \
    return expression{ops::NAME##_f<x_val_t>{}, x};                          \
  }

UNARY_PREFIX_OPERATOR(neg, -);
UNARY_PREFIX_OPERATOR(not, !);
UNARY_PREFIX_OPERATOR(bwnot, ~);

#undef UNARY_PREFIX_OPERATOR

#define UNARY_FUNC_OPERATOR(NAME, OP_NAME, OP_APPLY)                     \
  namespace ops {                                                        \
  template <typename T>                                                  \
  struct NAME##_f {                                                      \
    static_assert(false, "not implemented");                             \
  };                                                                     \
  template <typename T>                                                  \
    requires requires(T a) {                                             \
      { OP_APPLY(a) };                                                   \
    }                                                                    \
  struct NAME##_f<T> {                                                   \
    auto operator()(T const& a) const {                                  \
      using namespace std;                                               \
      return OP_APPLY(a);                                                \
    }                                                                    \
  };                                                                     \
  }                                                                      \
  AKS_SEMD_OP_FUNC_PREFIX auto OP_NAME(concepts::is_semd auto const x) { \
    using x_val_t = std::ranges::range_value_t<decltype(x)>;             \
    return expression{ops::NAME##_f<x_val_t>{}, x};                      \
  }

UNARY_FUNC_OPERATOR(sin, sin, sin);
UNARY_FUNC_OPERATOR(cos, cos, cos);
UNARY_FUNC_OPERATOR(tan, tan, tan);
UNARY_FUNC_OPERATOR(asin, asin, asin);
UNARY_FUNC_OPERATOR(acos, acos, acos);
UNARY_FUNC_OPERATOR(atan, atan, atan);
UNARY_FUNC_OPERATOR(abs, abs, abs);
UNARY_FUNC_OPERATOR(sqrt, sqrt, sqrt);
UNARY_FUNC_OPERATOR(exp, exp, exp);
UNARY_FUNC_OPERATOR(log, log, log);
UNARY_FUNC_OPERATOR(log10, log10, log10);

#undef UNARY_FUNC_OPERATOR

namespace ops {
template <typename T, typename U, typename V>
struct select_f {
  auto operator()(T x, U y, V z) const { return x ? y : z; }
};
}  // namespace ops
AKS_SEMD_OP_FUNC_PREFIX auto select(concepts::is_semd auto const a,
                                    concepts::is_semd auto const b,
                                    concepts::is_semd auto const c) {
  // assert(a.size() == b.size() && b.size() == c.size());
  return expression{ops::select_f<std::ranges::range_value_t<decltype(a)>,
                                  std::ranges::range_value_t<decltype(b)>,
                                  std::ranges::range_value_t<decltype(c)>>{},
                    a, b, c};
}
AKS_SEMD_OP_FUNC_PREFIX auto select(auto const a,
                                    concepts::is_semd auto const b,
                                    concepts::is_semd auto const c) {
  // assert(b.size() == c.size());
  return expression{ops::select_f<std::remove_cvref_t<decltype(a)>,
                                  std::ranges::range_value_t<decltype(b)>,
                                  std::ranges::range_value_t<decltype(c)>>{},
                    std::ranges::repeat_view{a}, b, c};
}
AKS_SEMD_OP_FUNC_PREFIX auto select(concepts::is_semd auto const a,
                                    auto const b,
                                    concepts::is_semd auto const c) {
  // assert(a.size() == c.size());
  return expression{ops::select_f<std::ranges::range_value_t<decltype(a)>,
                                  std::remove_cvref_t<decltype(b)>,
                                  std::ranges::range_value_t<decltype(c)>>{},
                    a, std::ranges::repeat_view{b}, c};
}
AKS_SEMD_OP_FUNC_PREFIX auto select(concepts::is_semd auto const a,
                                    concepts::is_semd auto const b,
                                    auto const c) {
  // assert(a.size() == b.size());
  return expression{ops::select_f<std::ranges::range_value_t<decltype(a)>,
                                  std::ranges::range_value_t<decltype(b)>,
                                  std::remove_cvref_t<decltype(c)>>{},
                    a, b, std::ranges::repeat_view{c}};
}
AKS_SEMD_OP_FUNC_PREFIX auto select(auto const a,
                                    auto const b,
                                    concepts::is_semd auto const c) {
  return expression{ops::select_f<std::remove_cvref_t<decltype(a)>,
                                  std::remove_cvref_t<decltype(b)>,
                                  std::ranges::range_value_t<decltype(c)>>{},
                    std::ranges::repeat_view{a}, std::ranges::repeat_view{b},
                    c};
}
AKS_SEMD_OP_FUNC_PREFIX auto select(concepts::is_semd auto const a,
                                    auto const b,
                                    auto const c) {
  return expression{ops::select_f<std::ranges::range_value_t<decltype(a)>,
                                  std::remove_cvref_t<decltype(b)>,
                                  std::remove_cvref_t<decltype(c)>>{},
                    a, std::ranges::repeat_view{b},
                    std::ranges::repeat_view{c}};
}
AKS_SEMD_OP_FUNC_PREFIX auto select(auto const a,
                                    concepts::is_semd auto const b,
                                    auto const c) {
  return expression{ops::select_f<std::remove_cvref_t<decltype(a)>,
                                  std::ranges::range_value_t<decltype(b)>,
                                  std::remove_cvref_t<decltype(c)>>{},
                    std::ranges::repeat_view{a}, b,
                    std::ranges::repeat_view{c}};
}

#undef AKS_SEMD_OP_FUNC_PREFIX

}  // namespace semd
}  // namespace aks

#endif  // !AKS_SEMD_HPP
