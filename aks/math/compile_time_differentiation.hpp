#pragma once

//-std=c++20 -O3

#include <tuple>

namespace aks {
namespace compile_time_diff {

namespace detail {
using real = double;

template <std::size_t N, typename T, typename... Xs>
constexpr auto nth(T x, Xs... xs) {
  if constexpr (N == 0) {
    return x;
  } else {
    return nth<N - 1>(xs...);
  }
}

template <std::size_t N> struct var {
  static constexpr std::size_t id = N;
  static constexpr std::size_t arity = N + 1;
  using type = var<N>;
  template <typename... Ts> real value(Ts... xs) const {
    static_assert(sizeof...(Ts) >= arity,
                  "trying to get index, but not enough arguments");
    // return xs...[N]; --> C++26
    return nth<N>(xs...);
  }

  template <typename... Ts> real operator()(Ts... xs) const {
    return value(xs...);
  }
};

template <typename T> struct is_var_s : std::bool_constant<false> {};
template <std::size_t N> struct is_var_s<var<N>> : std::bool_constant<true> {};
template <typename T> constexpr bool is_var_v = is_var_s<T>::value;
template <typename T>
concept is_var = is_var_v<T>;

struct constant {
  static constexpr std::size_t arity = 0;
  using type = constant;
  constant(real v) : v_(v) {}
  real v_;
  template <typename... Ts> real value(Ts...) const { return v_; }

  template <typename... Ts> real operator()(Ts...) const { return value(); }
};

template <typename T> struct is_constant_s : std::bool_constant<false> {};
template <> struct is_constant_s<constant> : std::bool_constant<true> {};
template <typename T> constexpr bool is_constant_v = is_constant_s<T>::value;
template <typename T>
concept is_constant = is_constant_v<T>;

struct op_neg {
  template <typename... Ts, typename... Xs>
  real value(std::tuple<Ts...> ts, Xs... xs) const {
    return -std::get<0>(ts).value(xs...);
  }
};
struct op_add {
  template <typename... Ts, typename... Xs>
  real value(std::tuple<Ts...> ts, Xs... xs) const {
    return std::get<0>(ts).value(xs...) + std::get<1>(ts).value(xs...);
  }
};
struct op_sub {
  template <typename... Ts, typename... Xs>
  real value(std::tuple<Ts...> ts, Xs... xs) const {
    return std::get<0>(ts).value(xs...) - std::get<1>(ts).value(xs...);
  }
};
struct op_mul {
  template <typename... Ts, typename... Xs>
  real value(std::tuple<Ts...> ts, Xs... xs) const {
    return std::get<0>(ts).value(xs...) * std::get<1>(ts).value(xs...);
  }
};
struct op_div {
  template <typename... Ts, typename... Xs>
  real value(std::tuple<Ts...> ts, Xs... xs) const {
    return std::get<0>(ts).value(xs...) / std::get<1>(ts).value(xs...);
  }
};
struct op_lt {
  // bool value(real l, real r) const { return l < r; }

  template <typename... Ts, typename... Xs>
  bool value(std::tuple<Ts...> ts, Xs... xs) const {
    return std::get<0>(ts).value(xs...) < std::get<1>(ts).value(xs...);
  }
};
struct op_gt {
  // bool value(real l, real r) const { return l > r; }

  template <typename... Ts, typename... Xs>
  bool value(std::tuple<Ts...> ts, Xs... xs) const {
    return std::get<0>(ts).value(xs...) > std::get<1>(ts).value(xs...);
  }
};

template <typename... Xs, typename T> real compose(std::tuple<Xs...> xs, T t) {
  auto apply = [&](Xs... vs) { return t.value(vs...); };
  return std::apply(apply, xs);
}

template <typename... Xs, typename... Ts, typename T>
real compose(std::tuple<Xs...> xs, T t, Ts... ts) {
  return t.value(compose(xs, ts...));
}

template <typename... Xs, typename T> real compose2(real r, T t) {
  return t.value(r);
}

template <typename... Xs, typename... Ts, typename T>
real compose2(real r, T t, Ts... ts) {
  return compose2(t.value(r), ts...);
}

template <typename... Xs, typename T, typename... Ts>
real compose2(std::tuple<Xs...> xs, T t, Ts... ts) {
  auto apply = [&](Xs... vs) { return t.value(vs...); };
  return compose2(std::apply(apply, xs), ts...);
}

struct op_compose {
  template <typename... Ts, typename... Xs>
  real value(std::tuple<Ts...> ts, Xs... xs) const {
    auto apply = [&](Ts... vs) {
      return compose2(std::make_tuple(xs...), vs...);
    };
    return std::apply(apply, ts);
  }
};

template <typename T, typename... Ts> struct max_arity {
  static constexpr std::size_t arity =
      T::arity > max_arity<Ts...>::arity ? T::arity : max_arity<Ts...>::arity;
};

template <typename T> struct max_arity<T> {
  static constexpr std::size_t arity = T::arity;
};

template <typename OP, typename... Ts> struct expr {
  static constexpr std::size_t arity = max_arity<Ts...>::arity;
  expr(OP op, Ts... ts) : ts_(ts...), op_(op) {}
  std::tuple<Ts...> ts_;
  using type = std::tuple<Ts...>;
  using op_type = OP;
  OP op_;

  template <typename... Xs> real value(Xs... xs) const {
    static_assert(sizeof...(Xs) >= arity,
                  "trying to get index, but not enough arguments");
    // auto apply = [&](Ts... vs) { return op_.value(vs.value(xs...)...); };
    // return std::apply(apply, ts_);
    return op_.value(ts_, xs...);
  }

  template <typename... Xs> real operator()(Xs... xs) const {
    return value(xs...);
  }
};

template <typename T> struct is_expr_s : std::bool_constant<false> {};
template <typename OP, typename... Ts>
struct is_expr_s<expr<OP, Ts...>> : std::bool_constant<true> {};
template <typename T> constexpr bool is_expr_v = is_expr_s<T>::value;
template <typename T>
concept is_expr = is_expr_v<T>;

template <typename T>
concept is_value = is_expr<T> || is_constant<T> || is_var<T>;

template <typename OP, typename... Ts> struct cexpr {
  static constexpr std::size_t arity = max_arity<Ts...>::arity;
  cexpr(OP op, Ts... ts) : ts_(ts...), op_(op) {}
  std::tuple<Ts...> ts_;
  using type = std::tuple<Ts...>;
  using op_type = OP;
  OP op_;

  template <typename... Xs> bool value(Xs... xs) const {
    static_assert(sizeof...(Xs) >= arity,
                  "trying to get index, but not enough arguments");
    // auto apply = [&](Ts... vs) { return op_.value(vs.value(xs...)...); };
    // return std::apply(apply, ts_);
    return op_.value(ts_, xs...);
  }
};

template <typename T> struct is_cexpr_s : std::bool_constant<false> {};
template <typename OP, typename... Ts>
struct is_cexpr_s<cexpr<OP, Ts...>> : std::bool_constant<true> {};
template <typename T> constexpr bool is_cexpr_v = is_cexpr_s<T>::value;
template <typename T>
concept is_cexpr = is_cexpr_v<T>;

template <typename C, typename E0, typename E1> struct if_ {
  C c_;
  E0 e0_;
  E1 e1_;
  static constexpr std::size_t arity = max_arity<C, E0, E1>::arity;

  template <typename... Xs> real value(Xs... xs) const {
    static_assert(sizeof...(Xs) >= arity,
                  "trying to get index, but not enough arguments");
    return c_.value(xs...) ? e0_.value(xs...) : e1_.value(xs...);
  }
};

template <typename T> struct is_cond_s : std::bool_constant<false> {};
template <typename C, typename E0, typename E1>
struct is_cond_s<if_<C, E0, E1>> : std::bool_constant<true> {};
template <typename T> constexpr bool is_cond_v = is_cond_s<T>::value;
template <typename T>
concept is_cond = is_cond_v<T>;

template <typename T>
concept is_conditional = is_cond<T> || is_cexpr<T>;

auto if_else(is_conditional auto c, auto e0, auto e1) { return if_{c, e0, e1}; }

template <std::size_t N, typename OP, typename... Ts>
auto get_op(expr<OP, Ts...> ex) {
  return std::get<N>(ex.ts_);
}

template <typename T>
concept is_any_type = is_conditional<T> || is_value<T>;

auto operator-(is_value auto x) { return expr{op_neg(), x}; }
auto operator+(is_value auto x, is_value auto y) {
  return expr{op_add(), x, y};
}
auto operator-(is_value auto x, is_value auto y) {
  return expr{op_sub(), x, y};
}
auto operator*(is_value auto x, is_value auto y) {
  return expr{op_mul(), x, y};
}
auto operator/(is_value auto x, is_value auto y) {
  return expr{op_div(), x, y};
}
auto operator<(is_value auto x, is_value auto y) {
  return cexpr{op_lt(), x, y};
}
auto operator>(is_value auto x, is_value auto y) {
  return cexpr{op_gt(), x, y};
}
auto operator>>(is_value auto x, is_value auto y) {
  static_assert(decltype(y)::arity == 1,
                "cannot compose function of multiple variables");
  return expr{op_compose(), x, y};
}

auto operator-(is_cond auto x) { return if_{x.c_, -x.e0_, -x.e1_}; }
auto operator+(is_cond auto x, is_value auto y) {
  return if_{x.c_, x.e0_ + y, x.e1_ + y};
}
auto operator-(is_cond auto x, is_value auto y) {
  return if_{x.c_, x.e0_ - y, x.e1_ - y};
}
auto operator*(is_cond auto x, is_value auto y) {
  return if_{x.c_, x.e0_ * y, x.e1_ * y};
}
auto operator/(is_cond auto x, is_value auto y) {
  return if_{x.c_, x.e0_ / y, x.e1_ / y};
}
auto operator<(is_cond auto x, is_value auto y) {
  return if_{x.c_, x.e0_ < y, x.e1_ < y};
}
auto operator>(is_cond auto x, is_value auto y) {
  return if_{x.c_, x.e0_ > y, x.e1_ > y};
}
auto operator+(is_value auto y, is_cond auto x) {
  return if_{x.c_, y + x.e0_, y + x.e1_};
}
auto operator-(is_value auto y, is_cond auto x) {
  return if_{x.c_, y - x.e0_, y - x.e1_};
}
auto operator*(is_value auto y, is_cond auto x) {
  return if_{x.c_, y * x.e0_, y * x.e1_};
}
auto operator/(is_value auto y, is_cond auto x) {
  return if_{x.c_, y / x.e0_, y / x.e1_};
}
auto operator<(is_value auto y, is_cond auto x) {
  return if_{x.c_, y < x.e0_, y < x.e1_};
}
auto operator>(is_value auto y, is_cond auto x) {
  return if_{x.c_, y > x.e0_, y > x.e1_};
}

auto operator+(is_cond auto x, is_cond auto y) {
  return if_{x.c_, if_{y.c_, x.e0_ + y.e0_, x.e0_ + y.e1_},
             if_{y.c_, x.e1_ + y.e0_, x.e1_ + y.e1_}};
}
auto operator-(is_cond auto x, is_cond auto y) {
  return if_{x.c_, if_{y.c_, x.e0_ - y.e0_, x.e0_ - y.e1_},
             if_{y.c_, x.e1_ - y.e0_, x.e1_ - y.e1_}};
}
auto operator*(is_cond auto x, is_cond auto y) {
  return if_{x.c_, if_{y.c_, x.e0_ * y.e0_, x.e0_ * y.e1_},
             if_{y.c_, x.e1_ * y.e0_, x.e1_ * y.e1_}};
}
auto operator/(is_cond auto x, is_cond auto y) {
  return if_{x.c_, if_{y.c_, x.e0_ / y.e0_, x.e0_ / y.e1_},
             if_{y.c_, x.e1_ / y.e0_, x.e1_ / y.e1_}};
}
auto operator<(is_cond auto x, is_cond auto y) {
  return if_{x.c_, if_{y.c_, x.e0_ < y.e0_, x.e0_ < y.e1_},
             if_{y.c_, x.e1_ < y.e0_, x.e1_ < y.e1_}};
}
auto operator>(is_cond auto x, is_cond auto y) {
  return if_{x.c_, if_{y.c_, x.e0_ > y.e0_, x.e0_ > y.e1_},
             if_{y.c_, x.e1_ > y.e0_, x.e1_ > y.e1_}};
}

auto operator+(is_value auto x, real y) {
  return expr{op_add(), x, constant(y)};
}
auto operator-(is_value auto x, real y) {
  return expr{op_sub(), x, constant(y)};
}
auto operator*(is_value auto x, real y) {
  return expr{op_mul(), x, constant(y)};
}
auto operator/(is_value auto x, real y) {
  return expr{op_div(), x, constant(y)};
}
auto operator<(is_value auto x, real y) {
  return cexpr{op_lt(), x, constant(y)};
}
auto operator>(is_value auto x, real y) {
  return cexpr{op_gt(), x, constant(y)};
}

auto operator+(real x, is_value auto y) {
  return expr{op_add(), constant(x), y};
}
auto operator-(real x, is_value auto y) {
  return expr{op_sub(), constant(x), y};
}
auto operator*(real x, is_value auto y) {
  return expr{op_mul(), constant(x), y};
}
auto operator/(real x, is_value auto y) {
  return expr{op_div(), constant(x), y};
}
auto operator<(real x, is_value auto y) {
  return cexpr{op_lt(), constant(x), y};
}
auto operator>(real x, is_value auto y) {
  return cexpr{op_gt(), constant(x), y};
}
//
// template <typename X, typename A, typename B>
// struct repl_helper {
//  using type = std::conditional_t<std::is_same_v<X, A>, B, A>;
//};
//
// template <typename X, typename A, typename B>
// using repl_helper_t = typename repl_helper<X, A, B>::type;
//
//// replace X in A with B
// template <typename X, typename A, typename B>
// struct replace;
//
// template <typename X, typename A, typename B>
// using replace_t = typename replace<X, A, B>::type;
//
// template <is_var X, is_var A, is_value B>
// struct replace<X, A, B> {
//   using type = repl_helper_t<X, A, B>;
// };
//
// template <is_var X, typename OP, typename E0, is_value B>
// struct replace<X, expr<OP, E0>, B> {
//   using type = expr<OP, replace_t<X, E0, B>>;
// };
//
// template <is_var X, typename OP, typename E0, typename E1, is_value B>
// struct replace<X, expr<OP, E0, E1>, B> {
//   using type = expr<OP, replace_t<X, E0, B>, replace_t<X, E1, B>>;
// };

template <is_var X, is_value A, is_value B> auto substitute(X x, A a, B b) {
  if constexpr (std::is_same_v<X, A>) {
    return b;
  } else if constexpr (is_expr_v<A>) {
    auto apply = [&](auto... vs) {
      return expr{a.op_, substitute(x, vs, b)...};
    };
    return std::apply(apply, a.ts_);
  } else {
    return a;
  }
}

template <std::size_t N> auto d_wrt(var<N> e, var<N> v) {
  return constant{1.0};
}

template <std::size_t N> auto d_wrt(constant e, var<N> v) {
  return constant{0.0};
}

template <std::size_t N, std::size_t M> auto d_wrt(var<N> e, var<M> v) {
  return constant{0.0};
}

template <typename T0, std::size_t N>
auto d_wrt(expr<op_neg, T0> ex, var<N> v) {
  return -d_wrt(get_op<0>(ex), v);
}

template <typename T0, typename T1, std::size_t N>
auto d_wrt(expr<op_add, T0, T1> ex, var<N> v) {
  return d_wrt(get_op<0>(ex), v) + d_wrt(get_op<1>(ex), v);
}

template <typename T0, typename T1, std::size_t N>
auto d_wrt(expr<op_sub, T0, T1> ex, var<N> v) {
  return d_wrt(get_op<0>(ex), v) - d_wrt(get_op<1>(ex), v);
}

template <typename T0, typename T1, std::size_t N>
auto d_wrt(expr<op_mul, T0, T1> ex, var<N> v) {
  return (d_wrt(get_op<0>(ex), v) * get_op<1>(ex)) +
         (get_op<0>(ex) * d_wrt(get_op<1>(ex), v));
}

template <typename T0, typename T1, std::size_t N>
auto d_wrt(expr<op_div, T0, T1> ex, var<N> v) {
  auto f = get_op<0>(ex);
  auto g = get_op<1>(ex);
  auto f_p = d_wrt(f, v);
  auto g_p = d_wrt(g, v);
  return ((f_p * g) - (g_p * f)) / (g * g);
}

auto d_wrt(is_cond auto c, is_var auto v) {
  return if_{c.c_, d_wrt(c.e0_, v), d_wrt(c.e1_, v)};
}

constexpr std::size_t arity(is_any_type auto x) { return decltype(x)::arity; }
} // namespace detail

using detail::constant;
// using detail::d_wrt;
using detail::real;
using detail::var;

} // namespace compile_time_diff
} // namespace aks

#if 0

#include <functional>
#include <iostream>
#include <vector>

// useful to get a compiler error showing the type being passed
void err(char c) {}

using namespace std;

void pp(aks::compile_time_diff::real f, const char *s = "",
        const char *p = "") {
  printf("%s%.16f%s", p, f, s);
}

void pp(bool f, const char *s = "", const char *p = "") {
  printf("%s%s%s", p, (f ? "true" : "false"), s);
}

void pp(std::size_t f, const char *s = "", const char *p = "") {
  printf("%s%zu%s", p, f, s);
}

void examples() {
  // proof of concept, only support simple operations (-,+,-,*,/,<,>)

  using X = aks::compile_time_diff::var<0>;
  using Y = aks::compile_time_diff::var<1>;
  using Z = aks::compile_time_diff::var<2>;
  using W = aks::compile_time_diff::var<3>;
  using C = aks::compile_time_diff::constant;
  using real = aks::compile_time_diff::real; // this is a double

  auto x = X{};
  auto y = Y{};
  auto z = Z{};
  auto w = W{};

  {
    // general usage
    {
      // to get the value of the function given the inputs, call value
      // explicitly
      pp(x.value(2.0), "\n");                // prints 2.0
      pp(y.value(2.0, 3.0), "\n");           // prints 3.0
      pp(z.value(2.0, 3.0, 4.0), "\n");      // prints 4.0
      pp(w.value(2.0, 3.0, 4.0, 5.0), "\n"); // prints 5.0
    }
    {
      // or use the function interface
      pp(x(2.0), "\n");                // prints 2.0
      pp(y(2.0, 3.0), "\n");           // prints 3.0
      pp(z(2.0, 3.0, 4.0), "\n");      // prints 4.0
      pp(w(2.0, 3.0, 4.0, 5.0), "\n"); // prints 5.0
    }
    {
      // need to provide for all the var<N-1> if the equation has var<N>
      // does not compile as y needs atleast an 2 arguments (1 given)
      //
      // variable const auto y__3_0 = y.value(3.0);
      //
    }
    {
      // create functions from variables
      auto f0 = x * x;
      pp(f0.value(2.0), "\n"); // prints 4.0 using value explicitly
      pp(f0(2.0), "\n");       // prints 4.0 using function interface
    }
    {
      // use constants
      auto f0 = 2.0 * x;
      pp(f0.value(3.0), "\n"); // prints 6.0
    }
    {
      // the value can be called in place
      pp((x * x).value(2.0), "\n"); // prints 4.0
      pp((x * x)(2.0), "\n");       // prints 4.0
    }
    {
      auto f0 = x * y;              // y is var<1>, so it requires 2 arguments.
      pp(f0.value(2.0, 3.0), "\n"); // prints 6.0
    }
    {
      auto f0 = 3.0 * x / y;
      pp(f0.value(2.0, 3.0), "\n"); // prints 2.0
    }
    {
      auto f = (x * y) / (z + w);
      pp(f.value(3.0, 5.0, 1.0, 2.0), "\n"); // prints 5.0
    }
    {
      // functions can be built from functions;
      auto f0 = x * y;
      auto f1 = z + w;
      auto f = f0 / f1;
      pp(f.value(3.0, 5.0, 1.0, 2.0), "\n"); // prints 5.0
    }
    {
      // conditions can be added
      auto f = if_else(x > y, x, y);
      pp(f.value(3.0, 5.0), "\n"); // prints 5.0
      pp(f.value(5.0, 2.0), "\n"); // prints 5.0
    }
    {
      // conditions can be used as part of other functions
      auto f = (x * y) + if_else(z < w, z * z, w * w);
      pp(f.value(3.0, 5.0, 1.0, 2.0), "\n"); // prints 16.0
      pp(f.value(3.0, 5.0, 3.0, 2.0), "\n"); // prints 19.0
    }
    {
      // the condition can be used to implement other sub functions
      auto zw_min = if_else(z < w, z, w);
      auto f = (x * y) + zw_min * zw_min;
      pp(f.value(3.0, 5.0, 1.0, 2.0), "\n"); // prints 16.0
      pp(f.value(3.0, 5.0, 3.0, 2.0), "\n"); // prints 19.0
    }
    {
      // create multiple branches
      auto xyz_min = if_else(z < if_else(x < y, x, y), z, if_else(x < y, x, y));
      pp(xyz_min.value(3.0, 5.0, 1.0), "\n"); // prints 1.0
      pp(xyz_min.value(1.0, 5.0, 3.0), "\n"); // prints 1.0
      pp(xyz_min.value(5.0, 1.0, 3.0), "\n"); // prints 1.0
    }
    {
      // functions of only var<0> or x can be composed using >>
      auto f = 2.0 + x;
      auto g = 3.0 * x;

      auto h = f >> g; // calculate f and then pass value to g
      // so h = g ( f ( x ) )
      pp(h.value(2.0), "\n"); // prints 12.0
    }
    {
      // substitute a variable in a function with constant
      auto f = x * y;
      auto g = substitute(y, f, C{8.0}); // substitute y in f with constant 8

      pp(f.value(2.0, 3.0), "\n"); // prints 6.0 - takes 2 args as y present
      pp(g.value(2.0), "\n"); // prints 16.0, takes only 1 arg as y not present
    }
    {
      // substitute a variable in a function with another variable
      auto f = x * y;
      auto g = substitute(x, f, z); // substitute x in f with z

      pp(f.value(2.0, 3.0),
         "\n"); // prints 6.0 - takes only 2 args as no z yet
      pp(g.value(2.0, 3.0, 5.0), "\n"); // prints 15.0 - 3 args as z introduced
    }
    {
      // substitute a variable in a function with another function
      auto f = x * y;
      auto g = substitute(x, f, y * y); // substitute x in f with (y*y)
      pp(f.value(2.0, 3.0), "\n");      // prints 6.0
      pp(g.value(2.0, 3.0), "\n");      // prints 27.0
    }
  }
  { // compile time derivatives
    {
      // take derivatives
      auto f = x * x * x;
      auto df_dx = d_wrt(f, x);   // derivate or f with respect to x
      pp(f.value(2.0), "\n");     // prints 8.0
      pp(df_dx.value(2.0), "\n"); // prints 12.0
    }
    {
      // multi variable derivatives
      auto f = x * y * z;
      auto df_dx = d_wrt(f, x);
      auto df_dy = d_wrt(f, y);
      auto df_dz = d_wrt(f, z);
      pp(f.value(2.0, 3.0, 4.0), "\n");     // prints 24.0
      pp(df_dx.value(2.0, 3.0, 4.0), "\n"); // prints 12.0
      pp(df_dy.value(2.0, 3.0, 4.0), "\n"); // prints 8.0
      pp(df_dz.value(2.0, 3.0, 4.0), "\n"); // print 6.0
    }
    {
      // take higher order derivatives
      auto f = x * x * x;
      auto df_dx = d_wrt(f, x);
      auto d2f_dx2 = d_wrt(df_dx, x);
      pp(f.value(5.0), "\n");       // prints 125.0
      pp(df_dx.value(5.0), "\n");   // prints 75.0
      pp(d2f_dx2.value(5.0), "\n"); // prints 30.0
    }
    {
      // take higher order derivatives with multiple variables
      auto f = x * x * y * y * z * z;
      auto df_dx = d_wrt(f, x);
      auto d2f_dxdy = d_wrt(df_dx, y);
      auto d3f_dxdydz = d_wrt(d2f_dxdy, z);
      pp(f.value(5.0, 3.0, 4.0), "\n");          // prints 3600.0
      pp(df_dx.value(5.0, 3.0, 4.0), "\n");      // prints 1440.0
      pp(d2f_dxdy.value(5.0, 3.0, 4.0), "\n");   // prints 960.0
      pp(d3f_dxdydz.value(5.0, 3.0, 4.0), "\n"); // prints 480.0
    }
  }
  {
    // example, a simple newton solver

    auto solve = [x](auto f, real guess) {
      auto newton_step = -f / d_wrt(f, x);

      for (int i = 0; i < 10; i++) {
        auto dx = newton_step(guess);
        guess = guess + dx;
        if (std::abs(dx) < 1e-8)
          break;
      }
      return guess;
    };

    {
      pp(solve(x * x - 9.0, 2.0), "\n");      // prints 3.0
      pp(solve(x * x * x - 27.0, 2.0), "\n"); // prints 3.0
    }
  }
}

int main() {
  examples();
  return 0;
}

#endif