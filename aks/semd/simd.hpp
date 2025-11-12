
#ifndef AKS_SIMD_COMMON_HPP__
#define AKS_SIMD_COMMON_HPP__

#include <concepts>
#include <type_traits>

namespace aks {

namespace simd {

#define AKS_SIMD_OP_FUNC_PREFIX [[nodiscard]] inline

struct zero_t {};

constexpr zero_t const zero_v{};

template <typename simd_specification_type_>
struct vec {
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

  template <typename... Args>
  vec(Args... args) : data_({args...}) {
    static_assert(sizeof...(Args) == size_, "invalid number of values");
  }

  explicit vec(simd_type v) : data_({v}) {}

  template <typename scalar_type_>
  explicit vec(scalar_type_ v) : data_({spec::set1(v)}) {}

  explicit vec(zero_t) : data_({spec::setzero()}) {}

  operator simd_type() const { return data_.simd_; }

  scalar_type const& at(std::size_t i) const { return data_.scalar_[i]; }
  scalar_type& at(std::size_t i) { return data_.scalar_[i]; }

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

template <typename simd_specification_type_>
struct mask {
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

}  // namespace simd

}  // namespace aks

#endif  // !AKS_COMMON_HPP

#ifndef AKS_SIMD_ONE_VALUE_HPP__
#define AKS_SIMD_ONE_VALUE_HPP__

#include <array>
#include <cmath>
#include <type_traits>

namespace aks {
namespace simd {

template <typename T>
struct one_value_spec {
  using simd_type = T;
  using mask_type = bool;
  using scalar_type = T;
  static constexpr std::size_t size = 1;

  AKS_SIMD_OP_FUNC_PREFIX static simd_type set1(scalar_type v) { return v; }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type setzero() { return scalar_type(0); }

  AKS_SIMD_OP_FUNC_PREFIX static simd_type add(simd_type a, simd_type b) {
    return a + b;
  }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type sub(simd_type a, simd_type b) {
    return a - b;
  }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type mul(simd_type a, simd_type b) {
    return a * b;
  }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type div(simd_type a, simd_type b) {
    return a / b;
  }

  AKS_SIMD_OP_FUNC_PREFIX static simd_type neg(simd_type x) { return -x; }

  AKS_SIMD_OP_FUNC_PREFIX static simd_type sqrt(simd_type a) {
    return std::sqrt(a);
  }

  AKS_SIMD_OP_FUNC_PREFIX static simd_type abs(simd_type a) {
    return std::abs(a);
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type eq(simd_type a, simd_type b) {
    return a == b;
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type neq(simd_type a, simd_type b) {
    return a != b;
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type lt(simd_type a, simd_type b) {
    return a < b;
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type lte(simd_type a, simd_type b) {
    return a <= b;
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type gt(simd_type a, simd_type b) {
    return a > b;
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type gte(simd_type a, simd_type b) {
    return a >= b;
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type and_(mask_type a, mask_type b) {
    return a && b;
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type or_(mask_type a, mask_type b) {
    return a || b;
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type xor_(mask_type a, mask_type b) {
    return a != b;
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type not_(mask_type a) { return !a; }

  AKS_SIMD_OP_FUNC_PREFIX static auto simd_as_array(simd_type a) {
    std::array<scalar_type, size> result;
    result[0] = a;
    return result;
  }

  AKS_SIMD_OP_FUNC_PREFIX static auto mask_as_array(mask_type a) {
    std::array<bool, size> result;
    result[0] = a;
    return result;
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type from_bools(bool a0) {
    return mask_type(a0);
  }

  AKS_SIMD_OP_FUNC_PREFIX static simd_type select(mask_type a,
                                                  simd_type b,
                                                  simd_type c) {
    return a ? b : c;
  }

  AKS_SIMD_OP_FUNC_PREFIX static bool any(mask_type a) { return a; }

  AKS_SIMD_OP_FUNC_PREFIX static bool all(mask_type a) { return a; }
};
using vec1f = simd::vec<one_value_spec<float>>;
using vec1f_mask = simd::mask<one_value_spec<float>>;
using vec1d = simd::vec<one_value_spec<double>>;
using vec1d_mask = simd::mask<one_value_spec<double>>;
using vec1i = simd::vec<one_value_spec<int>>;
using vec1i_mask = simd::mask<one_value_spec<int>>;
}  // namespace simd
}  // namespace aks

#endif  // !AKS_SIMD_ONE_VALUE_HPP__

#ifndef AKS_SIMD_VEC4D_HPP__
#define AKS_SIMD_VEC4D_HPP__

#include <immintrin.h>
#include <array>
#include <type_traits>

namespace aks {
namespace simd {
namespace avx {
namespace detail_vec4d {
static auto const zeros{_mm256_setzero_pd()};
static auto const neg_zeros{_mm256_set1_pd(-0.0)};
}  // namespace detail_vec4d

struct vec4d_spec {
  using simd_type = __m256d;
  using mask_type = __m256d;
  using scalar_type = double;
  static constexpr std::size_t size = 4;

  AKS_SIMD_OP_FUNC_PREFIX static simd_type set1(scalar_type v) {
    return _mm256_set1_pd(v);
  }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type setzero() {
    return _mm256_setzero_pd();
  }

  AKS_SIMD_OP_FUNC_PREFIX static simd_type add(simd_type a, simd_type b) {
    return _mm256_add_pd(a, b);
  }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type sub(simd_type a, simd_type b) {
    return _mm256_sub_pd(a, b);
  }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type mul(simd_type a, simd_type b) {
    return _mm256_mul_pd(a, b);
  }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type div(simd_type a, simd_type b) {
    return _mm256_div_pd(a, b);
  }

  AKS_SIMD_OP_FUNC_PREFIX static simd_type neg(simd_type x) {
    return _mm256_xor_pd(x, detail_vec4d::neg_zeros);
  }

  AKS_SIMD_OP_FUNC_PREFIX static simd_type sqrt(simd_type a) {
    return _mm256_sqrt_pd(a);
  }

  AKS_SIMD_OP_FUNC_PREFIX static simd_type abs(simd_type a) {
    return _mm256_andnot_pd(detail_vec4d::neg_zeros, a);
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type eq(simd_type a, simd_type b) {
    return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type neq(simd_type a, simd_type b) {
    return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ);
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type lt(simd_type a, simd_type b) {
    return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type lte(simd_type a, simd_type b) {
    return _mm256_cmp_pd(a, b, _CMP_LE_OQ);
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type gt(simd_type a, simd_type b) {
    return _mm256_cmp_pd(a, b, _CMP_GT_OQ);
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type gte(simd_type a, simd_type b) {
    return _mm256_cmp_pd(a, b, _CMP_GE_OQ);
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type and_(mask_type a, mask_type b) {
    return _mm256_and_pd(a, b);
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type or_(mask_type a, mask_type b) {
    return _mm256_or_pd(a, b);
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type xor_(mask_type a, mask_type b) {
    return _mm256_xor_pd(a, b);
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type not_(mask_type a) {
    return _mm256_xor_pd(a, detail_vec4d::zeros);
  }

  AKS_SIMD_OP_FUNC_PREFIX static auto simd_as_array(simd_type a) {
    std::array<scalar_type, size> result;
    _mm256_storeu_pd(result.data(), a);
    return result;
  }

  AKS_SIMD_OP_FUNC_PREFIX static auto mask_as_array(mask_type a) {
    std::array<bool, size> result;
    auto mvmsk = _mm256_movemask_pd(a);
    for (int i = 0; i < size; i++) {
      result[i] = mvmsk & (1 << i);
    }
    return result;
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type from_bools(bool a0,
                                                      bool a1,
                                                      bool a2,
                                                      bool a3) {
    return _mm256_castsi256_pd(_mm256_set_epi64x((a3 ? -1 : 0), (a2 ? -1 : 0),
                                                 (a1 ? -1 : 0), (a0 ? -1 : 0)));
  }

  AKS_SIMD_OP_FUNC_PREFIX static simd_type select(mask_type a,
                                                  simd_type b,
                                                  simd_type c) {
    return _mm256_blendv_pd(c, b, a);
  }

  AKS_SIMD_OP_FUNC_PREFIX static bool any(mask_type a) {
    return _mm256_movemask_pd(a) != 0;
  }

  AKS_SIMD_OP_FUNC_PREFIX static bool all(mask_type a) {
    return _mm256_movemask_pd(a) == 0xf;
  }
};
}  // namespace avx
using vec4d = simd::vec<avx::vec4d_spec>;
using vec4d_mask = simd::mask<avx::vec4d_spec>;
}  // namespace simd
}  // namespace aks

#endif  // !AKS_SIMD_VEC4D_HPP__

#ifndef AKS_SIMD_VEC8F_HPP__
#define AKS_SIMD_VEC8F_HPP__

namespace aks {
namespace simd {
namespace avx {
namespace detail_vec8f {
static auto const zeros{_mm256_setzero_ps()};
static auto const neg_zeros{_mm256_set1_ps(-0.f)};
}  // namespace detail_vec8f

struct vec8f_spec {
  using simd_type = __m256;
  using mask_type = __m256;
  using scalar_type = float;
  static constexpr std::size_t size = 8;
  AKS_SIMD_OP_FUNC_PREFIX static simd_type set1(scalar_type v) {
    return _mm256_set1_ps(v);
  }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type setzero() {
    return _mm256_setzero_ps();
  }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type add(simd_type a, simd_type b) {
    return _mm256_add_ps(a, b);
  }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type sub(simd_type a, simd_type b) {
    return _mm256_sub_ps(a, b);
  }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type mul(simd_type a, simd_type b) {
    return _mm256_mul_ps(a, b);
  }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type div(simd_type a, simd_type b) {
    return _mm256_div_ps(a, b);
  }
  AKS_SIMD_OP_FUNC_PREFIX static simd_type neg(simd_type x) {
    return _mm256_xor_ps(x, detail_vec8f::neg_zeros);
  };
  AKS_SIMD_OP_FUNC_PREFIX static simd_type sqrt(simd_type x) {
    return _mm256_sqrt_ps(x);
  };
  AKS_SIMD_OP_FUNC_PREFIX static simd_type abs(simd_type x) {
    return _mm256_andnot_ps(detail_vec8f::neg_zeros, x);
  };
  AKS_SIMD_OP_FUNC_PREFIX static mask_type eq(simd_type a, simd_type b) {
    return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
  };
  AKS_SIMD_OP_FUNC_PREFIX static mask_type ne(simd_type a, simd_type b) {
    return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);
  };
  AKS_SIMD_OP_FUNC_PREFIX static mask_type lt(simd_type a, simd_type b) {
    return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
  };
  AKS_SIMD_OP_FUNC_PREFIX static mask_type le(simd_type a, simd_type b) {
    return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
  };
  AKS_SIMD_OP_FUNC_PREFIX static mask_type gt(simd_type a, simd_type b) {
    return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
  };
  AKS_SIMD_OP_FUNC_PREFIX static mask_type ge(simd_type a, simd_type b) {
    return _mm256_cmp_ps(a, b, _CMP_GE_OQ);
  }
  AKS_SIMD_OP_FUNC_PREFIX static mask_type and_(mask_type a, mask_type b) {
    return _mm256_and_ps(a, b);
  }
  AKS_SIMD_OP_FUNC_PREFIX static mask_type or_(mask_type a, mask_type b) {
    return _mm256_or_ps(a, b);
  }
  AKS_SIMD_OP_FUNC_PREFIX static mask_type xor_(mask_type a, mask_type b) {
    return _mm256_xor_ps(a, b);
  }
  AKS_SIMD_OP_FUNC_PREFIX static mask_type not_(mask_type a) {
    return _mm256_xor_ps(a, detail_vec8f::zeros);
  }

  AKS_SIMD_OP_FUNC_PREFIX static auto simd_as_array(simd_type a) {
    std::array<scalar_type, size> result;
    _mm256_storeu_ps(result.data(), a);
    return result;
  }

  AKS_SIMD_OP_FUNC_PREFIX static auto mask_as_array(mask_type a) {
    std::array<bool, size> result;
    auto mvmsk = _mm256_movemask_ps(a);
    for (int i = 0; i < size; i++) {
      result[i] = mvmsk & (1 << i);
    }
    return result;
  }

  AKS_SIMD_OP_FUNC_PREFIX static mask_type from_bools(bool a0,
                                                      bool a1,
                                                      bool a2,
                                                      bool a3,
                                                      bool a4,
                                                      bool a5,
                                                      bool a6,
                                                      bool a7) {
    return _mm256_castsi256_ps(_mm256_set_epi32(
        (a7 ? -1 : 0), (a6 ? -1 : 0), (a5 ? -1 : 0), (a4 ? -1 : 0),
        (a3 ? -1 : 0), (a2 ? -1 : 0), (a1 ? -1 : 0), (a0 ? -1 : 0)));
  }

  AKS_SIMD_OP_FUNC_PREFIX static simd_type select(mask_type a,
                                                  simd_type b,
                                                  simd_type c) {
    return _mm256_blendv_ps(c, b, a);
  }

  AKS_SIMD_OP_FUNC_PREFIX static bool any(mask_type a) {
    return _mm256_movemask_ps(a) != 0;
  }

  AKS_SIMD_OP_FUNC_PREFIX static bool all(mask_type a) {
    return _mm256_movemask_ps(a) == 0xf;
  }
};
}  // namespace avx

using vec8f = simd::vec<avx::vec8f_spec>;
using vec8f_mask = simd::mask<avx::vec8f_spec>;

}  // namespace simd
}  // namespace aks

#endif  // !AKS_SIMD_VEC8F_HPP__

#ifndef AKS_SIMD_COMMON_OPS_HPP
#define AKS_SIMD_COMMON_OPS_HPP

// #include "aks_simd_common.hpp"
#include <type_traits>

namespace aks {
namespace simd {

namespace detail_ops {

template <typename T>
struct spec {
  using type = std::remove_cvref_t<T>::spec;
};

template <typename T>
using spec_t = typename spec<T>::type;

template <typename F>
AKS_SIMD_OP_FUNC_PREFIX auto check_apply(is_vec_c auto const a,
                                         is_vec_c auto const b,
                                         F f) {
  static_assert(std::is_same_v<decltype(a), decltype(b)>, "mismatched types");
  using simd_vec_type = std::remove_cvref_t<decltype(a)>;
  return simd_vec_type{f(a, b)};
}

}  // namespace detail_ops

AKS_SIMD_OP_FUNC_PREFIX auto operator+(is_vec_c auto const a,
                                       is_vec_c auto const b) {
  return detail_ops::check_apply(a, b, detail_ops::spec_t<decltype(a)>::add);
}

AKS_SIMD_OP_FUNC_PREFIX auto operator-(is_vec_c auto const a,
                                       is_vec_c auto const b) {
  return detail_ops::check_apply(a, b, detail_ops::spec_t<decltype(a)>::sub);
}

AKS_SIMD_OP_FUNC_PREFIX auto operator*(is_vec_c auto const a,
                                       is_vec_c auto const b) {
  return detail_ops::check_apply(a, b, detail_ops::spec_t<decltype(a)>::mul);
}

AKS_SIMD_OP_FUNC_PREFIX auto operator/(is_vec_c auto const a,
                                       is_vec_c auto const b) {
  return detail_ops::check_apply(a, b, detail_ops::spec_t<decltype(a)>::div);
}

AKS_SIMD_OP_FUNC_PREFIX auto operator-(is_vec_c auto const a) {
  return std::remove_cvref_t<decltype(a)>{
      detail_ops::spec_t<decltype(a)>::neg(a)};
}

AKS_SIMD_OP_FUNC_PREFIX auto sqrt(is_vec_c auto const a) {
  return std::remove_cvref_t<decltype(a)>{
      detail_ops::spec_t<decltype(a)>::sqrt(a)};
}

AKS_SIMD_OP_FUNC_PREFIX auto abs(is_vec_c auto const a) {
  return std::remove_cvref_t<decltype(a)>{
      detail_ops::spec_t<decltype(a)>::abs(a)};
}

AKS_SIMD_OP_FUNC_PREFIX auto as_array(is_vec_c auto const a) {
  return detail_ops::spec_t<decltype(a)>::simd_as_array(a);
}

}  // namespace simd
}  // namespace aks

#endif  // !AKS_SIMD_COMMON_OPS_HPP

#ifndef AKS_SIMD_COMMON_COMPARE_OPS_HPP
#define AKS_SIMD_COMMON_COMPARE_OPS_HPP

// #include "aks_simd_common.hpp"
#include <type_traits>

namespace aks {
namespace simd {

namespace detail_cops {

template <typename T>
struct spec {
  using type = std::remove_cvref_t<T>::spec;
};

template <typename T>
using spec_t = typename spec<T>::type;

template <typename F>
AKS_SIMD_OP_FUNC_PREFIX auto check_apply(is_vec_c auto const a,
                                         is_vec_c auto const b,
                                         F f) {
  static_assert(std::is_same_v<decltype(a), decltype(b)>, "mismatched types");
  using mask_type = mask<spec_t<decltype(a)>>;
  return mask_type{f(a, b)};
}

}  // namespace detail_cops

AKS_SIMD_OP_FUNC_PREFIX auto operator<(is_vec_c auto const a,
                                       is_vec_c auto const b) {
  return detail_cops::check_apply(a, b, detail_cops::spec_t<decltype(a)>::lt);
}

AKS_SIMD_OP_FUNC_PREFIX auto operator<=(is_vec_c auto const a,
                                        is_vec_c auto const b) {
  return detail_cops::check_apply(a, b, detail_cops::spec_t<decltype(a)>::lte);
}

AKS_SIMD_OP_FUNC_PREFIX auto operator>(is_vec_c auto const a,
                                       is_vec_c auto const b) {
  return detail_cops::check_apply(a, b, detail_cops::spec_t<decltype(a)>::gt);
}

AKS_SIMD_OP_FUNC_PREFIX auto operator>=(is_vec_c auto const a,
                                        is_vec_c auto const b) {
  return detail_cops::check_apply(a, b, detail_cops::spec_t<decltype(a)>::gte);
}

AKS_SIMD_OP_FUNC_PREFIX auto operator==(is_vec_c auto const a,
                                        is_vec_c auto const b) {
  return detail_cops::check_apply(a, b, detail_cops::spec_t<decltype(a)>::eq);
}

AKS_SIMD_OP_FUNC_PREFIX auto operator!=(is_vec_c auto const a,
                                        is_vec_c auto const b) {
  return detail_cops::check_apply(a, b, detail_cops::spec_t<decltype(a)>::neq);
}

}  // namespace simd
}  // namespace aks

#endif  // !AKS_SIMD_COMMON_COMPARE_OPS_HPP

#ifndef AKS_SIMD_COMMON_LOGICAL_OPS_HPP
#define AKS_SIMD_COMMON_LOGICAL_OPS_HPP

// #include "aks_simd_common.hpp"

namespace aks {
namespace simd {

namespace detail_lops {
template <typename T>
struct spec {
  using type = std::remove_cvref_t<T>::spec;
};

template <typename T>
using spec_t = typename spec<T>::type;

template <typename F>
AKS_SIMD_OP_FUNC_PREFIX auto check_apply(is_mask_c auto const a,
                                         is_mask_c auto const b,
                                         F f) {
  static_assert(std::is_same_v<decltype(a), decltype(b)>, "mismatched types");
  using mask_vec_type = std::remove_cvref_t<decltype(a)>;
  return mask_vec_type{f(a, b)};
}

}  // namespace detail_lops

AKS_SIMD_OP_FUNC_PREFIX auto operator&&(is_mask_c auto const a,
                                        is_mask_c auto const b) {
  return detail_lops::check_apply(a, b, detail_lops::spec_t<decltype(a)>::and_);
}

AKS_SIMD_OP_FUNC_PREFIX auto operator||(is_mask_c auto const a,
                                        is_mask_c auto const b) {
  return detail_lops::check_apply(a, b, detail_lops::spec_t<decltype(a)>::or_);
}

AKS_SIMD_OP_FUNC_PREFIX auto operator^(is_mask_c auto const a,
                                       is_mask_c auto const b) {
  return detail_lops::check_apply(a, b, detail_lops::spec_t<decltype(a)>::xor_);
}

AKS_SIMD_OP_FUNC_PREFIX auto operator!(is_mask_c auto const a) {
  return std::remove_cvref_t<decltype(a)>{
      detail_lops::spec_t<decltype(a)>::not_(a)};
}

AKS_SIMD_OP_FUNC_PREFIX auto all(is_mask_c auto const a) {
  return detail_lops::spec_t<decltype(a)>::all(a);
}

AKS_SIMD_OP_FUNC_PREFIX auto any(is_mask_c auto const a) {
  return detail_lops::spec_t<decltype(a)>::any(a);
}

AKS_SIMD_OP_FUNC_PREFIX auto as_array(is_mask_c auto const a) {
  return detail_lops::spec_t<decltype(a)>::mask_as_array(a);
}

template <is_mask_c T, typename... Ts>
  requires(std::same_as<Ts, bool> && ...)
AKS_SIMD_OP_FUNC_PREFIX auto from_bools(Ts... ts) {
  return T{detail_ops::spec_t<T>::from_bools(ts...)};
}

}  // namespace simd
}  // namespace aks

#endif  // !AKS_SIMD_COMMON_LOGICAL_OPS_HPP

#ifndef AKS_SIMD_COMMON_SELECT_OPS_HPP
#define AKS_SIMD_COMMON_SELECT_OPS_HPP

// #include "aks_simd_common.hpp"

namespace aks {
namespace simd {
namespace detail_sops {

template <typename T>
struct spec {
  using type = std::remove_cvref_t<T>::spec;
};

template <typename T>
using spec_t = typename spec<T>::type;

}  // namespace detail_sops

AKS_SIMD_OP_FUNC_PREFIX auto select(is_mask_c auto const a,
                                    is_vec_c auto const b,
                                    is_vec_c auto const c) {
  static_assert(std::is_same_v<decltype(b), decltype(c)>, "mismatched types");
  static_assert(std::is_same_v<detail_sops::spec_t<decltype(a)>,
                               detail_sops::spec_t<decltype(b)>>,
                "mismatched types between mask and vec");

  using mask_vec_type = std::remove_cvref_t<decltype(a)>;
  using simd_vec_type = std::remove_cvref_t<decltype(b)>;

  return simd_vec_type{detail_sops::spec_t<decltype(a)>::select(a, b, c)};
}

}  // namespace simd
}  // namespace aks

#endif  // !AKS_SIMD_COMMON_SELECT_OPS_HPP

#ifndef AKS_SIMD_COMMON_FORMAT_OPS_HPP
#define AKS_SIMD_COMMON_FORMAT_OPS_HPP

#include <fmt/format.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

template <aks::simd::is_simd_c simd_>
struct fmt::formatter<simd_> {
  using custom_range_t = simd_;

  // Parse format specifiers if needed
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  // Format the CustomRange
  template <typename format_ctx>
  auto format(const custom_range_t& range, format_ctx& ctx) const {
    // Format the range as a list
    fmt::format_to(ctx.out(), "[");
    auto data = as_array(range);
    std::size_t size = range.size();

    for (std::size_t i = 0; i < size; ++i) {
      if (i > 0) {
        fmt::format_to(ctx.out(), ", ");
      }
      fmt::format_to(ctx.out(), "{}", data[i]);
    }

    return fmt::format_to(ctx.out(), "]");
  }
};

#undef AKS_SIMD_OP_FUNC_PREFIX

#endif  // !AKS_SIMD_COMMON_FORMAT_OPS_HPP
