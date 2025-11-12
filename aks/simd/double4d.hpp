
#ifndef DOUBLE4D_HPP
#define DOUBLE4D_HPP

#include <immintrin.h>
#include <smmintrin.h>

#include <bitset>
#include <cmath>
#include <numbers>
#include <ostream>

// NOLINTBEGIN

#include "simd_type.hpp"

namespace aks {
using d4_vt = double;
using d4_st = simd_type<d4_vt, 4>::type;

namespace detail {
double hadd_m256d(__m256d vec) {
  // Step 1: Horizontal addition within the 256-bit vector
  // _mm256_hadd_pd adds adjacent pairs of doubles
  // If vec is [a, b, c, d], sum becomes [a+b, c+d, a+b, c+d]
  __m256d sum = _mm256_hadd_pd(vec, vec);

  // Step 2: Extract the lower 128 bits of the result
  // This gets the first two doubles: [a+b, c+d]
  __m128d low = _mm256_extractf128_pd(sum, 0);

  // Step 3: Extract the upper 128 bits of the result
  // This also gets [a+b, c+d] (remember, sum had two copies)
  __m128d high = _mm256_extractf128_pd(sum, 1);

  // Step 4: Add the lower and upper 128-bit parts
  // This performs the final addition: [a+b+c+d, a+b+c+d]
  __m128d result = _mm_add_pd(low, high);

  // Step 5: Extract the first (and only relevant) double from the result
  // This converts the first 64-bit double from the 128-bit vector to a scalar
  // double
  return _mm_cvtsd_f64(result);
}

double hmul_m256d(__m256d vec) {
  // Step 1: Extract the lower 128 bits of the vector
  // If vec is [a, b, c, d], low becomes [a, b]
  __m128d low = _mm256_extractf128_pd(vec, 0);

  // Step 2: Extract the upper 128 bits of the vector
  // high becomes [c, d]
  __m128d high = _mm256_extractf128_pd(vec, 1);

  // Step 3: Multiply the lower and upper parts element-wise
  // prod becomes [a*c, b*d]
  __m128d prod = _mm_mul_pd(low, high);

  // Step 4: Shuffle the product to swap its elements
  // The '1' argument means to reverse the order of elements
  // shuffled becomes [b*d, a*c]
  __m128d shuffled = _mm_shuffle_pd(prod, prod, 1);

  // Step 5: Multiply the original product with the shuffled product
  // result becomes [a*c*b*d, b*d*a*c] (both elements are identical)
  __m128d result = _mm_mul_pd(prod, shuffled);

  // Step 6: Extract the first (and only relevant) double from the result
  // This converts the first 64-bit double from the 128-bit vector to a scalar
  // double
  return _mm_cvtsd_f64(result);
}
}  // namespace detail

struct alignas(32) msk_d4 {
  using simd_type = d4_st;
  constexpr static size_t const dim = 4;
  constexpr static size_t alignment = 32;

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4201)
#else
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
  union alignas(32) {
    d4_st data;
    struct alignas(32) {
      d4_vt x, y, z, w;
    };
  };
#if defined(_MSC_VER)
#pragma warning(pop)
#else
#pragma GCC diagnostic warning "-Wpedantic"
#endif

  explicit msk_d4(d4_st adata) : data(adata) {}

  msk_d4 and_(msk_d4 other) const {
    return msk_d4(_mm256_and_pd(data, other.data));
  }

  msk_d4 or_(msk_d4 other) const {
    return msk_d4(_mm256_or_pd(data, other.data));
  }

  msk_d4 xor_(msk_d4 other) const {
    return msk_d4(_mm256_xor_pd(data, other.data));
  }

  msk_d4 not_() const {
    return msk_d4(_mm256_xor_pd(data, _mm256_set1_pd(-0.0)));
  }

  msk_d4 and_not(msk_d4 other) const {
    return msk_d4(_mm256_andnot_pd(other.data, data));
  }

  msk_d4 or_not(msk_d4 other) const { return ((*this).or_(other)).not_(); }

  msk_d4 xor_not(msk_d4 other) const { return ((*this).xor_(other)).not_(); }

  int mask() const { return _mm256_movemask_pd(data); }

  int popcount() const {
#ifdef _WIN32
    return __popcnt(mask());
#else
    return __builtin_popcount(static_cast<unsigned int>(mask()));
#endif
  }

  bool operator[](int i) const { return mask() & (1 << i); }

  bool operator==(msk_d4 other) const { return mask() == other.mask(); }

  bool operator!=(msk_d4 other) const { return mask() != other.mask(); }

  msk_d4 operator&(msk_d4 other) const { return and_(other); }

  msk_d4 operator|(msk_d4 other) const { return or_(other); }

  msk_d4 operator^(msk_d4 other) const { return xor_(other); }

  msk_d4 operator~() const { return not_(); }

  static msk_d4 create_mask(int mask) {
    __m256i int_mask =
        _mm256_set_epi64x((mask & 0x8) ? -1LL : 0, (mask & 0x4) ? -1LL : 0,
                          (mask & 0x2) ? -1LL : 0, (mask & 0x1) ? -1LL : 0);
    return msk_d4(_mm256_castsi256_pd(int_mask));
  }

  static msk_d4 create_mask(bool b0, bool b1, bool b2, bool b3) {
    auto v = [](bool av) { return av ? -1LL : 0; };
    __m256i int_mask = _mm256_set_epi64x(v(b3), v(b2), v(b1), v(b0));
    return msk_d4(_mm256_castsi256_pd(int_mask));
  }

  static msk_d4 create_mask(int b0, int b1, int b2, int b3) {
    auto v = [](int av) { return av ? -1LL : 0; };
    __m256i int_mask = _mm256_set_epi64x(v(b3), v(b2), v(b1), v(b0));
    return msk_d4(_mm256_castsi256_pd(int_mask));
  }

  static msk_d4 create_mask(std::bitset<4> mask) {
    auto v = [&](std::size_t av) { return mask[av] ? -1LL : 0LL; };
    __m256i int_mask = _mm256_set_epi64x(v(0), v(1), v(2), v(3));
    return msk_d4(_mm256_castsi256_pd(int_mask));
  }
};

std::ostream& operator<<(std::ostream& os, msk_d4 const p) {
  auto as_str = [](bool mask) {
    if (mask == 0) {
      return "F";
    } else {
      return "T";
    }
  };

  os << "mask(" << as_str(p[0]) << "," << as_str(p[1]) << "," << as_str(p[2])
     << "," << as_str(p[3]) << ")";
  return os;
}

struct alignas(32) dbl4 {
  using mask_type = msk_d4;
  using value_type = d4_vt;
  using simd_type = d4_st;
  constexpr static size_t alignment = 32;
  constexpr static size_t dim = 4;

  static dbl4 load(double const* p) { return dbl4(_mm256_load_pd(p)); }
  static dbl4 loadu(double const* p) { return dbl4(_mm256_loadu_pd(p)); }
  static void store(double* p, dbl4 const& v) { _mm256_store_pd(p, v.data); }
  static void storeu(double* p, dbl4 const& v) { _mm256_storeu_pd(p, v.data); }

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4201)
#else
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
  union alignas(32) {
    simd_type data;
    struct {
      value_type x, y, z, w;
    };
    struct {
      value_type _0, _1, _2, _3;
    };
  };
#if defined(_MSC_VER)
#pragma warning(pop)
#else
#pragma GCC diagnostic warning "-Wpedantic"
#endif

  dbl4() = default;
  dbl4(value_type ax, value_type ay, value_type az, value_type aw)
      : x(ax), y(ay), z(az), w(aw) {}

  explicit dbl4(simd_type adata) : data(adata) {}
  explicit dbl4(value_type v) : x(v), y(v), z(v), w(v) {}

  operator simd_type() { return data; }

  dbl4(dbl4 const& other) : data(other.data) {}

  d4_vt& operator[](int i) { return (&x)[i]; }

  d4_vt operator[](int i) const { return (&x)[i]; }

  template <typename F>
  dbl4 transform(F f) const {
    return dbl4(f(x), f(y), f(z), f(w));
  }

  void store(double* p) const { _mm256_store_pd(p, data); }
  void storeu(double* p) const { _mm256_storeu_pd(p, data); }

  dbl4& operator=(dbl4 const& other) = default;
  dbl4& operator=(dbl4&& other) = default;

  dbl4(dbl4&& other) = default;

  dbl4 operator+(dbl4 const other) const {
    return dbl4(_mm256_add_pd(data, other.data));
  }

  dbl4 operator-(dbl4 const other) const {
    return dbl4(_mm256_sub_pd(data, other.data));
  }

  dbl4 operator*(dbl4 const other) const {
    return dbl4(_mm256_mul_pd(data, other.data));
  }

  dbl4 operator/(dbl4 const other) const {
    return dbl4(_mm256_div_pd(data, other.data));
  }

  dbl4 min(dbl4 const other) const {
    return dbl4(_mm256_min_pd(data, other.data));
  }

  dbl4 max(dbl4 const other) const {
    return dbl4(_mm256_max_pd(data, other.data));
  }

  dbl4 operator+(d4_vt other) const {
    return dbl4(_mm256_add_pd(data, _mm256_set1_pd(other)));
  }

  dbl4 operator-(d4_vt other) const {
    return dbl4(_mm256_sub_pd(data, _mm256_set1_pd(other)));
  }

  dbl4 operator*(d4_vt other) const {
    return dbl4(_mm256_mul_pd(data, _mm256_set1_pd(other)));
  }

  dbl4 operator/(d4_vt other) const {
    return dbl4(_mm256_div_pd(data, _mm256_set1_pd(other)));
  }

  dbl4& operator+=(dbl4 const other) {
    data = _mm256_add_pd(data, other.data);
    return *this;
  }

  dbl4& operator-=(dbl4 const other) {
    data = _mm256_sub_pd(data, other.data);
    return *this;
  }

  dbl4& operator*=(dbl4 const other) {
    data = _mm256_mul_pd(data, other.data);
    return *this;
  }

  dbl4& operator/=(dbl4 const other) {
    data = _mm256_div_pd(data, other.data);
    return *this;
  }

  dbl4& operator+=(d4_vt other) {
    data = _mm256_add_pd(data, _mm256_set1_pd(other));
    return *this;
  }

  dbl4& operator-=(d4_vt other) {
    data = _mm256_sub_pd(data, _mm256_set1_pd(other));
    return *this;
  }

  dbl4& operator*=(d4_vt other) {
    data = _mm256_mul_pd(data, _mm256_set1_pd(other));
    return *this;
  }

  dbl4& operator/=(d4_vt other) {
    data = _mm256_div_pd(data, _mm256_set1_pd(other));
    return *this;
  }

  dbl4 operator-() const {
    return dbl4(_mm256_sub_pd(_mm256_setzero_pd(), data));
  }

  dbl4 reciprocal() const {
    return dbl4(_mm256_div_pd(_mm256_set1_pd(1.0), data));
  }

  dbl4 square() const { return dbl4(_mm256_mul_pd(data, data)); }

  dbl4 sqrt() const { return dbl4(_mm256_sqrt_pd(data)); }

  dbl4 rsqrt() const {
    return dbl4(_mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(data)));
  }

  static dbl4 zeros() { return dbl4(_mm256_setzero_pd()); }

  static dbl4 ones() { return dbl4(_mm256_set1_pd(1.0)); }

  static dbl4 set1(d4_vt x) { return dbl4(_mm256_set1_pd(x)); }

  dbl4 abs() const {
    return dbl4(_mm256_andnot_pd(_mm256_set1_pd(-0.0), data));
  }

  dbl4 floor() const { return dbl4(_mm256_floor_pd(data)); }

  dbl4 ceil() const { return dbl4(_mm256_ceil_pd(data)); }

  dbl4 round() const {
    return dbl4(_mm256_round_pd(data, _MM_FROUND_TO_NEAREST_INT));
  }

  dbl4 trunc() const { return dbl4(_mm256_round_pd(data, _MM_FROUND_TO_ZERO)); }

  dbl4 round_up() const {
    return dbl4(_mm256_round_pd(data, _MM_FROUND_TO_POS_INF));
  }

  dbl4 round_down() const {
    return dbl4(_mm256_round_pd(data, _MM_FROUND_TO_NEG_INF));
  }

  dbl4 round_half_to_even() const {
    return dbl4(
        _mm256_round_pd(data, _MM_FROUND_TO_NEG_INF | _MM_FROUND_TO_POS_INF));
  }

  dbl4 round_half_to_odd() const {
    return dbl4(_mm256_round_pd(data, _MM_FROUND_TO_NEG_INF |
                                          _MM_FROUND_TO_POS_INF |
                                          _MM_FROUND_NO_EXC));
  }

  dbl4 round_half_away_from_zero() const {
    return dbl4(
        _mm256_round_pd(data, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
  }

  dbl4 round_half_towards_zero() const {
    return dbl4(
        _mm256_round_pd(data, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
  }

  dbl4 round_half_to_zero() const {
    return dbl4(_mm256_round_pd(data, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }

  dbl4 round_half_towards_neg_infinity() const {
    return dbl4(_mm256_round_pd(data, _MM_FROUND_TO_NEG_INF));
  }

  dbl4 round_half_towards_pos_infinity() const {
    return dbl4(_mm256_round_pd(data, _MM_FROUND_TO_POS_INF));
  }

  dbl4 clamp(dbl4 const min, dbl4 const max) const {
    return dbl4(_mm256_min_pd(_mm256_max_pd(data, min.data), max.data));
  }

  dbl4 clamp(d4_vt const min, d4_vt const max) const {
    return clamp(set1(min), set1(max));
  }

  d4_vt horizontal_add() const { return detail::hadd_m256d(data); }

  d4_vt horizontal_mul() const { return detail::hmul_m256d(data); }

  mask_type operator>(dbl4 const other) const {
    return mask_type(_mm256_cmp_pd(data, other.data, _CMP_GT_OQ));
  }

  mask_type operator<(dbl4 const other) const {
    return mask_type(_mm256_cmp_pd(data, other.data, _CMP_LT_OQ));
  }

  mask_type operator>=(dbl4 const other) const {
    return mask_type(_mm256_cmp_pd(data, other.data, _CMP_GE_OQ));
  }

  mask_type operator<=(dbl4 const other) const {
    return mask_type(_mm256_cmp_pd(data, other.data, _CMP_LE_OQ));
  }

  mask_type operator==(dbl4 const other) const {
    return mask_type(_mm256_cmp_pd(data, other.data, _CMP_EQ_OQ));
  }

  mask_type operator!=(dbl4 const other) const {
    return mask_type(_mm256_cmp_pd(data, other.data, _CMP_NEQ_OQ));
  }

  mask_type operator>(d4_vt other) const {
    return mask_type(_mm256_cmp_pd(data, _mm256_set1_pd(other), _CMP_GT_OQ));
  }

  mask_type operator<(d4_vt other) const {
    return mask_type(_mm256_cmp_pd(data, _mm256_set1_pd(other), _CMP_LT_OQ));
  }

  mask_type operator>=(d4_vt other) const {
    return mask_type(_mm256_cmp_pd(data, _mm256_set1_pd(other), _CMP_GE_OQ));
  }

  mask_type operator<=(d4_vt other) const {
    return mask_type(_mm256_cmp_pd(data, _mm256_set1_pd(other), _CMP_LE_OQ));
  }

  mask_type operator==(d4_vt other) const {
    return mask_type(_mm256_cmp_pd(data, _mm256_set1_pd(other), _CMP_EQ_OQ));
  }

  mask_type operator!=(d4_vt other) const {
    return mask_type(_mm256_cmp_pd(data, _mm256_set1_pd(other), _CMP_NEQ_OQ));
  }

  mask_type is_close(dbl4 const other, d4_vt threshold) const {
    return (*this - other).abs() < threshold;
  }

  mask_type is_nan() {
    return mask_type(_mm256_cmp_pd(data, data, _CMP_UNORD_Q));
  }

  mask_type is_inf() {
    d4_st const inf_mask =
        _mm256_set1_pd(std::numeric_limits<d4_vt>::infinity());

    // Compare x with infinity (this catches both positive and negative
    // infinity)
    d4_st cmp_result = _mm256_cmp_pd(
        _mm256_andnot_pd(_mm256_set1_pd(-0.0), data), inf_mask, _CMP_EQ_OQ);

    return mask_type(cmp_result);
  }

  mask_type is_finite() { return ~(is_nan() | is_inf()); }

  int make_mask() const { return _mm256_movemask_pd(data); }

#ifndef AKS_DO_NOT_USE_SVML_INTRINSICS

  dbl4 exp() const { return dbl4(_mm256_exp_pd(data)); }

  dbl4 log() const { return dbl4(_mm256_log_pd(data)); }

  dbl4 sin() const { return dbl4(_mm256_sin_pd(data)); }

  dbl4 cos() const { return dbl4(_mm256_cos_pd(data)); }

  dbl4 tan() const { return dbl4(_mm256_tan_pd(data)); }

  dbl4 sinh() const { return dbl4(_mm256_sinh_pd(data)); }

  dbl4 cosh() const { return dbl4(_mm256_cosh_pd(data)); }

  dbl4 tanh() const { return dbl4(_mm256_tanh_pd(data)); }

  dbl4 asin() const { return dbl4(_mm256_asin_pd(data)); }

  dbl4 acos() const { return dbl4(_mm256_acos_pd(data)); }

  dbl4 atan() const { return dbl4(_mm256_atan_pd(data)); }

  dbl4 sind() const { return dbl4(_mm256_sind_pd(data)); }

  dbl4 cosd() const { return dbl4(_mm256_cosd_pd(data)); }

  dbl4 tand() const { return dbl4(_mm256_tand_pd(data)); }

  dbl4 asinh() const { return dbl4(_mm256_asinh_pd(data)); }

  dbl4 acosh() const { return dbl4(_mm256_acosh_pd(data)); }

  dbl4 atanh() const { return dbl4(_mm256_atanh_pd(data)); }

  dbl4 erf() const { return dbl4(_mm256_erf_pd(data)); }

  dbl4 erfc() const { return dbl4(_mm256_erfc_pd(data)); }

  dbl4 erfinv() const { return dbl4(_mm256_erfinv_pd(data)); }

  dbl4 cdfnorm() const { return dbl4(_mm256_cdfnorm_pd(data)); }

  dbl4 cdfnorminv() const { return dbl4(_mm256_cdfnorminv_pd(data)); }
#else

  static d4_vt d2r(d4_vt degree) { return degree * (std::numbers::pi / 180.0); }

  dbl4 log() const {
    return transform([](d4_vt ax) { return std::log(ax); });
  }

  dbl4 exp() const {
    return transform([](d4_vt ax) { return std::exp(ax); });
  }

  dbl4 sin() const {
    return transform([](d4_vt ax) { return std::sin(ax); });
  }

  dbl4 cos() const {
    return transform([](d4_vt ax) { return std::cos(ax); });
  }

  dbl4 tan() const {
    return transform([](d4_vt ax) { return std::tan(ax); });
  }

  dbl4 sinh() const {
    return transform([](d4_vt ax) { return std::sinh(ax); });
  }

  dbl4 cosh() const {
    return transform([](d4_vt ax) { return std::cosh(ax); });
  }

  dbl4 tanh() const {
    return transform([](d4_vt ax) { return std::tanh(ax); });
  }

  dbl4 asin() const {
    return transform([](d4_vt ax) { return std::asin(ax); });
  }

  dbl4 acos() const {
    return transform([](d4_vt ax) { return std::acos(ax); });
  }

  dbl4 atan() const {
    return transform([](d4_vt ax) { return std::atan(ax); });
  }

  dbl4 sind() const {
    return transform([](d4_vt ax) { return std::sin(d2r(ax)); });
  }

  dbl4 cosd() const {
    return transform([](d4_vt ax) { return std::cos(d2r(ax)); });
  }

  dbl4 tand() const {
    return transform([](d4_vt ax) { return std::tan(d2r(ax)); });
  }

  dbl4 asinh() const {
    return transform([](d4_vt ax) { return std::asinh(ax); });
  }

  dbl4 acosh() const {
    return transform([](d4_vt ax) { return std::acosh(ax); });
  }

  dbl4 atanh() const {
    return transform([](d4_vt ax) { return std::atanh(ax); });
  }

  dbl4 erf() const {
    return transform([](d4_vt ax) { return std::erf(ax); });
  }

  dbl4 erfc() const {
    return transform([](d4_vt ax) { return std::erfc(ax); });
  }

  dbl4 erfinv() const {
    return ones() / transform([](d4_vt ax) { return std::erf(ax); });
  }

  dbl4 cdfnorm() const {
    auto normcdf = [](d4_vt ax, d4_vt mean = 0.0, d4_vt stddev = 1.0) {
      // Standardize the input
      d4_vt z_ = (ax - mean) / stddev;

      // Constants
      d4_vt const a1 = 0.254829592;
      d4_vt const a2 = -0.284496736;
      d4_vt const a3 = 1.421413741;
      d4_vt const a4 = -1.453152027;
      d4_vt const a5 = 1.061405429;
      d4_vt const p = 0.3275911;

      // Save the sign of z
      int sign = (z_ < 0) ? -1 : 1;
      z_ = std::fabs(z_) / std::sqrt(2.0);

      // A&S formula 7.1.26
      d4_vt t = 1.0 / (1.0 + p * z_);
      d4_vt y_ = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
                           std::exp(-z_ * z_);

      return 0.5 * (1.0 + sign * y_);
    };

    return transform([&normcdf](d4_vt ax) { return normcdf(ax); });
  }

  dbl4 cdfnorminv() const {
    auto inverseNormCDF = [](d4_vt p, d4_vt mean = 0.0, d4_vt stddev = 1.0) {
      if (p <= 0.0 || p >= 1.0) {
        return std::numeric_limits<d4_vt>::quiet_NaN();
      }

      // Coefficients in rational approximations
      d4_vt const a1 = -3.969683028665376e+01;
      d4_vt const a2 = 2.209460984245205e+02;
      d4_vt const a3 = -2.759285104469687e+02;
      d4_vt const a4 = 1.383577518672690e+02;
      d4_vt const a5 = -3.066479806614716e+01;
      d4_vt const a6 = 2.506628277459239e+00;

      d4_vt const b1 = -5.447609879822406e+01;
      d4_vt const b2 = 1.615858368580409e+02;
      d4_vt const b3 = -1.556989798598866e+02;
      d4_vt const b4 = 6.680131188771972e+01;
      d4_vt const b5 = -1.328068155288572e+01;

      d4_vt const c1 = -7.784894002430293e-03;
      d4_vt const c2 = -3.223964580411365e-01;
      d4_vt const c3 = -2.400758277161838e+00;
      d4_vt const c4 = -2.549732539343734e+00;
      d4_vt const c5 = 4.374664141464968e+00;
      d4_vt const c6 = 2.938163982698783e+00;

      d4_vt const d1 = 7.784695709041462e-03;
      d4_vt const d2 = 3.224671290700398e-01;
      d4_vt const d3 = 2.445134137142996e+00;
      d4_vt const d4 = 3.754408661907416e+00;

      // Define break-points
      d4_vt const p_low = 0.02425;
      d4_vt const p_high = 1 - p_low;

      // Rational approximation for lower region
      if (p < p_low) {
        d4_vt q = std::sqrt(-2 * std::log(p));
        d4_vt x_ = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
        return mean - x_ * stddev;
      }

      // Rational approximation for central region
      if (p <= p_high) {
        d4_vt q = p - 0.5;
        d4_vt r = q * q;
        d4_vt x_ = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) *
                   q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
        return mean + x_ * stddev;
      }

      // Rational approximation for upper region
      if (p < 1) {
        d4_vt q = std::sqrt(-2 * std::log(1 - p));
        d4_vt x_ = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
        return mean + x_ * stddev;
      }

      // Case when p = 1
      return std::numeric_limits<d4_vt>::infinity();
    };

    return transform(
        [&inverseNormCDF](d4_vt ax) { return inverseNormCDF(ax); });
  }

#endif
};

dbl4 operator+(d4_vt x, dbl4 const y) {
  return y + x;
}

dbl4 operator-(d4_vt x, dbl4 const y) {
  return -y + x;
}

dbl4 operator*(d4_vt x, dbl4 const y) {
  return y * x;
}

dbl4 operator/(d4_vt x, dbl4 const y) {
  return dbl4::set1(x) / y;
}

dbl4 if_then_else(dbl4::mask_type const mask,
                  dbl4 const true_value,
                  dbl4 const false_value) {
  return dbl4(_mm256_blendv_pd(false_value.data, true_value.data, mask.data));
}

dbl4 fmadd(dbl4 const x, dbl4 const y, dbl4 const z) {
  return dbl4(_mm256_fmadd_pd(x.data, y.data, z.data));
}

dbl4 fmsub(dbl4 const x, dbl4 const y, dbl4 const z) {
  return dbl4(_mm256_fmsub_pd(x.data, y.data, z.data));
}

dbl4 fnmadd(dbl4 const x, dbl4 const y, dbl4 const z) {
  return dbl4(_mm256_fnmadd_pd(x.data, y.data, z.data));
}

dbl4 fnmsub(dbl4 const x, dbl4 const y, dbl4 const z) {
  return dbl4(_mm256_fnmsub_pd(x.data, y.data, z.data));
}

template <typename F>
dbl4 transform(dbl4 const x, dbl4 const y, F f) {
  return dbl4(f(x.x, y.x), f(x.y, y.y), f(x.z, y.z), f(x.w, y.w));
}

dbl4 atan2(dbl4 const x, dbl4 const y) {
#ifndef AKS_DO_NOT_USE_SVML_INTRINSICS
  return dbl4(_mm256_atan2_pd(x.data, y.data));
#else
  auto atan2f = [](d4_vt ax, d4_vt ay) { return std::atan2(ax, ay); };
  return transform(x, y, atan2f);
#endif
}

dbl4 pow(dbl4 const x, dbl4 const y) {
#ifndef AKS_DO_NOT_USE_SVML_INTRINSICS
  return dbl4(_mm256_pow_pd(x.data, y.data));
#else
  auto powf = [](d4_vt ax, d4_vt ay) { return std::pow(ax, ay); };
  return transform(x, y, powf);
#endif
}

std::ostream& operator<<(std::ostream& os, dbl4 const p) {
  os << "dbl4( " << p.x << ", " << p.y << ", " << p.z << ", " << p.w << " )";
  return os;
}

}  // namespace aks

#endif  // DOUBLE4D_HPP

// NOLINTEND
