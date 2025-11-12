#ifndef FLOAT8D_HPP
#define FLOAT8D_HPP

#include <immintrin.h>
#include <smmintrin.h>

#include <cmath>
#include <numbers>
#include <ostream>

#include "simd_type.hpp"

namespace aks {
using f8_vt = float;
using f8_simd = aks::simd_type<f8_vt, 8>;
using f8_st = f8_simd::type;

namespace detail {
float hadd_m256(__m256 vec) {
  // Step 1: First horizontal add
  // This adds adjacent pairs of floats within the 256-bit vector
  // Result: [a+b, c+d, e+f, g+h, a+b, c+d, e+f, g+h]
  __m256 sum = _mm256_hadd_ps(vec, vec);

  // Step 2: Second horizontal add
  // This further reduces the result, adding pairs from the previous step
  // Result: [a+b+c+d, e+f+g+h, a+b+c+d, e+f+g+h, a+b+c+d, e+f+g+h, a+b+c+d,
  // e+f+g+h]
  sum = _mm256_hadd_ps(sum, sum);

  // Step 3: Extract lower 128 bits
  // This gets the lower half of the 256-bit vector
  // Result: [a+b+c+d, e+f+g+h, a+b+c+d, e+f+g+h]
  __m128 lo = _mm256_castps256_ps128(sum);

  // Step 4: Extract upper 128 bits
  // This gets the upper half of the 256-bit vector
  // Result: [a+b+c+d, e+f+g+h, a+b+c+d, e+f+g+h]
  __m128 hi = _mm256_extractf128_ps(sum, 1);

  // Step 5: Add the lower and upper 128-bit halves
  // This performs the final addition to get the total sum
  // Result: [a+b+c+d+e+f+g+h, ..., ...]
  __m128 result = _mm_add_ps(lo, hi);

  // Step 6: Extract the first (and only relevant) float from the result
  // This converts the first 32-bit float from the 128-bit vector to a scalar
  // float
  return _mm_cvtss_f32(result);
}

float hmul_m256(__m256 vec) {
  // Step 1: Permute the vector to swap upper and lower 128-bit lanes
  // This creates a new vector where the upper 128 bits are now in the lower
  // half and vice versa If vec is [a, b, c, d, e, f, g, h], tmp becomes [e, f,
  // g, h, a, b, c, d]
  __m256 tmp = _mm256_permute2f128_ps(vec, vec, 1);

  // Step 2: Multiply the original vector with the permuted vector
  // This multiplies corresponding elements: [a*e, b*f, c*g, d*h, e*a, f*b, g*c,
  // h*d]
  __m256 mul = _mm256_mul_ps(vec, tmp);

  // Step 3: Permute within 128-bit lanes and multiply
  // 0xB1 = 10110001 in binary, which swaps the two 64-bit halves within each
  // 128-bit lane This creates [b*f, a*e, d*h, c*g, f*b, e*a, h*d, g*c] Then we
  // multiply this with the previous result
  mul = _mm256_mul_ps(mul, _mm256_permute_ps(mul, 0xB1));

  // Step 4: Permute 128-bit lanes and multiply
  // 0x4E = 01001110 in binary, which swaps the two 128-bit lanes
  // This creates [f*b, e*a, h*d, g*c, b*f, a*e, d*h, c*g]
  // Then we multiply this with the previous result
  mul = _mm256_mul_ps(mul, _mm256_permute_ps(mul, 0x4E));

  // Step 5: Extract the first float from the result vector
  // At this point, the first float contains the product of all 8 original
  // floats
  return _mm256_cvtss_f32(mul);
}
}  // namespace detail

struct alignas(32) msk_f8 {
  using simd_t = f8_simd;
  using simd_type = f8_st;
  constexpr static std::size_t const dim = simd_t::size;
  constexpr static std::size_t const alignment = 32;
  union alignas(32) {
    f8_st data;
    struct alignas(32) tokens {
      f8_vt f0, f1, f2, f3, f4, f5, f6, f7;
    };
  };

  explicit msk_f8(f8_st data) : data(data) {}

  msk_f8 and_(msk_f8 other) const {
    return msk_f8(_mm256_and_ps(data, other.data));
  }

  msk_f8 or_(msk_f8 other) const {
    return msk_f8(_mm256_or_ps(data, other.data));
  }

  msk_f8 xor_(msk_f8 other) const {
    return msk_f8(_mm256_xor_ps(data, other.data));
  }

  msk_f8 not_() const {
    return msk_f8(_mm256_xor_ps(data, _mm256_set1_ps(-0.0)));
  }

  msk_f8 and_not(msk_f8 other) const {
    return msk_f8(_mm256_andnot_ps(other.data, data));
  }

  msk_f8 or_not(msk_f8 other) const { return ((*this).or_(other)).not_(); }

  msk_f8 xor_not(msk_f8 other) const { return ((*this).xor_(other)).not_(); }

  int mask() const { return _mm256_movemask_ps(data); }

  int popcount() const {
#ifdef _WIN32
    return __popcnt(mask());
#else
    return __builtin_popcount(mask());
#endif
  }

  bool operator[](int i) const { return mask() & (1 << i); }

  bool operator==(msk_f8 other) const { return mask() == other.mask(); }

  bool operator!=(msk_f8 other) const { return mask() != other.mask(); }

  msk_f8 operator&(msk_f8 other) const { return and_(other); }

  msk_f8 operator|(msk_f8 other) const { return or_(other); }

  msk_f8 operator^(msk_f8 other) const { return xor_(other); }

  msk_f8 operator~() const { return not_(); }

  bool all() const { return mask() == 0xF; }
  bool any() const { return mask() != 0; }

  static msk_f8 create_mask(int mask) {
    // Step 1: Create an integer vector with the mask bits
    __m256i int_mask = _mm256_set_epi32(
        (mask & 0x80) ? -1 : 0, (mask & 0x40) ? -1 : 0, (mask & 0x20) ? -1 : 0,
        (mask & 0x10) ? -1 : 0, (mask & 0x08) ? -1 : 0, (mask & 0x04) ? -1 : 0,
        (mask & 0x02) ? -1 : 0, (mask & 0x01) ? -1 : 0);

    // Step 2: Shift the bits to the most significant bit position
    __m256i shifted_mask = _mm256_slli_epi32(int_mask, 31);

    // Step 3: Convert to __m256 float vector
    __m256 float_mask = _mm256_castsi256_ps(shifted_mask);

    return msk_f8(float_mask);
  }

  static msk_f8 create_mask(bool b0,
                            bool b1,
                            bool b2,
                            bool b3,
                            bool b4,
                            bool b5,
                            bool b6,
                            bool b7) {
    auto v = [](bool v) { return v ? -1 : 0; };
    __m256i int_mask = _mm256_set_epi32(v(b7), v(b6), v(b5), v(b4), v(b3),
                                        v(b2), v(b1), v(b0));
    return msk_f8(_mm256_castsi256_ps(int_mask));
  }

  static msk_f8
  create_mask(int b0, int b1, int b2, int b3, int b4, int b5, int b6, int b7) {
    auto v = [](int v) { return v ? -1 : 0; };
    __m256i int_mask = _mm256_set_epi32(v(b7), v(b6), v(b5), v(b4), v(b3),
                                        v(b2), v(b1), v(b0));
    return msk_f8(_mm256_castsi256_ps(int_mask));
  }

  static msk_f8 create_mask(std::bitset<8> mask) {
    auto v = [&](int v) { return mask[v] ? -1 : 0; };
    __m256i int_mask =
        _mm256_set_epi32(v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7));
    return msk_f8(_mm256_castsi256_ps(int_mask));
  }
};

std::ostream& operator<<(std::ostream& os, msk_f8 const p) {
  auto as_str = [](bool mask) {
    if (mask == 0) {
      return "F";
    } else {
      return "T";
    }
  };

  os << "mask(" << as_str(p[0]) << "," << as_str(p[1]) << "," << as_str(p[2])
     << "," << as_str(p[3]) << "," << as_str(p[4]) << "," << as_str(p[5]) << ","
     << as_str(p[6]) << "," << as_str(p[7]) << ")";
  return os;
}

struct alignas(32) flt8 {
  using mask_type = msk_f8;
  using value_type = f8_vt;
  using simd_type = f8_st;
  constexpr static std::size_t const alignment = 32;
  constexpr static std::size_t const dim = f8_simd::size;

  static flt8 load(float const* p) { return flt8(_mm256_load_ps(p)); }
  static flt8 loadu(float const* p) { return flt8(_mm256_loadu_ps(p)); }

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4201)
#endif
  union alignas(32) {
    simd_type data;
    struct {
      value_type f0, f1, f2, f3, f4, f5, f6, f7;
    };
    struct {
      value_type _0, _1, _2, _3, _4, _5, _6, _7;
    };
  };
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

  flt8() = default;
  flt8(value_type f0,
       value_type f1,
       value_type f2,
       value_type f3,
       value_type f4,
       value_type f5,
       value_type f6,
       value_type f7)
      : f0(f0), f1(f1), f2(f2), f3(f3), f4(f4), f5(f5), f6(f6), f7(f7) {}

  explicit flt8(simd_type data) : data(data) {}
  explicit flt8(value_type v)
      : f0(v), f1(v), f2(v), f3(v), f4(v), f5(v), f6(v), f7(v) {}

  operator simd_type() { return data; }

  flt8(flt8 const& other) : data(other.data) {}

  f8_vt& operator[](int i) { return (&f0)[i]; }

  f8_vt operator[](int i) const { return (&f0)[i]; }

  template <typename F>
  flt8 transform(F f) const {
    return flt8(f(f0), f(f1), f(f2), f(f3), f(f4), f(f5), f(f6), f(f7));
  }

  flt8& operator=(flt8 const& other) = default;
  flt8& operator=(flt8&& other) = default;

  flt8(flt8&& other) = default;

  flt8 operator+(flt8 const other) const {
    return flt8(_mm256_add_ps(data, other.data));
  }

  flt8 operator-(flt8 const other) const {
    return flt8(_mm256_sub_ps(data, other.data));
  }

  flt8 operator*(flt8 const other) const {
    return flt8(_mm256_mul_ps(data, other.data));
  }

  flt8 operator/(flt8 const other) const {
    return flt8(_mm256_div_ps(data, other.data));
  }

  flt8 min(flt8 const other) const {
    return flt8(_mm256_min_ps(data, other.data));
  }

  flt8 max(flt8 const other) const {
    return flt8(_mm256_max_ps(data, other.data));
  }

  flt8 operator+(f8_vt other) const {
    return flt8(_mm256_add_ps(data, _mm256_set1_ps(other)));
  }

  flt8 operator-(f8_vt other) const {
    return flt8(_mm256_sub_ps(data, _mm256_set1_ps(other)));
  }

  flt8 operator*(f8_vt other) const {
    return flt8(_mm256_mul_ps(data, _mm256_set1_ps(other)));
  }

  flt8 operator/(f8_vt other) const {
    return flt8(_mm256_div_ps(data, _mm256_set1_ps(other)));
  }

  flt8& operator+=(flt8 const other) {
    data = _mm256_add_ps(data, other.data);
    return *this;
  }

  flt8& operator-=(flt8 const other) {
    data = _mm256_sub_ps(data, other.data);
    return *this;
  }

  flt8& operator*=(flt8 const other) {
    data = _mm256_mul_ps(data, other.data);
    return *this;
  }

  flt8& operator/=(flt8 const other) {
    data = _mm256_div_ps(data, other.data);
    return *this;
  }

  flt8& operator+=(f8_vt other) {
    data = _mm256_add_ps(data, _mm256_set1_ps(other));
    return *this;
  }

  flt8& operator-=(f8_vt other) {
    data = _mm256_sub_ps(data, _mm256_set1_ps(other));
    return *this;
  }

  flt8& operator*=(f8_vt other) {
    data = _mm256_mul_ps(data, _mm256_set1_ps(other));
    return *this;
  }

  flt8& operator/=(f8_vt other) {
    data = _mm256_div_ps(data, _mm256_set1_ps(other));
    return *this;
  }

  flt8 operator-() const {
    return flt8(_mm256_sub_ps(_mm256_setzero_ps(), data));
  }

  flt8 reciprocal() const {
    return flt8(_mm256_div_ps(_mm256_set1_ps(1.0), data));
  }

  flt8 square() const { return flt8(_mm256_mul_ps(data, data)); }

  flt8 sqrt() const { return flt8(_mm256_sqrt_ps(data)); }

  flt8 rsqrt() const {
    return flt8(_mm256_div_ps(_mm256_set1_ps(1.0), _mm256_sqrt_ps(data)));
  }

  static flt8 zeros() { return flt8(_mm256_setzero_ps()); }

  static flt8 ones() { return flt8(_mm256_set1_ps(1.0f)); }

  static flt8 set1(f8_vt x) { return flt8(_mm256_set1_ps(x)); }

  flt8 abs() const {
    return flt8(_mm256_andnot_ps(_mm256_set1_ps(-0.0), data));
  }

  flt8 floor() const { return flt8(_mm256_floor_ps(data)); }

  flt8 ceil() const { return flt8(_mm256_ceil_ps(data)); }

  flt8 round() const {
    return flt8(_mm256_round_ps(data, _MM_FROUND_TO_NEAREST_INT));
  }

  flt8 trunc() const { return flt8(_mm256_round_ps(data, _MM_FROUND_TO_ZERO)); }

  flt8 round_up() const {
    return flt8(_mm256_round_ps(data, _MM_FROUND_TO_POS_INF));
  }

  flt8 round_down() const {
    return flt8(_mm256_round_ps(data, _MM_FROUND_TO_NEG_INF));
  }

  flt8 round_half_to_even() const {
    return flt8(
        _mm256_round_ps(data, _MM_FROUND_TO_NEG_INF | _MM_FROUND_TO_POS_INF));
  }

  flt8 round_half_to_odd() const {
    return flt8(_mm256_round_ps(data, _MM_FROUND_TO_NEG_INF |
                                          _MM_FROUND_TO_POS_INF |
                                          _MM_FROUND_NO_EXC));
  }

  flt8 round_half_away_from_zero() const {
    return flt8(
        _mm256_round_ps(data, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
  }

  flt8 round_half_towards_zero() const {
    return flt8(
        _mm256_round_ps(data, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
  }

  flt8 round_half_to_zero() const {
    return flt8(_mm256_round_ps(data, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }

  flt8 round_half_towards_neg_infinity() const {
    return flt8(_mm256_round_ps(data, _MM_FROUND_TO_NEG_INF));
  }

  flt8 round_half_towards_pos_infinity() const {
    return flt8(_mm256_round_ps(data, _MM_FROUND_TO_POS_INF));
  }

  flt8 clamp(flt8 const min, flt8 const max) const {
    return flt8(_mm256_min_ps(_mm256_max_ps(data, min.data), max.data));
  }

  flt8 clamp(f8_vt const min, f8_vt const max) const {
    return clamp(set1(min), set1(max));
  }

  f8_vt horizontal_add() const { return detail::hadd_m256(data); }

  f8_vt horizontal_mul() const { return detail::hmul_m256(data); }

  mask_type operator>(flt8 const other) const {
    return mask_type(_mm256_cmp_ps(data, other.data, _CMP_GT_OQ));
  }

  mask_type operator<(flt8 const other) const {
    return mask_type(_mm256_cmp_ps(data, other.data, _CMP_LT_OQ));
  }

  mask_type operator>=(flt8 const other) const {
    return mask_type(_mm256_cmp_ps(data, other.data, _CMP_GE_OQ));
  }

  mask_type operator<=(flt8 const other) const {
    return mask_type(_mm256_cmp_ps(data, other.data, _CMP_LE_OQ));
  }

  mask_type operator==(flt8 const other) const {
    return mask_type(_mm256_cmp_ps(data, other.data, _CMP_EQ_OQ));
  }

  mask_type operator!=(flt8 const other) const {
    return mask_type(_mm256_cmp_ps(data, other.data, _CMP_NEQ_OQ));
  }

  mask_type operator>(f8_vt other) const {
    return mask_type(_mm256_cmp_ps(data, _mm256_set1_ps(other), _CMP_GT_OQ));
  }

  mask_type operator<(f8_vt other) const {
    return mask_type(_mm256_cmp_ps(data, _mm256_set1_ps(other), _CMP_LT_OQ));
  }

  mask_type operator>=(f8_vt other) const {
    return mask_type(_mm256_cmp_ps(data, _mm256_set1_ps(other), _CMP_GE_OQ));
  }

  mask_type operator<=(f8_vt other) const {
    return mask_type(_mm256_cmp_ps(data, _mm256_set1_ps(other), _CMP_LE_OQ));
  }

  mask_type operator==(f8_vt other) const {
    return mask_type(_mm256_cmp_ps(data, _mm256_set1_ps(other), _CMP_EQ_OQ));
  }

  mask_type operator!=(f8_vt other) const {
    return mask_type(_mm256_cmp_ps(data, _mm256_set1_ps(other), _CMP_NEQ_OQ));
  }

  mask_type is_close(flt8 const other, f8_vt threshold) const {
    return (*this - other).abs() < threshold;
  }

  mask_type is_nan() {
    return mask_type(_mm256_cmp_ps(data, data, _CMP_UNORD_Q));
  }

  mask_type is_inf() {
    f8_st const inf_mask =
        _mm256_set1_ps(std::numeric_limits<f8_vt>::infinity());

    // Compare x with infinity (this catches both positive and negative
    // infinity)
    f8_st cmp_result = _mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), data), inf_mask, _CMP_EQ_OQ);

    return mask_type(cmp_result);
  }

  mask_type is_finite() { return ~(is_nan() | is_inf()); }

  int make_mask() const { return _mm256_movemask_ps(data); }

#ifndef AKS_DO_NOT_USE_SVML_INTRINSICS

  flt8 exp() const { return flt8(_mm256_exp_ps(data)); }

  flt8 log() const { return flt8(_mm256_log_ps(data)); }

  flt8 sin() const { return flt8(_mm256_sin_ps(data)); }

  flt8 cos() const { return flt8(_mm256_cos_ps(data)); }

  flt8 tan() const { return flt8(_mm256_tan_ps(data)); }

  flt8 sinh() const { return flt8(_mm256_sinh_ps(data)); }

  flt8 cosh() const { return flt8(_mm256_cosh_ps(data)); }

  flt8 tanh() const { return flt8(_mm256_tanh_ps(data)); }

  flt8 asin() const { return flt8(_mm256_asin_ps(data)); }

  flt8 acos() const { return flt8(_mm256_acos_ps(data)); }

  flt8 atan() const { return flt8(_mm256_atan_ps(data)); }

  flt8 sind() const { return flt8(_mm256_sind_ps(data)); }

  flt8 cosd() const { return flt8(_mm256_cosd_ps(data)); }

  flt8 tand() const { return flt8(_mm256_tand_ps(data)); }

  flt8 asinh() const { return flt8(_mm256_asinh_ps(data)); }

  flt8 acosh() const { return flt8(_mm256_acosh_ps(data)); }

  flt8 atanh() const { return flt8(_mm256_atanh_ps(data)); }

  flt8 erf() const { return flt8(_mm256_erf_ps(data)); }

  flt8 erfc() const { return flt8(_mm256_erfc_ps(data)); }

  flt8 erfinv() const { return flt8(_mm256_erfinv_ps(data)); }

  flt8 cdfnorm() const { return flt8(_mm256_cdfnorm_ps(data)); }

  flt8 cdfnorminv() const { return flt8(_mm256_cdfnorminv_ps(data)); }
#else

  static f8_vt d2r(f8_vt degree) { return degree * (std::numbers::pi / 180.0); }

  flt8 log() const {
    return transform([](d4_vt x) { return std::log(x); });
  }

  flt8 exp() const {
    return transform([](d4_vt x) { return std::exp(x); });
  }

  flt8 sin() const {
    return transform([](f8_vt x) { return std::sin(x); });
  }

  flt8 cos() const {
    return transform([](f8_vt x) { return std::cos(x); });
  }

  flt8 tan() const {
    return transform([](f8_vt x) { return std::tan(x); });
  }

  flt8 sinh() const {
    return transform([](f8_vt x) { return std::sinh(x); });
  }

  flt8 cosh() const {
    return transform([](f8_vt x) { return std::cosh(x); });
  }

  flt8 tanh() const {
    return transform([](f8_vt x) { return std::tanh(x); });
  }

  flt8 asin() const {
    return transform([](f8_vt x) { return std::asin(x); });
  }

  flt8 acos() const {
    return transform([](f8_vt x) { return std::acos(x); });
  }

  flt8 atan() const {
    return transform([](f8_vt x) { return std::atan(x); });
  }

  flt8 sind() const {
    return transform([](f8_vt x) { return std::sin(d2r(x)); });
  }

  flt8 cosd() const {
    return transform([](f8_vt x) { return std::cos(d2r(x)); });
  }

  flt8 tand() const {
    return transform([](f8_vt x) { return std::tan(d2r(x)); });
  }

  flt8 asinh() const {
    return transform([](f8_vt x) { return std::asinh(x); });
  }

  flt8 acosh() const {
    return transform([](f8_vt x) { return std::acosh(x); });
  }

  flt8 atanh() const {
    return transform([](f8_vt x) { return std::atanh(x); });
  }

  flt8 erf() const {
    return transform([](f8_vt x) { return std::erf(x); });
  }

  flt8 erfc() const {
    return transform([](f8_vt x) { return std::erfc(x); });
  }

  flt8 erfinv() const {
    return ones() / transform([](f8_vt x) { return std::erf(x); });
  }

  flt8 cdfnorm() const {
    auto normcdf = [](f8_vt x, f8_vt mean = 0.0, f8_vt stddev = 1.0) {
      // Standardize the input
      f8_vt z = (x - mean) / stddev;

      // Constants
      f8_vt const a1 = 0.254829592;
      f8_vt const a2 = -0.284496736;
      f8_vt const a3 = 1.421413741;
      f8_vt const a4 = -1.453152027;
      f8_vt const a5 = 1.061405429;
      f8_vt const p = 0.3275911;

      // Save the sign of z
      int sign = (z < 0) ? -1 : 1;
      z = std::fabs(z) / std::sqrt(2.0);

      // A&S formula 7.1.26
      f8_vt t = 1.0 / (1.0 + p * z);
      f8_vt y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
                          std::exp(-z * z);

      return 0.5 * (1.0 + sign * y);
    };

    return transform([&normcdf](f8_vt x) { return normcdf(x); });
  }

  flt8 cdfnorminv() const {
    auto inverseNormCDF = [](f8_vt p, f8_vt mean = 0.0, f8_vt stddev = 1.0) {
      if (p <= 0.0 || p >= 1.0) {
        return std::numeric_limits<f8_vt>::quiet_NaN();
      }

      // Coefficients in rational approximations
      f8_vt const a1 = -3.969683028665376e+01;
      f8_vt const a2 = 2.209460984245205e+02;
      f8_vt const a3 = -2.759285104469687e+02;
      f8_vt const a4 = 1.383577518672690e+02;
      f8_vt const a5 = -3.066479806614716e+01;
      f8_vt const a6 = 2.506628277459239e+00;

      f8_vt const b1 = -5.447609879822406e+01;
      f8_vt const b2 = 1.615858368580409e+02;
      f8_vt const b3 = -1.556989798598866e+02;
      f8_vt const b4 = 6.680131188771972e+01;
      f8_vt const b5 = -1.328068155288572e+01;

      f8_vt const c1 = -7.784894002430293e-03;
      f8_vt const c2 = -3.223964580411365e-01;
      f8_vt const c3 = -2.400758277161838e+00;
      f8_vt const c4 = -2.549732539343734e+00;
      f8_vt const c5 = 4.374664141464968e+00;
      f8_vt const c6 = 2.938163982698783e+00;

      f8_vt const d1 = 7.784695709041462e-03;
      f8_vt const d2 = 3.224671290700398e-01;
      f8_vt const d3 = 2.445134137142996e+00;
      f8_vt const d4 = 3.754408661907416e+00;

      // Define break-points
      f8_vt const p_low = 0.02425;
      f8_vt const p_high = 1 - p_low;

      // Rational approximation for lower region
      if (p < p_low) {
        f8_vt q = std::sqrt(-2 * std::log(p));
        f8_vt x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                  ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
        return mean - x * stddev;
      }

      // Rational approximation for central region
      if (p <= p_high) {
        f8_vt q = p - 0.5;
        f8_vt r = q * q;
        f8_vt x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) *
                  q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
        return mean + x * stddev;
      }

      // Rational approximation for upper region
      if (p < 1) {
        f8_vt q = std::sqrt(-2 * std::log(1 - p));
        f8_vt x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                  ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
        return mean + x * stddev;
      }

      // Case when p = 1
      return std::numeric_limits<f8_vt>::infinity();
    };

    return transform([&inverseNormCDF](f8_vt x) { return inverseNormCDF(x); });
  }

#endif
};

flt8 operator+(f8_vt x, flt8 const y) {
  return y + x;
}

flt8 operator-(f8_vt x, flt8 const y) {
  return -y + x;
}

flt8 operator*(f8_vt x, flt8 const y) {
  return y * x;
}

flt8 operator/(f8_vt x, flt8 const y) {
  return flt8::set1(x) / y;
}

flt8 if_then_else(flt8::mask_type const mask,
                  flt8 const true_value,
                  flt8 const false_value) {
  return flt8(_mm256_blendv_ps(false_value.data, true_value.data, mask.data));
}

flt8 fmadd(flt8 const x, flt8 const y, flt8 const z) {
  return flt8(_mm256_fmadd_ps(x.data, y.data, z.data));
}

flt8 fmsub(flt8 const x, flt8 const y, flt8 const z) {
  return flt8(_mm256_fmsub_ps(x.data, y.data, z.data));
}

flt8 fnmadd(flt8 const x, flt8 const y, flt8 const z) {
  return flt8(_mm256_fnmadd_ps(x.data, y.data, z.data));
}

flt8 fnmsub(flt8 const x, flt8 const y, flt8 const z) {
  return flt8(_mm256_fnmsub_ps(x.data, y.data, z.data));
}

template <typename F>
flt8 transform(flt8 const x, flt8 const y, F f) {
  return flt8(f(x.f0, y.f0), f(x.f1, y.f1), f(x.f2, y.f2), f(x.f3, y.f3),
              f(x.f4, y.f4), f(x.f5, y.f5), f(x.f6, y.f6), f(x.f7, y.f7));
}

flt8 atan2(flt8 const x, flt8 const y) {
#ifndef AKS_DO_NOT_USE_SVML_INTRINSICS
  return flt8(_mm256_atan2_ps(x.data, y.data));
#else
  auto atan2f = [](f8_vt x, f8_vt y) { return std::atan2(x, y); };
  return transform(x, y, atan2f);
#endif
}

flt8 pow(flt8 const x, flt8 const y) {
#ifndef AKS_DO_NOT_USE_SVML_INTRINSICS
  return flt8(_mm256_pow_ps(x.data, y.data));
#else
  auto powf = [](f8_vt x, f8_vt y) { return std::pow(x, y); };
  return transform(x, y, powf);
#endif
}

std::ostream& operator<<(std::ostream& os, flt8 const p) {
  os << "flt8( " << p.f0 << ", " << p.f1 << ", " << p.f2 << ", " << p.f3 << ", "
     << p.f4 << ", " << p.f5 << ", " << p.f6 << ", " << p.f7 << " )";
  return os;
}

}  // namespace aks

#endif  // FLOAT8D_HPP
