#ifndef DOUBLE8D_HPP
#define DOUBLE8D_HPP

#include <immintrin.h>
#include <smmintrin.h>

#include <bitset>
#include <cmath>
#include <numbers>
#include <ostream>

// NOLINTBEGIN

#include "simd_type.hpp"

namespace aks {
using d8_vt = double;
using d8_st = typename simd_type<d8_vt, 8>::type;
using d8_mt = typename simd_type<d8_vt, 8>::mask_type;

namespace detail {
double hadd_m512d(__m512d x) {
  return _mm512_reduce_add_pd(x);
}

double hmul_m512d(__m512d x) {
  return _mm512_reduce_mul_pd(x);
}
}  // namespace detail

struct msk_d8 {
  using simd_type                              = d8_mt;
  constexpr static std::size_t const size      = 8;
  constexpr static std::size_t       alignment = 64;

  d8_mt data;

  explicit msk_d8(d8_mt data) : data(data) {}

  msk_d8 and_(msk_d8 other) const {
    return msk_d8{_kand_mask8(data, other.data)};
  }

  msk_d8 or_(msk_d8 other) const {
    return msk_d8{_kor_mask8(data, other.data)};
  }

  msk_d8 xor_(msk_d8 other) const {
    return msk_d8{_kxor_mask8(data, other.data)};
  }

  msk_d8 not_() const { return msk_d8{_knot_mask8(data)}; }

  msk_d8 and_not(msk_d8 other) const {
    return msk_d8{_kandn_mask8(other.data, data)};
  }

  msk_d8 or_not(msk_d8 other) const { return ((*this).or_(other)).not_(); }

  msk_d8 xor_not(msk_d8 other) const { return ((*this).xor_(other)).not_(); };

  int mask() const { return (int)data; }

  int popcount() const {
#ifdef _WIN32
    return __popcnt(mask());
#else
    return __builtin_popcount(static_cast<unsigned int>(mask()));
#endif
  }

  bool operator[](int i) const { return mask() & (1 << i); }

  bool operator==(msk_d8 other) const { return mask() == other.mask(); }

  bool operator!=(msk_d8 other) const { return mask() != other.mask(); }

  msk_d8 operator&(msk_d8 other) const { return and_(other); }

  msk_d8 operator|(msk_d8 other) const { return or_(other); }

  msk_d8 operator^(msk_d8 other) const { return xor_(other); }

  msk_d8 operator~() const { return not_(); }

  static msk_d8 create_mask(char mask) { return msk_d8(mask); }

  static msk_d8 create_mask(bool b0,
                            bool b1,
                            bool b2,
                            bool b3,
                            bool b4,
                            bool b5,
                            bool b6,
                            bool b7) {
    return msk_d8((d8_mt)((b0 << 0) | (b1 << 1) | (b2 << 2) | (b3 << 3) |
                          (b4 << 4) | (b5 << 5) | (b6 << 6) | (b7 << 7)));
  }

  static msk_d8
  create_mask(int b0, int b1, int b2, int b3, int b4, int b5, int b6, int b7) {
    return create_mask(bool(b0), bool(b1), bool(b2), bool(b3), bool(b4),
                       bool(b5), bool(b6), bool(b7));
  }

  static msk_d8 create_mask(std::bitset<8> mask) {
    return create_mask(mask[0], mask[1], mask[2], mask[3], mask[4], mask[5],
                       mask[6], mask[7]);
  }
};

std::ostream& operator<<(std::ostream& os, msk_d8 const& mask) {
  auto as_str = [](bool mask) {
    if (mask == 0) {
      return "F";
    } else {
      return "T";
    }
  };

  os << "[" << as_str(mask[0]) << as_str(mask[1]) << as_str(mask[2])
     << as_str(mask[3]) << as_str(mask[4]) << as_str(mask[5]) << as_str(mask[6])
     << as_str(mask[7]) << "]";
  return os;
}

struct alignas(64) dbl8 {
  using mask_type                   = msk_d8;
  using value_type                  = d8_vt;
  using simd_type                   = d8_st;
  constexpr static size_t alignment = 64;
  constexpr static size_t dim       = 8;

  union alignas(64) {
    simd_type data;
    struct {
      value_type _0, _1, _2, _3, _4, _5, _6, _7;
    };
  };

  dbl8() = default;
  dbl8(value_type a0,
       value_type a1,
       value_type a2,
       value_type a3,
       value_type a4,
       value_type a5,
       value_type a6,
       value_type a7)
      : _0(a0), _1(a1), _2(a2), _3(a3), _4(a4), _5(a5), _6(a6), _7(a7) {}

  explicit dbl8(simd_type d) : data(d) {}
  explicit dbl8(value_type v) : data(_mm512_set1_pd(v)) {}

  static dbl8 load(double const* p) { return dbl8(_mm512_load_pd(p)); }
  static dbl8 loadu(double const* p) { return dbl8(_mm512_loadu_pd(p)); }

  operator simd_type() const { return data; }

  dbl8(dbl8 const& other)            = default;
  dbl8(dbl8&& other)                 = default;
  dbl8& operator=(dbl8 const& other) = default;
  dbl8& operator=(dbl8&& other)      = default;

  d8_vt& operator[](int i) { return (&_0)[i]; }
  d8_vt  operator[](int i) const { return (&_0)[i]; }

  template <typename F>
  dbl8 transform(F f) const {
    return dbl8(f(_0), f(_1), f(_2), f(_3), f(_4), f(_5), f(_6), f(_7));
  }

  template <typename F>
  dbl8 T(F f) const {
    return transform(f);
  }

  dbl8 operator+(dbl8 const other) const {
    return dbl8(_mm512_add_pd(data, other.data));
  }

  dbl8 operator-(dbl8 const other) const {
    return dbl8(_mm512_sub_pd(data, other.data));
  }

  dbl8 operator*(dbl8 const other) const {
    return dbl8(_mm512_mul_pd(data, other.data));
  }

  dbl8 operator/(dbl8 const other) const {
    return dbl8(_mm512_div_pd(data, other.data));
  }

  dbl8 min(dbl8 const other) const {
    return dbl8(_mm512_min_pd(data, other.data));
  }

  dbl8 max(dbl8 const other) const {
    return dbl8(_mm512_max_pd(data, other.data));
  }

  dbl8 operator+(value_type other) const {
    return dbl8(_mm512_add_pd(data, _mm512_set1_pd(other)));
  }

  dbl8 operator-(value_type other) const {
    return dbl8(_mm512_sub_pd(data, _mm512_set1_pd(other)));
  }

  dbl8 operator*(value_type other) const {
    return dbl8(_mm512_mul_pd(data, _mm512_set1_pd(other)));
  }

  dbl8 operator/(value_type other) const {
    return dbl8(_mm512_div_pd(data, _mm512_set1_pd(other)));
  }

  dbl8& operator+=(dbl8 const other) {
    data = _mm512_add_pd(data, other.data);
    return *this;
  }

  dbl8& operator-=(dbl8 const other) {
    data = _mm512_sub_pd(data, other.data);
    return *this;
  }

  dbl8& operator*=(dbl8 const other) {
    data = _mm512_mul_pd(data, other.data);
    return *this;
  }

  dbl8& operator/=(dbl8 const other) {
    data = _mm512_div_pd(data, other.data);
    return *this;
  }

  dbl8& operator+=(value_type other) {
    data = _mm512_add_pd(data, _mm512_set1_pd(other));
    return *this;
  }

  dbl8& operator-=(value_type other) {
    data = _mm512_sub_pd(data, _mm512_set1_pd(other));
    return *this;
  }

  dbl8& operator*=(value_type other) {
    data = _mm512_mul_pd(data, _mm512_set1_pd(other));
    return *this;
  }

  dbl8& operator/=(value_type other) {
    data = _mm512_div_pd(data, _mm512_set1_pd(other));
    return *this;
  }

  dbl8 operator-() const {
    return dbl8(_mm512_sub_pd(_mm512_setzero_pd(), data));
  }

  dbl8 reciprocal() const {
    return dbl8(_mm512_div_pd(_mm512_set1_pd(1.0), data));
  }

  dbl8 square() const { return dbl8(_mm512_mul_pd(data, data)); }

  dbl8 sqrt() const { return dbl8(_mm512_sqrt_pd(data)); }

  dbl8 rsqrt() const { return dbl8(_mm512_div_pd(_mm512_set1_pd(1.0), data)); }

  static dbl8 zero() { return dbl8(_mm512_setzero_pd()); }

  static dbl8 ones() { return dbl8(_mm512_set1_pd(1.0)); }

  static dbl8 set1(value_type x) { return dbl8(_mm512_set1_pd(x)); }

  dbl8 abs() const { return dbl8(_mm512_abs_pd(data)); }

  dbl8 floor() const { return dbl8(_mm512_floor_pd(data)); }

  dbl8 ceil() const { return dbl8(_mm512_ceil_pd(data)); }

  dbl8 round() const {
    return dbl8(_mm512_roundscale_pd(data, _MM_FROUND_TO_NEAREST_INT));
  }

  dbl8 round_to_zero() const {
    return dbl8(_mm512_roundscale_pd(data, _MM_FROUND_TO_ZERO));
  }

  dbl8 trunc() const {
    return dbl8(_mm512_roundscale_pd(data, _MM_FROUND_TO_ZERO));
  }

  dbl8 round_up() const {
    return dbl8(_mm512_roundscale_pd(data, _MM_FROUND_TO_POS_INF));
  }

  dbl8 round_down() const {
    return dbl8(_mm512_roundscale_pd(data, _MM_FROUND_TO_NEG_INF));
  }

  dbl8 round_half_to_even() const {
    return dbl8(_mm512_roundscale_pd(
        data, _MM_FROUND_TO_NEG_INF | _MM_FROUND_TO_POS_INF));
  }

  dbl8 round_half_to_odd() const {
    return dbl8(_mm512_roundscale_pd(data, _MM_FROUND_TO_NEG_INF |
                                               _MM_FROUND_TO_POS_INF |
                                               _MM_FROUND_NO_EXC));
  }

  dbl8 round_half_away_from_zero() const {
    return dbl8(
        _mm512_roundscale_pd(data, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
  }

  dbl8 round_half_towards_zero() const {
    return dbl8(
        _mm512_roundscale_pd(data, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
  }

  dbl8 round_half_to_zero() const {
    return dbl8(
        _mm512_roundscale_pd(data, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }

  dbl8 round_half_towards_neg_infinity() const {
    return dbl8(
        _mm512_roundscale_pd(data, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
  }

  dbl8 round_half_towards_pos_infinity() const {
    return dbl8(
        _mm512_roundscale_pd(data, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
  }

  dbl8 clamp(dbl8 const min, dbl8 const max) const {
    return dbl8(_mm512_min_pd(_mm512_max_pd(data, min.data), max.data));
  }

  dbl8 clamp(value_type min, value_type max) const {
    return clamp(dbl8::set1(min), dbl8::set1(max));
  }

  value_type horizontal_add() const { return detail::hadd_m512d(data); }

  value_type horizontal_mul() const { return detail::hmul_m512d(data); }

  mask_type operator>(dbl8 const other) const {
    return mask_type(_mm512_cmp_pd_mask(data, other.data, _CMP_GT_OQ));
  }

  mask_type operator<(dbl8 const other) const {
    return mask_type(_mm512_cmp_pd_mask(data, other.data, _CMP_LT_OQ));
  }

  mask_type operator>=(dbl8 const other) const {
    return mask_type(_mm512_cmp_pd_mask(data, other.data, _CMP_GE_OQ));
  }

  mask_type operator<=(dbl8 const other) const {
    return mask_type(_mm512_cmp_pd_mask(data, other.data, _CMP_LE_OQ));
  }

  mask_type operator==(dbl8 const other) const {
    return mask_type(_mm512_cmp_pd_mask(data, other.data, _CMP_EQ_OQ));
  }

  mask_type operator!=(dbl8 const other) const {
    return mask_type(_mm512_cmp_pd_mask(data, other.data, _CMP_NEQ_OQ));
  }

  mask_type operator>(d8_vt other) const {
    return mask_type(
        _mm512_cmp_pd_mask(data, _mm512_set1_pd(other), _CMP_GT_OQ));
  }

  mask_type operator<(d8_vt other) const {
    return mask_type(
        _mm512_cmp_pd_mask(data, _mm512_set1_pd(other), _CMP_LT_OQ));
  }

  mask_type operator>=(d8_vt other) const {
    return mask_type(
        _mm512_cmp_pd_mask(data, _mm512_set1_pd(other), _CMP_GE_OQ));
  }

  mask_type operator<=(d8_vt other) const {
    return mask_type(
        _mm512_cmp_pd_mask(data, _mm512_set1_pd(other), _CMP_LE_OQ));
  }

  mask_type operator==(d8_vt other) const {
    return mask_type(
        _mm512_cmp_pd_mask(data, _mm512_set1_pd(other), _CMP_EQ_OQ));
  }

  mask_type operator!=(d8_vt other) const {
    return mask_type(
        _mm512_cmp_pd_mask(data, _mm512_set1_pd(other), _CMP_NEQ_OQ));
  }

  mask_type is_close(dbl8 const other, d8_vt threshold) const {
    return (*this - other).abs() < threshold;
  }

  mask_type is_nan() const {
    return mask_type(_mm512_cmp_pd_mask(data, data, _CMP_UNORD_Q));
  }

  mask_type is_inf() const {
    d8_st const inf_mask =
        _mm512_set1_pd(std::numeric_limits<d8_vt>::infinity());

    // Compare x with infinity (this catches both positive and negative
    // infinity)
    d8_mt cmp_result = _mm512_cmp_pd_mask(
        _mm512_andnot_pd(_mm512_set1_pd(-0.0), data), inf_mask, _CMP_EQ_OQ);

    return mask_type(cmp_result);
  }

  mask_type is_finite() const { return ~(is_nan() | is_inf()); }

#ifndef AKS_DO_NOT_USE_SVML_INTRINSICS

  dbl8 exp() const { return dbl8(_mm512_exp_pd(data)); }
  dbl8 log() const { return dbl8(_mm512_log_pd(data)); }
  dbl8 sin() const { return dbl8(_mm512_sin_pd(data)); }
  dbl8 cos() const { return dbl8(_mm512_cos_pd(data)); }
  dbl8 tan() const { return dbl8(_mm512_tan_pd(data)); }
  dbl8 sinh() const { return dbl8(_mm512_sinh_pd(data)); }
  dbl8 cosh() const { return dbl8(_mm512_cosh_pd(data)); }
  dbl8 tanh() const { return dbl8(_mm512_tanh_pd(data)); }
  dbl8 asin() const { return dbl8(_mm512_asin_pd(data)); }
  dbl8 acos() const { return dbl8(_mm512_acos_pd(data)); }
  dbl8 atan() const { return dbl8(_mm512_atan_pd(data)); }
  dbl8 sind() const { return dbl8(_mm512_sind_pd(data)); }
  dbl8 cosd() const { return dbl8(_mm512_cosd_pd(data)); }
  dbl8 tand() const { return dbl8(_mm512_tand_pd(data)); }
  dbl8 asinh() const { return dbl8(_mm512_asinh_pd(data)); }
  dbl8 acosh() const { return dbl8(_mm512_acosh_pd(data)); }
  dbl8 atanh() const { return dbl8(_mm512_atanh_pd(data)); }
  dbl8 erf() const { return dbl8(_mm512_erf_pd(data)); }
  dbl8 erfc() const { return dbl8(_mm512_erfc_pd(data)); }
  dbl8 erfinv() const { return dbl8(_mm512_erfinv_pd(data)); }
  dbl8 cdfnorm() const { return dbl8(_mm512_cdfnorm_pd(data)); }
  dbl8 cdfnorminv() const { return dbl8(_mm512_cdfnorminv_pd(data)); }

#else

  static d8_vt d2r(d8_vt degree) { return degree * (std::numbers::pi / 180.0); }

  dbl8 log() const {
    return T([](d8_vt x) { return std::log(x); });
  }

  dbl8 exp() const {
    return T([](d8_vt x) { return std::exp(x); });
  }

  dbl8 sin() const {
    return T([](d8_vt x) { return std::sin(x); });
  }

  dbl8 cos() const {
    return T([](d8_vt x) { return std::cos(x); });
  }

  dbl8 tan() const {
    return T([](d8_vt x) { return std::tan(x); });
  }

  dbl8 sinh() const {
    return T([](d8_vt x) { return std::sinh(x); });
  }

  dbl8 cosh() const {
    return T([](d8_vt x) { return std::cosh(x); });
  }

  dbl8 tanh() const {
    return T([](d8_vt x) { return std::tanh(x); });
  }

  dbl8 asin() const {
    return T([](d8_vt x) { return std::asin(x); });
  }

  dbl8 acos() const {
    return T([](d8_vt x) { return std::acos(x); });
  }

  dbl8 atan() const {
    return T([](d8_vt x) { return std::atan(x); });
  }

  dbl8 sind() const {
    return T([](d8_vt x) { return std::sin(d2r(x)); });
  }

  dbl8 cosd() const {
    return T([](d8_vt x) { return std::cos(d2r(x)); });
  }

  dbl8 tand() const {
    return T([](d8_vt x) { return std::tan(d2r(x)); });
  }

  dbl8 asinh() const {
    return T([](d8_vt x) { return std::asinh(x); });
  }

  dbl8 acosh() const {
    return T([](d8_vt x) { return std::acosh(x); });
  }

  dbl8 atanh() const {
    return T([](d8_vt x) { return std::atanh(x); });
  }

  dbl8 erf() const {
    return T([](d8_vt x) { return std::erf(x); });
  }

  dbl8 erfc() const {
    return T([](d8_vt x) { return std::erfc(x); });
  }

  dbl8 erfinv() const {
    return ones() / T([](d8_vt ax) { return std::erf(ax); });
  }

  dbl8 cdfnorm() const {
    auto normcdf = [](d8_vt ax, d8_vt mean = 0.0, d8_vt stddev = 1.0) {
      // Standardize the input
      d8_vt z_ = (ax - mean) / stddev;

      // Constants
      d8_vt const a1 = 0.254829592;
      d8_vt const a2 = -0.284496736;
      d8_vt const a3 = 1.421413741;
      d8_vt const a4 = -1.453152027;
      d8_vt const a5 = 1.061405429;
      d8_vt const p  = 0.3275911;

      // Save the sign of z
      int sign = (z_ < 0) ? -1 : 1;
      z_       = std::fabs(z_) / std::sqrt(2.0);

      // A&S formula 7.1.26
      d8_vt t  = 1.0 / (1.0 + p * z_);
      d8_vt y_ = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
                           std::exp(-z_ * z_);

      return 0.5 * (1.0 + sign * y_);
    };

    return transform([&normcdf](d8_vt ax) { return normcdf(ax); });
  }

  dbl8 cdfnorminv() const {
    auto inverseNormCDF = [](d8_vt p, d8_vt mean = 0.0, d8_vt stddev = 1.0) {
      if (p <= 0.0 || p >= 1.0) {
        return std::numeric_limits<d8_vt>::quiet_NaN();
      }

      // Coefficients in rational approximations
      d8_vt const a1 = -3.969683028665376e+01;
      d8_vt const a2 = 2.209460984245205e+02;
      d8_vt const a3 = -2.759285104469687e+02;
      d8_vt const a4 = 1.383577518672690e+02;
      d8_vt const a5 = -3.066479806614716e+01;
      d8_vt const a6 = 2.506628277459239e+00;

      d8_vt const b1 = -5.447609879822406e+01;
      d8_vt const b2 = 1.615858368580409e+02;
      d8_vt const b3 = -1.556989798598866e+02;
      d8_vt const b4 = 6.680131188771972e+01;
      d8_vt const b5 = -1.328068155288572e+01;

      d8_vt const c1 = -7.784894002430293e-03;
      d8_vt const c2 = -3.223964580411365e-01;
      d8_vt const c3 = -2.400758277161838e+00;
      d8_vt const c4 = -2.549732539343734e+00;
      d8_vt const c5 = 4.374664141464968e+00;
      d8_vt const c6 = 2.938163982698783e+00;

      d8_vt const d1 = 7.784695709041462e-03;
      d8_vt const d2 = 3.224671290700398e-01;
      d8_vt const d3 = 2.445134137142996e+00;
      d8_vt const d4 = 3.754408661907416e+00;

      // Define break-points
      d8_vt const p_low  = 0.02425;
      d8_vt const p_high = 1 - p_low;

      // Rational approximation for lower region
      if (p < p_low) {
        d8_vt q  = std::sqrt(-2 * std::log(p));
        d8_vt x_ = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
        return mean - x_ * stddev;
      }

      // Rational approximation for central region
      if (p <= p_high) {
        d8_vt q  = p - 0.5;
        d8_vt r  = q * q;
        d8_vt x_ = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) *
                   q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
        return mean + x_ * stddev;
      }

      // Rational approximation for upper region
      if (p < 1) {
        d8_vt q  = std::sqrt(-2 * std::log(1 - p));
        d8_vt x_ = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
        return mean + x_ * stddev;
      }

      // Case when p = 1
      return std::numeric_limits<d8_vt>::infinity();
    };

    return transform(
        [&inverseNormCDF](d8_vt ax) { return inverseNormCDF(ax); });
  }

#endif
};

dbl8 operator+(d8_vt x, dbl8 const y) {
  return y + x;
}

dbl8 operator-(d8_vt x, dbl8 const y) {
  return dbl8::set1(x) - y;
}

dbl8 operator*(d8_vt x, dbl8 const y) {
  return y * x;
}

dbl8 operator/(d8_vt x, dbl8 const y) {
  return dbl8::set1(x) / y;
}

dbl8 if_then_else(dbl8::mask_type const mask,
                  dbl8 const            true_value,
                  dbl8 const            false_value) {
  return dbl8(
      _mm512_mask_blend_pd(mask.data, true_value.data, false_value.data));
}

dbl8 fmadd(dbl8 const a, dbl8 const b, dbl8 const c) {
  return dbl8(_mm512_fmadd_pd(a.data, b.data, c.data));
}

dbl8 fmsub(dbl8 const a, dbl8 const b, dbl8 const c) {
  return dbl8(_mm512_fmsub_pd(a.data, b.data, c.data));
}

dbl8 fnmadd(dbl8 const a, dbl8 const b, dbl8 const c) {
  return dbl8(_mm512_fnmadd_pd(a.data, b.data, c.data));
}

dbl8 fnmsub(dbl8 const a, dbl8 const b, dbl8 const c) {
  return dbl8(_mm512_fnmsub_pd(a.data, b.data, c.data));
}

template <typename F>
dbl8 transform(dbl8 const x, dbl8 const y, F f) {
  return dbl4(f(x._0, y._0), f(x._1, y._1), f(x._2, y._2), f(x._3, y._3),
              f(x._4, y._4), f(x._5, y._5), f(x._6, y._6), f(x._7, y._7));
}

dbl8 atan2(dbl8 const x, dbl8 const y) {
#ifndef AKS_DO_NOT_USE_SVML_INTRINSICS
  return dbl8(_mm512_atan2_pd(x.data, y.data));
#else
  auto atan2f = [](d8_vt ax, d8_vt ay) { return std::atan2(ax, ay); };
  return transform(x, y, atan2f);
#endif
}

dbl8 pow(dbl8 const x, dbl8 const y) {
#ifndef AKS_DO_NOT_USE_SVML_INTRINSICS
  return dbl8(_mm512_pow_pd(x.data, y.data));
#else
  auto powf = [](d8_vt ax, d8_vt ay) { return std::pow(ax, ay); };
  return transform(x, y, powf);
#endif
}

std::ostream& operator<<(std::ostream& os, dbl8 const& v) {
  os << "dbl8( " << v._0 << ", " << v._1 << ", " << v._2 << ", " << v._3 << ", "
     << v._4 << ", " << v._5 << ", " << v._6 << ", " << v._7 << " )";
  return os;
}

}  // namespace aks

#endif  // DOUBLE8D_HPP
