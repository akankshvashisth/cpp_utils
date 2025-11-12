
#ifndef AKS_SIMD2_D4_HPP
#define AKS_SIMD2_D4_HPP

#include <cmath>
#include <immintrin.h>
#include <ostream>

#include "common.hpp"

namespace aks {
namespace simd2 {

namespace d4 {
using simd_type = type_wrapper<__m256d, 0>;
using simd_mask = type_wrapper<__m256d, 1>;
constexpr static const size_t simd_size = 4;
static const simd_type zeros{_mm256_setzero_pd()};
static const simd_type neg_zeros{_mm256_set1_pd(-0.0)};

struct simd_data {
  double data[simd_size] = {0.0, 0.0, 0.0, 0.0};
};

simd_type simd_from(double d) { return {_mm256_set1_pd(d)}; }

simd_type simd_from(double d0, double d1, double d2, double d3) {
  return {_mm256_setr_pd(d0, d1, d2, d3)};
}

simd_type simd_from(double const *ds) { return {_mm256_loadu_pd(ds)}; }

simd_data simd_to_data(simd_type x) {
  simd_data ret;
  _mm256_storeu_pd(ret.data, x.data);
  return ret;
}

simd_data simd_mask_to_data(simd_mask x) {
  simd_data ret;
  _mm256_storeu_pd(ret.data, x.data);
  return ret;
}

simd_type if_then_else(simd_mask mask, simd_type x, simd_type y) {
  return {_mm256_blendv_pd(y.data, x.data, mask.data)};
}

std::ostream &operator<<(std::ostream &os, simd_type const &x) {
  simd_data d = simd_to_data(x);
  os << "{" << d.data[0] << "," << d.data[1] << "," << d.data[2] << ","
     << d.data[3] << "}";
  return os;
}

std::ostream &operator<<(std::ostream &os, simd_mask const &x) {
  simd_data d = simd_mask_to_data(x);
  auto b = [](double d) { return (d != 0 ? "T" : "F"); };
  os << "{" << b(d.data[0]) << "," << b(d.data[1]) << "," << b(d.data[2]) << ","
     << b(d.data[3]) << "}";
  return os;
}

simd_type operator+(simd_type x, simd_type y) {
  return {_mm256_add_pd(x.data, y.data)};
}

simd_type operator-(simd_type x, simd_type y) {
  return {_mm256_sub_pd(x.data, y.data)};
}

simd_type operator-(simd_type x) {
  return {_mm256_xor_pd(x.data, neg_zeros.data)};
}

simd_type operator*(simd_type x, simd_type y) {
  return {_mm256_mul_pd(x.data, y.data)};
}

simd_type operator/(simd_type x, simd_type y) {
  return {_mm256_div_pd(x.data, y.data)};
}

simd_type sqrt(simd_type x) { return {_mm256_sqrt_pd(x.data)}; }

simd_type abs(simd_type x) {
  return {_mm256_andnot_pd(neg_zeros.data, x.data)};
}

#ifdef _WIN32

simd_type pow(simd_type x, simd_type y) {
  return {_mm256_pow_pd(x.data, y.data)};
}

simd_type exp(simd_type x) { return {_mm256_exp_pd(x.data)}; }

simd_type log(simd_type x) { return {_mm256_log_pd(x.data)}; }

simd_type sin(simd_type x) { return {_mm256_sin_pd(x.data)}; }

simd_type atan2(simd_type x, simd_type y) {
  return {_mm256_atan2_pd(x.data, y.data)};
}
#else //
simd_type atan2(simd_type x, simd_type y) {
  simd_data xs = simd_to_data(x);
  simd_data ys = simd_to_data(y);
  auto atan2at = [&](size_t i) { return std::atan2(xs.data[i], ys.data[i]); };
  return {simd_from(atan2at(0), atan2at(1), atan2at(2), atan2at(3))};
}
#endif

simd_mask operator<(simd_type x, simd_type y) {
  return {_mm256_cmp_pd(x.data, y.data, _CMP_LT_OS)};
}

simd_mask operator>(simd_type x, simd_type y) {
  return {_mm256_cmp_pd(x.data, y.data, _CMP_GT_OS)};
}

bool any(simd_mask msk) {
  // return !_mm256_testz_pd(msk.data, msk.data);
  return _mm256_movemask_pd(msk.data) != 0;
}

bool all(simd_mask msk) {
  // return _mm256_testc_pd(msk.data, msk.data);
  return _mm256_movemask_pd(msk.data) == 0xF;
}

double horizontal_min(simd_type v) {
  __m128d low = _mm256_castpd256_pd128(v.data);
  __m128d high = _mm256_extractf128_pd(v.data, 1);
  __m128d min = _mm_min_pd(low, high);
  min = _mm_min_pd(min, _mm_permute_pd(min, 1));
  return _mm_cvtsd_f64(min);
}

double horizontal_max(simd_type v) {
  __m128d low = _mm256_castpd256_pd128(v.data);
  __m128d high = _mm256_extractf128_pd(v.data, 1);
  __m128d max = _mm_max_pd(low, high);
  max = _mm_max_pd(max, _mm_permute_pd(max, 1));
  return _mm_cvtsd_f64(max);
}

double horizontal_add(simd_type v) {
  __m128d vlow = _mm256_castpd256_pd128(v.data);
  __m128d vhigh = _mm256_extractf128_pd(v.data, 1);
  __m128d total = _mm_add_pd(vlow, vhigh);
  total = _mm_hadd_pd(total, total);
  return _mm_cvtsd_f64(total);
}

double horizontal_product(simd_type v) {
  __m128d low = _mm256_castpd256_pd128(v.data);
  __m128d high = _mm256_extractf128_pd(v.data, 1);
  __m128d prod = _mm_mul_pd(low, high);
  prod = _mm_mul_pd(prod, _mm_permute_pd(prod, 1));
  return _mm_cvtsd_f64(prod);
}

double max(simd_type v) {
  __m256d lo = _mm256_unpacklo_pd(v.data, v.data);
  __m256d hi = _mm256_unpackhi_pd(v.data, v.data);

  __m256d max_pairs = _mm256_max_pd(lo, hi);

  __m128d max_doubles = _mm256_extractf128_pd(max_pairs, 1);
  max_doubles = _mm_max_sd(max_doubles, _mm256_castpd256_pd128(max_pairs));

  return _mm_cvtsd_f64(max_doubles);
}

double min(simd_type v) {
  __m256d lo = _mm256_unpacklo_pd(v.data, v.data);
  __m256d hi = _mm256_unpackhi_pd(v.data, v.data);

  __m256d min_pairs = _mm256_min_pd(lo, hi);

  __m128d min_doubles = _mm256_extractf128_pd(min_pairs, 1);
  min_doubles = _mm_min_sd(min_doubles, _mm256_castpd256_pd128(min_pairs));

  return _mm_cvtsd_f64(min_doubles);
}

double sum(simd_type v) {
  __m256d lo = _mm256_unpacklo_pd(v.data, v.data);
  __m256d hi = _mm256_unpackhi_pd(v.data, v.data);

  __m256d add_pairs = _mm256_add_pd(lo, hi);

  __m128d add_doubles = _mm256_extractf128_pd(add_pairs, 1);
  add_doubles = _mm_add_sd(add_doubles, _mm256_castpd256_pd128(add_pairs));

  return _mm_cvtsd_f64(add_doubles);
}

double product(simd_type v) {
  __m256d lo = _mm256_unpacklo_pd(v.data, v.data);
  __m256d hi = _mm256_unpackhi_pd(v.data, v.data);

  __m256d pairs = _mm256_mul_pd(lo, hi);

  __m128d doubles = _mm256_extractf128_pd(pairs, 1);
  doubles = _mm_mul_sd(doubles, _mm256_castpd256_pd128(pairs));

  return _mm_cvtsd_f64(doubles);
}
} // namespace d4
} // namespace simd2
} // namespace aks

#endif // !AKS_SIMD2_D4_HPP