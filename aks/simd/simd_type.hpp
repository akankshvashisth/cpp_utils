#ifndef SIMD_TYPE_HPP
#define SIMD_TYPE_HPP

#include <immintrin.h>
#include <smmintrin.h>

#ifdef __clang_analyzer__
#define AKS_DO_NOT_USE_SVML_INTRINSICS
#endif

#ifndef _WIN32
#define AKS_DO_NOT_USE_SVML_INTRINSICS
#endif

namespace aks {

template <typename X_, std::size_t N>
struct simd_type;

template <>
struct simd_type<float, 16> {
  using type                              = __m512;
  using mask_type                         = __mmask16;
  constexpr static std::size_t const size = 16;
};

template <>
struct simd_type<double, 8> {
  using type                              = __m512d;
  using mask_type                         = __mmask8;
  constexpr static std::size_t const size = 8;
};

template <>
struct simd_type<int, 16> {
  using type                              = __m512i;
  using mask_type                         = __mmask16;
  constexpr static std::size_t const size = 16;
};

template <>
struct simd_type<float, 8> {
  using type                              = __m256;
  constexpr static std::size_t const size = 8;
};

template <>
struct simd_type<float, 4> {
  using type                              = __m128;
  constexpr static std::size_t const size = 4;
};

template <>
struct simd_type<double, 4> {
  using type                              = __m256d;
  constexpr static std::size_t const size = 4;
};

template <>
struct simd_type<double, 2> {
  using type                              = __m128d;
  constexpr static std::size_t const size = 2;
};

template <>
struct simd_type<int, 8> {
  using type                              = __m256i;
  constexpr static std::size_t const size = 8;
};

template <>
struct simd_type<int, 4> {
  using type                              = __m128i;
  constexpr static std::size_t const size = 4;
};

template <>
struct simd_type<long, 4> {
  using type                              = __m256i;
  constexpr static std::size_t const size = 4;
};
}  // namespace aks

#endif  // SIMD_TYPE_HPP
