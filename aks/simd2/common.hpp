#ifndef AKS_SIMD2_COMMON_HPP
#define AKS_SIMD2_COMMON_HPP

namespace aks {
namespace simd2 {
template <typename T, size_t N> struct alignas(32) type_wrapper {
  using type = T;
  type data;
};
} // namespace simd2
} // namespace aks

#endif // AKS_SIMD2_COMMON_HPP