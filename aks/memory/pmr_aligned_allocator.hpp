#include <cassert>
#include <cstddef>
#include <memory>
#include <memory_resource>
#include <new>

namespace aks {
class aligned_memory_resource : public std::pmr::memory_resource {
 private:
  std::size_t                alignment_;
  std::pmr::memory_resource* upstream_;

  // Helper function to calculate aligned size
  std::size_t align_size(std::size_t size) const noexcept {
    return (size + alignment_ - 1) & ~(alignment_ - 1);
  }

 public:
  explicit aligned_memory_resource(
      std::size_t                alignment = alignof(std::max_align_t),
      std::pmr::memory_resource* upstream =
          std::pmr::get_default_resource()) noexcept
      : alignment_(alignment), upstream_(upstream) {
    // Ensure alignment is a power of 2
    assert((alignment & (alignment - 1)) == 0);
  }

 private:
  void* do_allocate(std::size_t bytes, std::size_t align) override {
    // Use the larger of the requested alignment and our minimum alignment
    std::size_t actual_alignment = std::max(alignment_, align);

    // Calculate total size needed including alignment padding
    std::size_t aligned_size = align_size(bytes);

    // Allocate memory from upstream resource
    void* ptr = upstream_->allocate(aligned_size, actual_alignment);

    // Ensure the returned pointer meets alignment requirements
    assert(reinterpret_cast<std::uintptr_t>(ptr) % actual_alignment == 0);

    return ptr;
  }

  void do_deallocate(void* ptr, std::size_t bytes, std::size_t align) override {
    std::size_t actual_alignment = std::max(alignment_, align);
    std::size_t aligned_size     = align_size(bytes);
    upstream_->deallocate(ptr, aligned_size, actual_alignment);
  }

  bool do_is_equal(
      std::pmr::memory_resource const& other) const noexcept override {
    if (this == &other)
      return true;

    auto* other_aligned = dynamic_cast<aligned_memory_resource const*>(&other);
    if (!other_aligned)
      return false;

    return alignment_ == other_aligned->alignment_ &&
           upstream_->is_equal(*other_aligned->upstream_);
  }
};

//// Helper template for creating aligned allocators
// template <typename T>
// class pmr_aligned_allocator {
//  private:
//   std::pmr::polymorphic_allocator<T> alloc_;
//
//  public:
//   using value_type = T;
//
//   explicit aligned_allocator(std::size_t alignment)
//       : alloc_(std::pmr::polymorphic_allocator<T>(
//             new aligned_memory_resource(alignment))) {}
//
//   template <typename U>
//   aligned_allocator(aligned_allocator<U> const& other) noexcept
//       : alloc_(other.alloc_) {}
//
//   T* allocate(std::size_t n) { return alloc_.allocate(n); }
//
//   void deallocate(T* p, std::size_t n) { alloc_.deallocate(p, n); }
//
//   template <typename U>
//   bool operator==(aligned_allocator<U> const& other) const noexcept {
//     return alloc_ == other.alloc_;
//   }
//
//   template <typename U>
//   bool operator!=(aligned_allocator<U> const& other) const noexcept {
//     return !(*this == other);
//   }
// };
}  // namespace aks
