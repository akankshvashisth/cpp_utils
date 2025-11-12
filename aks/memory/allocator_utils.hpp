#pragma once

#include <memory>
#include <utility>

// Document:	P0316R0
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0316r0.html

namespace aks {
template <class T, class Alloc>
class allocator_delete {
 public:
  using allocator_type = std::remove_cv_t<Alloc>;
  using pointer        = typename allocator_traits<allocator_type>::pointer;

  template <class OtherAlloc>
  allocator_delete(OtherAlloc&& other);

  void operator()(pointer p);

  allocator_type&       get_allocator();
  allocator_type const& get_allocator() const;

 private:
  allocator_type alloc;  // for exposition only
};

template <class T, class Alloc>
class allocator_delete<T, Alloc&> {
 public:
  using allocator_type = std::remove_cv_t<Alloc>;
  using pointer = typename std::allocator_traits<allocator_type>::pointer;

  allocator_delete(std::reference_wrapper<Alloc> alloc);

  void operator()(pointer p);

  Alloc& get_allocator() const;

 private:
  std::reference_wrapper<Alloc> alloc;  // for exposition only
};

template <class T, class OtherAlloc>
allocator_delete(OtherAlloc&& alloc) -> allocator_delete<
    T,
    typename std::allocator_traits<OtherAlloc>::template rebind_alloc<T>>;

template <class T, class Alloc>
allocator_delete(std::reference_wrapper<Alloc> alloc)
    -> allocator_delete<T, Alloc&>;

template <class T, class Alloc, class... Args>
std::unique_ptr<T,
                allocator_delete<T,
                                 typename std::allocator_traits<
                                     Alloc>::template rebind_alloc<T>>>
allocate_unique(Alloc&& alloc, Args&&... args);

template <class T, class Alloc, class... Args>
std::unique_ptr<T, allocator_delete<T, Alloc&>> allocate_unique(
    std::reference_wrapper<Alloc> alloc,
    Args&&... args);

template <class T, class Alloc, class... Args>
auto allocate_unique(Alloc const& alloc, Args&&... args) {
  using traits  = typename allocator_traits<Alloc>::template rebind_traits<T>;
  auto my_alloc = typename traits::allocator_type(alloc);
  auto hold_deleter = [&my_alloc](auto p) {
    traits::deallocate(my_alloc, p, 1);
  };
  using hold_t = unique_ptr<T, decltype(hold_deleter)>;
  auto hold    = hold_t(traits::allocate(my_alloc, 1), hold_deleter);
  traits::construct(my_alloc, hold.get(), forward<Args>(args)...);
  auto deleter = allocator_delete<T>(my_alloc);
  return std::unique_ptr<T, decltype(deleter)>{hold.release(), move(deleter)};
}
}  // namespace aks