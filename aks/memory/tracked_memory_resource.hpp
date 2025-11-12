#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <new>

namespace aks {

class tracked_memory_resource : public std::pmr::memory_resource {
 public:
  mutable std::mutex         mtx;
  std::int32_t               total_allocation_calls   = 0;
  std::int32_t               total_deallocation_calls = 0;
  std::int32_t               total_allocated_bytes    = 0;
  std::int32_t               total_deallocated_bytes  = 0;
  std::int32_t               active_bytes             = 0;
  std::int32_t               max_active_bytes         = 0;
  std::pmr::memory_resource* upstream_;

  explicit tracked_memory_resource(
      std::pmr::memory_resource* upstream =
          std::pmr::get_default_resource()) noexcept
      : upstream_(upstream) {}

 private:
  void* do_allocate(std::size_t bytes, std::size_t align) override {
    {
      std::lock_guard<std::mutex> lck(mtx);
      total_allocation_calls += 1;
      total_allocated_bytes += static_cast<std::int32_t>(bytes);
      active_bytes += static_cast<std::int32_t>(bytes);
      max_active_bytes = std::max(active_bytes, max_active_bytes);
    }
    void* ptr = upstream_->allocate(bytes, align);
    return ptr;
  }

  void do_deallocate(void* ptr, std::size_t bytes, std::size_t align) override {
    {
      std::lock_guard<std::mutex> lck(mtx);
      total_deallocation_calls += 1;
      total_deallocated_bytes += static_cast<std::int32_t>(bytes);
      active_bytes -= static_cast<std::int32_t>(bytes);
    }
    upstream_->deallocate(ptr, bytes, align);
  }

  bool do_is_equal(
      std::pmr::memory_resource const& other) const noexcept override {
    if (this == &other) {
      return true;
    }

    auto const* other_tracked =
        dynamic_cast<tracked_memory_resource const*>(&other);
    if (!static_cast<bool>(other_tracked)) {
      return false;
    }

    return upstream_->is_equal(*other_tracked->upstream_);
  }
};

}  // namespace aks

#include <ostream>

namespace aks {
std::ostream& operator<<(std::ostream& o, tracked_memory_resource const& mr) {
  o << "tracked_memory_resource:\n";
  o << "\ttotal_allocation_calls  : " << mr.total_allocation_calls << "\n";
  o << "\ttotal_deallocation_calls: " << mr.total_deallocation_calls << "\n";
  o << "\ttotal_allocated_bytes   : " << mr.total_allocated_bytes << "\n";
  o << "\ttotal_deallocated_bytes : " << mr.total_deallocated_bytes << "\n";
  o << "\tactive_bytes            : " << mr.active_bytes << "\n";
  o << "\tmax_active_bytes        : " << mr.max_active_bytes << "\n";
  return o;
}
}  // namespace aks
