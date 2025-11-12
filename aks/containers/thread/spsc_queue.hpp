#include <atomic>
#include <concepts>
#include <cstddef> // for std::byte
#include <stdexcept>
#include <vector>

namespace aks {

template <typename T>
  requires std::is_nothrow_move_constructible_v<T> &&
           std::is_nothrow_destructible_v<T>
class spsc_queue {
public:
  explicit spsc_queue(std::size_t capacity)
      : capacity_(capacity), buffer_(capacity), head_(0), tail_(0) {
    if (!is_power_of_two(capacity_)) {
      throw std::invalid_argument("Capacity must be a power of two.");
    }
  }

  // Disable copy semantics
  spsc_queue(const spsc_queue &) = delete;
  spsc_queue &operator=(const spsc_queue &) = delete;

  // Enqueue method (single producer)
  bool enqueue(const T &item) noexcept {
    const std::size_t next_head =
        increment(head_.load(std::memory_order_relaxed));
    if (next_head == tail_.load(std::memory_order_acquire)) {
      return false; // Queue is full
    }
    buffer_[head_ & (capacity_ - 1)] = item;
    head_.store(next_head, std::memory_order_release);
    return true;
  }

  bool enqueue(T &&item) noexcept {
    const std::size_t next_head =
        increment(head_.load(std::memory_order_relaxed));
    if (next_head == tail_.load(std::memory_order_acquire)) {
      return false; // Queue is full
    }
    buffer_[head_ & (capacity_ - 1)] = std::move(item);
    head_.store(next_head, std::memory_order_release);
    return true;
  }

  // Dequeue method (single consumer)
  bool dequeue(T &item) noexcept {
    if (tail_.load(std::memory_order_acquire) ==
        head_.load(std::memory_order_acquire)) {
      return false; // Queue is empty
    }
    item = std::move(buffer_[tail_ & (capacity_ - 1)]);
    tail_.store(increment(tail_.load(std::memory_order_relaxed)),
                std::memory_order_release);
    return true;
  }

  // Check if the queue is empty
  bool empty() const noexcept {
    return tail_.load(std::memory_order_acquire) ==
           head_.load(std::memory_order_acquire);
  }

  // Check if the queue is full
  bool full() const noexcept {
    const std::size_t next_head =
        increment(head_.load(std::memory_order_relaxed));
    return next_head == tail_.load(std::memory_order_acquire);
  }

private:
  static constexpr std::size_t CACHE_LINE_SIZE = 64;

  // Utility function to check if a number is a power of two
  static constexpr bool is_power_of_two(std::size_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
  }

  const std::size_t capacity_;
  std::vector<T> buffer_;

  // Head, padded to avoid false sharing
  alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> head_;
  std::byte
      padding1_[CACHE_LINE_SIZE -
                sizeof(std::atomic<std::size_t>)]; // Ensure 'tail_' starts on a
                                                   // new cache line

  // Tail, padded to avoid false sharing
  alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> tail_;
  std::byte
      padding2_[CACHE_LINE_SIZE -
                sizeof(std::atomic<std::size_t>)]; // Padding to avoid cache
                                                   // line false sharing

  // Increment with bitwise AND to avoid modulus
  std::size_t increment(std::size_t idx) const noexcept { return idx + 1; }
};

} // namespace aks