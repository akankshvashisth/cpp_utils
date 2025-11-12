#pragma once

// references:
// rigtorp
// cppcon2023/Fifo5b.hpp

#include <array>
#include <atomic>

namespace aks {
template <typename T, std::size_t Capacity> class lockfree_spsc_queue {
  static_assert(Capacity > 0 && ((Capacity & (Capacity - 1)) == 0),
                "Capacity must be a power of 2");

public:
  lockfree_spsc_queue() : head_(0), tail_(0) {}

  bool enqueue(const T &item) {
    std::size_t current_tail = tail_.load(std::memory_order_relaxed);
    std::size_t next_tail = increment(current_tail);
    if (next_tail != head_.load(std::memory_order_acquire)) {
      buffer_[current_tail & mask] = item;
      tail_.store(next_tail, std::memory_order_release);
      return true;
    }
    return false; // Queue is full
  }

  bool enqueue(T &&item) {
    std::size_t current_tail = tail_.load(std::memory_order_relaxed);
    std::size_t next_tail = increment(current_tail);
    if (next_tail != head_.load(std::memory_order_acquire)) {
      buffer_[current_tail & mask] = std::move(item);
      tail_.store(next_tail, std::memory_order_release);
      return true;
    }
    return false; // Queue is full
  }

  std::optional<T> dequeue() {
    std::size_t current_head = head_.load(std::memory_order_relaxed);
    if (current_head == tail_.load(std::memory_order_acquire)) {
      return std::nullopt; // Queue is empty
    }
    T item = std::move(buffer_[current_head & mask]);
    head_.store(increment(current_head), std::memory_order_release);
    return item;
  }

  bool empty() const {
    return head_.load(std::memory_order_acquire) ==
           tail_.load(std::memory_order_acquire);
  }

  bool full() const {
    std::size_t next_tail = increment(tail_.load(std::memory_order_acquire));
    return next_tail == head_.load(std::memory_order_acquire);
  }

  std::size_t size() const {
    return (tail_.load(std::memory_order_acquire) -
            head_.load(std::memory_order_acquire)) &
           mask;
  }

  std::size_t capacity() const { return Capacity; }

private:
  static constexpr std::size_t mask = Capacity - 1;

  std::size_t increment(std::size_t i) const { return (i + 1) & mask; }

  alignas(64) std::atomic<std::size_t> head_;
  alignas(64) std::atomic<std::size_t> tail_;
  std::array<T, Capacity> buffer_;
};
} // namespace aks