#pragma once

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace aks {
struct high_resolution_timer {
  using clock = std::chrono::steady_clock;
  using time_point = std::chrono::time_point<clock>;

  high_resolution_timer() : start_(clock::now()) {}

  void restart() { start_ = clock::now(); }

  std::chrono::nanoseconds elapsed_nanoseconds() const {
    return clock::now() - start_;
  }

  std::int64_t elapsed_nanoseconds_count() const {
    return elapsed_nanoseconds().count();
  }

  std::chrono::microseconds elapsed_microseconds() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() -
                                                                 start_);
  }

  std::int64_t elapsed_microseconds_count() const {
    return elapsed_microseconds().count();
  }

  double elapsed_seconds_count() const {
    return elapsed_microseconds_count() / 1e-6;
  }

  time_point start_;
};

struct high_resolution_timer_with_history {
  void full_restart() {
    timer_.restart();
    history_.clear();
  }

  void push() { history_.push_back(timer_.elapsed_microseconds()); }
  void push_and_reset() {
    push();
    timer_.restart();
  }
  void reset() { timer_.restart(); }

  auto time_expression(auto lambda, bool push_after = true) {
    reset();
    auto result = lambda();
    if (push_after)
      push();
    return result;
  }

  auto time_void_expression(auto lambda, bool push_after = true) {
    reset();
    lambda();
    if (push_after)
      push();
  }

  high_resolution_timer timer_;
  std::vector<std::chrono::microseconds> history_;
};

struct high_resolution_timer_manager {
  struct timer_context {
    explicit timer_context(high_resolution_timer_with_history &timer)
        : timer_(timer) {
      timer_.reset();
    }
    ~timer_context() { timer_.push(); }
    high_resolution_timer_with_history &timer_;
  };

  high_resolution_timer_with_history &get_timer(std::string const &name) {
    auto it = timers_.find(name);
    if (it == timers_.end()) {
      auto [new_it, inserted] =
          timers_.emplace(name, high_resolution_timer_with_history{});
      order_.push_back(name);
      return new_it->second;
    }
    return it->second;
  }

  timer_context get_context(std::string const &name) {
    return timer_context(get_timer(name));
  }

  void clear() {
    timers_.clear();
    order_.clear();
  }

  void restart_all() {
    for (auto &timer : timers_) {
      timer.second.full_restart();
    }
  }

  void dump_to_stream(std::ostream &os) const {
    if (timers_.empty()) {
      os << "(no timers)\n";
      return;
    }

    // find the widest timer name
    std::size_t max_name = 0;
    for (auto const &n : order_) {
      max_name = std::max(max_name, n.size());
    }
    max_name =
        std::max<std::size_t>(max_name, 4); // at least wide enough for "Name"

    // header
    os << std::format("{:<{}} {:>8} {:>12} {:>12} {:>12} {:>12}\n", "Name",
                      max_name, "Count", "Total(us)", "Avg(us)", "Min(us)",
                      "Max(us)");

    // separator line
    std::size_t total_width =
        max_name + 2 + 8 + 2 + 12 + 2 + 12 + 2 + 12 + 2 + 12;
    os << std::string(total_width, '-') << '\n';

    // rows (preserve insertion order)
    for (auto const &name : order_) {
      auto const &history = timers_.at(name).history_;
      std::size_t count = history.size();

      long long total_us = 0;
      long long min_us = 0;
      long long max_us = 0;
      if (count > 0) {
        auto sum = std::accumulate(history.begin(), history.end(),
                                   std::chrono::microseconds(0),
                                   [](auto a, auto b) { return a + b; });
        total_us = sum.count();
        min_us = std::min_element(history.begin(), history.end())->count();
        max_us = std::max_element(history.begin(), history.end())->count();
      }

      double avg_us =
          count ? static_cast<double>(total_us) / static_cast<double>(count)
                : 0.0;

      os << std::format("{:<{}} {:>8} {:>12} {:>12.3f} {:>12} {:>12}\n", name,
                        max_name, count, total_us, avg_us, min_us, max_us);
    }
  }

  void dump_to_stream_as_csv(std::ostream &os) const {
    // header
    os << "Name,Count,Total(us),Avg(us),Min(us),Max(us)\n";

    // rows (preserve insertion order)
    for (auto const &name : order_) {
      auto const &history = timers_.at(name).history_;
      std::size_t count = history.size();

      long long total_us = 0;
      long long min_us = 0;
      long long max_us = 0;
      if (count > 0) {
        auto sum = std::accumulate(history.begin(), history.end(),
                                   std::chrono::microseconds(0),
                                   [](auto a, auto b) { return a + b; });
        total_us = sum.count();
        min_us = std::min_element(history.begin(), history.end())->count();
        max_us = std::max_element(history.begin(), history.end())->count();
      }

      double avg_us =
          count ? static_cast<double>(total_us) / static_cast<double>(count)
                : 0.0;

      os << std::format("{},{},{},{},{},{}\n", name, count, total_us, avg_us,
                        min_us, max_us);
    }
  }

  std::unordered_map<std::string, high_resolution_timer_with_history> timers_;
  std::vector<std::string> order_;
};

std::string timestamp(auto const &now) {
  return std::format("{:%F_%H_%M_%S}",
                     std::chrono::time_point_cast<std::chrono::seconds>(now));
}

} // namespace aks
