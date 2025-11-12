#pragma once

#include <chrono>
#include <format>
#include <fstream>
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
    explicit timer_context(high_resolution_timer_with_history& timer)
        : timer_(timer) {
      timer_.reset();
    }
    ~timer_context() { timer_.push(); }
    high_resolution_timer_with_history& timer_;
  };

  high_resolution_timer_with_history& get_timer(std::string const& name) {
    return timers_[name];
  }

  timer_context get_context(std::string const& name) {
    return timer_context(get_timer(name));
  }

  void clear() { timers_.clear(); }

  void restart_all() {
    for (auto& timer : timers_) {
      timer.second.full_restart();
    }
  }

  void dump_to_file(std::string const& filename) const {
    std::ofstream file(filename);
    file << "timer,microseconds" << std::endl;
    for (auto const& timer : timers_) {
      for (auto const& duration : timer.second.history_) {
        file << timer.first << "," << duration.count() << std::endl;
      }
    }
  }

  std::unordered_map<std::string, high_resolution_timer_with_history> timers_;
};

std::string timestamp(auto const& now) {
  return std::format("{:%F_%H_%M_%S}",
                     std::chrono::time_point_cast<std::chrono::seconds>(now));
}

}  // namespace aks
