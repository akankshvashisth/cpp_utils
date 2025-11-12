#pragma once

#include <chrono>
#include <format>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace aks {
struct high_resolution_timer {
  using clock = std::chrono::steady_clock;
  using time_point = std::chrono::time_point<clock>;
  using duration = std::chrono::duration<double, std::nano>;

  high_resolution_timer() : start_(clock::now()) {}

  void restart() { start_ = clock::now(); }

  std::chrono::nanoseconds elapsed_nanoseconds() const {
    return clock::now() - start_;
  }

  std::int64_t elapsed_nanoseconds_count() const {
    return elapsed_nanoseconds().count();
  }

  double elapsed_seconds() const { return elapsed_nanoseconds_count() / 1e-9; }

  time_point start_;
};

struct high_resolution_timer_with_history {
  void full_restart() {
    timer_.restart();
    history_.clear();
  }

  void push() { history_.push_back(timer_.elapsed_nanoseconds()); }
  void push_and_reset() {
    push();
    timer_.restart();
  }
  void reset() { timer_.restart(); }

  high_resolution_timer timer_;
  std::vector<std::chrono::nanoseconds> history_;
};

struct high_resolution_timer_manager {
  high_resolution_timer_with_history &get_timer(std::string const &name) {
    return timers_[name];
  }

  void clear() { timers_.clear(); }

  void restart_all() {
    for (auto &timer : timers_) {
      timer.second.full_restart();
    }
  }

  void dump_to_file(std::string const &filename) const {
    using duration_t = high_resolution_timer::duration;
    std::ofstream file(filename);
    file << "timer,nanoseconds" << std::endl;
    for (auto const &timer : timers_) {
      for (auto const &duration : timer.second.history_) {
        file << timer.first << "," << duration.count() << std::endl;
      }
    }
  }

  std::unordered_map<std::string, high_resolution_timer_with_history> timers_;
};

std::string timestamp(auto const &now) {
  return std::format("{:%F_%H_%M_%S}",
                     std::chrono::time_point_cast<std::chrono::seconds>(now));
}

} // namespace aks
