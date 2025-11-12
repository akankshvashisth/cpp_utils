#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace aks {
class thread_pool {
 public:
  explicit thread_pool(
      std::size_t num_threads = std::thread::hardware_concurrency()) {
    _threads.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
      add_worker();
    }
  }

  std::size_t size() const { return _threads.size(); }

  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args) {
    using return_type = std::invoke_result_t<F, Args...>;
    auto task         = std::packaged_task<return_type()>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> future = task.get_future();

    {
      std::unique_lock lock(_queue_mtx);
      _queue.emplace([task = std::move(task)]() mutable { std::move(task)(); });
    }
    _cv.notify_one();
    return future;
  }

  void hard_shutdown() {
    _shutdown = true;
    _cv.notify_all();
    for (auto& t : _threads) {
      t.detach();
    }
  }

  ~thread_pool() = default;

 private:
  using f_type = std::move_only_function<void()>;
  mutable std::mutex _queue_mtx;
  std::queue<f_type>
      _queue;  // make sure the queue is the first declared, as that needs to be
               // destroyed last, else cleanup does not happen

  std::condition_variable_any _cv;
  std::vector<std::jthread>   _threads;
  std::atomic_bool            _shutdown{false};

  void add_worker() { _threads.emplace_back(worker()); }

  std::jthread worker() {
    return std::jthread{[this](std::stop_token stop) {
      while (!_shutdown) {
        f_type f;

        {
          std::unique_lock<std::mutex> lock(_queue_mtx);
          _cv.wait(lock, stop,
                   [this]() { return !_queue.empty() || _shutdown; });

          if (_shutdown)
            break;

          if (_queue.empty()) {
            if (stop.stop_requested()) {
              break;
            } else {
              continue;
            }
          }

          f = std::move(_queue.front());
          _queue.pop();
        }

        f();
      }
    }};
  }
};
}  // namespace aks
#endif  // THREAD_POOL_HPP
