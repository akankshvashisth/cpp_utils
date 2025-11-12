#ifndef __aks_cpputils_worker_thread_hpp__
#define __aks_cpputils_worker_thread_hpp__

#include <thread>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <atomic>
#include <array>
#include <memory>

#include "thread_safe_queue.hpp"



namespace aks
{
	namespace thread
	{
		template<typename T, typename U>
		class worker_thread_interface
		{
		public:
			using work_type = T;
			using result_type = U;

			virtual void add_work(work_type&& t) = 0;

			virtual std::optional<result_type> get_result() = 0;

			virtual bool has_result() const = 0;

			virtual ~worker_thread_interface() {}
		};

		template<typename T, typename U, typename F, size_t N>
		class worker_thread : public worker_thread_interface<T,U>
		{
		public:
			using work_type = T;
			using result_type = U;
			using work_function_type = F;
			static size_t const worker_count = N;

			worker_thread(work_function_type f)
				: _f(f)
				, _work()
				, _result()
				, _do_exit(false)
				, _worker_cv()
				, _worker_mtx()
				, _workers() 
			{
				start();
			}

			void add_work(work_type&& t) {
				_work.enqueue(std::forward<work_type>(t));
				_worker_cv.notify_all();
			}
			
			std::optional<result_type> get_result() {
				return _result.dequeue();
			}
			
			bool has_result() const { return !_result.empty(); }

			~worker_thread() {
				stop();
			}

		private:		
			void start() {
				for (auto& worker : _workers) {
					if (!worker.joinable())
						worker = std::thread(&worker_thread::do_work, this);
				}
			}

			void stop() {
				_do_exit.store(true);
				_worker_cv.notify_all();
				for(auto& worker:_workers)
					worker.join();
			}

			void do_work() {
				while (true) {
					std::unique_lock<std::mutex> lck(_worker_mtx);
					_worker_cv.wait(lck, [&]() {return !_work.empty() || _do_exit.load(); });
					if (_do_exit.load()) {
						break;
					}
					std::optional<work_type> work = _work.dequeue();
					lck.unlock();
					if (work.has_value()) {
						_result.enqueue(std::move(_f(std::move(work.value()))));
					}
				}
			}

			work_function_type						_f;
			thread_safe_queue<std::queue<work_type>>			_work;
			thread_safe_queue<std::queue<U>>					_result;
			std::atomic<bool>						_do_exit;
			std::condition_variable					_worker_cv;
			mutable std::mutex								_worker_mtx;
			std::array<std::thread, worker_count>	_workers;
		};

		template<typename T, typename U, size_t N, typename F>
		std::unique_ptr<worker_thread_interface<T, U>> make_worker(F f) {
			return std::unique_ptr<worker_thread_interface<T, U>>(new worker_thread<T, U, F, N>(f));
		}
	}
}


#endif // __aks_cpputils_worker_thread_hpp__

