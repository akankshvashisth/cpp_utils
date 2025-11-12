#ifndef __aks_thread_thread_safe_queue_hpp__
#define __aks_thread_thread_safe_queue_hpp__

#include <mutex>
#include <queue>
#include <optional>

namespace aks
{
	namespace thread
	{
		

		template<typename T>
		struct queue_traits {
			using value_type = typename T::value_type;
			using reference = typename T::reference;
			using const_reference = typename T::const_reference;
			using size_type = typename T::size_type;
		};

		template<typename _QueueType, typename _QueueTraits = queue_traits<_QueueType>>
		struct thread_safe_queue
		{
			using queue_type = _QueueType;
			using queue_traits = _QueueTraits;
			using value_type = typename queue_traits::value_type;
			using reference = typename queue_traits::reference;
			using size_type = typename queue_traits::size_type;

			thread_safe_queue(queue_type const& q) :_queue(q) {}
			thread_safe_queue() :_queue() {}

			void enqueue(const value_type& value) {
				lock_type _(_mtx);
				_queue.push(value);
			}
			void enqueue(value_type&& value) {
				lock_type _(_mtx);
				_queue.push(std::move(value));
			}
			template< class... Args >
			decltype(auto) enqueue_emplace(Args&&... args)
			{
				lock_type _(_mtx);
				_queue.emplace(std::forward<Args>(args)...);
			}			
			std::optional<value_type> dequeue() {
				lock_type _(_mtx);
				if (_queue.empty()) {
					return std::optional<value_type>();
				}
				std::optional<value_type> ret(std::move(_queue.front()));
				_queue.pop();
				return ret;
			}
			bool empty() const {
				lock_type _(_mtx);
				return _queue.empty();
			}
			size_type size() const {
				lock_type _(_mtx);
				return _queue.size();
			}
			void clear() {
				lock_type _(_mtx);
				_queue = queue_type();
			}

		private:
			using lock_type = std::lock_guard<std::mutex>;

			queue_type _queue;
			mutable std::mutex _mtx;

		};
	}
}


#endif __aks_thread_thread_safe_queue_hpp__

