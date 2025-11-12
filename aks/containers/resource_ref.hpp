
#ifndef __aks_memory_resource_ref_hpp__
#define __aks_memory_resource_ref_hpp__
//////////////////////////////////////////////////////////////////////////
#include <assert.h>
#include <memory>

//////////////////////////////////////////////////////////////////////////
namespace aks
{
	namespace memory
	{

		template<typename _ResourceType, typename _Allocator, typename _Deleter>
		class fast_resource_ref // not thread safe
		{
		public:
			using resource_type = _ResourceType;
			using allocator_type = _Allocator;
			using deleter_type = _Deleter;
		protected:			
			template<typename... Args>
			fast_resource_ref(allocator_type a, deleter_type d, Args... args):_deleter(d), _handle(nullptr), _i_set_it(false) {
				set(a, args...);
			}

			~fast_resource_ref() {
				clear();
			}

			fast_resource_ref(fast_resource_ref const&) = delete;
			fast_resource_ref& operator=(fast_resource_ref const&) = delete;
		public:			
			static bool is_in_use() { return get_global_handle(); }

			resource_type* operator->() const
			{
				assert(_handle);
				return _handle;
			}

			resource_type& operator*() const
			{
				assert(_handle);
				return *_handle;
			}

			resource_type* get() const
			{
				assert(_handle);
				return _handle;
			}

			resource_type* get_raw() const
			{				
				return _handle;
			}

		private:
			deleter_type _deleter;
			resource_type* _handle;
			bool _i_set_it;

			template<typename... Args>
			void set(allocator_type a, Args... args) {				
				if (!get_global_handle()) {
					get_global_handle() = a(args...);
					_i_set_it = true;
				}
				_handle = get_global_handle();
			}

			void clear() {
				if (get_global_handle() && _i_set_it) {
					_deleter(get_global_handle());
					get_global_handle() = nullptr;
					_i_set_it = false;
				}
			}

			static resource_type*& get_global_handle()
			{
				static resource_type* s_mGlobalHandle = nullptr;
				return s_mGlobalHandle;
			}
		};

		// A template class that wraps shared access to a single resource
		//
		// The purpose of this class is to avoid singletons while preserving
		// unique resource instantiation.  One advantage over singletons is that it is
		// easy to override the creation of the unique resource by subclassing.

		template <typename _ResourceTypeT>
		class resource_ref
		{
		public:
			typedef _ResourceTypeT resource_type;

		protected:

			template <class AllocatorT>
			resource_ref(AllocatorT A) // TODO: not thread-safe
			{
				this->mxpHandle = get_global_handle().lock();
				if (!this->mxpHandle)
				{
					this->reset(A);
				}
			}

			template <class AllocatorT, class DeletorT, typename... AllocArgs>
			resource_ref(AllocatorT A, DeletorT D, AllocArgs... allocargs) // TODO: not thread-safe
			{
				this->mxpHandle = get_global_handle().lock();
				if (!this->mxpHandle)
				{
					this->reset(A, D, allocargs...);
				}
			}

			~resource_ref() { this->mxpHandle.reset(); }

		public:

			static unsigned int get_reference_count() { return get_global_handle().use_count(); }

			static bool IsInUse() { return (!get_global_handle().expired()); }

			resource_type* operator->() const
			{
				assert(mxpHandle);
				return mxpHandle.get();
			}

			resource_type& operator*() const
			{
				assert(mxpHandle);
				return *mxpHandle;
			}

			resource_type* get() const
			{
				assert(mxpHandle);
				return mxpHandle.get();
			}

		protected:

			template <class AllocatorT> void reset(AllocatorT A) // TODO: not thread-safe
			{
				this->mxpHandle.reset(A());
				get_global_handle() = this->mxpHandle;
			}

			template <class AllocatorT, class DeletorT, typename... AllocArgs> void reset(AllocatorT A, DeletorT D, AllocArgs... allocArgs) // TODO: not thread-safe
			{
				this->mxpHandle.reset(A(allocArgs...), D);
				get_global_handle() = this->mxpHandle;
			}

			//template <class AllocatorT, class DeletorT> void reset(AllocatorT A, DeletorT D) // TODO: not thread-safe
			//{
			//	this->mxpHandle.reset(A(), D);
			//	get_global_handle() = this->mxpHandle;
			//}

			static std::weak_ptr<resource_type>& get_global_handle()
			{
				static std::weak_ptr<resource_type> s_mGlobalHandle;
				return s_mGlobalHandle;
			}

			std::shared_ptr<resource_type> mxpHandle;
		};
	}
}
//////////////////////////////////////////////////////////////////////////
#endif // __aks_memory_resource_ref_hpp__


