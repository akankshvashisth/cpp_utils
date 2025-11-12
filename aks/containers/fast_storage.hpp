
#ifndef __aks_memory_fast_storage_hpp__
#define __aks_memory_fast_storage_hpp__

#include <limits>

namespace aks
{
	namespace memory
	{
		template<typename _T>
		class fast_storage {
		public:
			using value_type = _T;
			constexpr static size_t const block_size = sizeof(value_type);

		private:
			static_assert(sizeof(value_type) >= sizeof(size_t), "sizeof(value_type) >= sizeof(size_t)");

			size_t total_blocks;
			size_t num_initialized;
			size_t num_free;
			size_t next_free;
			value_type* start;

			size_t idx_from_addr(value_type* p) const {
				return (p - start);// / block_size;
			}

			value_type* addr_from_idx(size_t const idx) const {
				return &start[idx];
			}

			void set_value_at_idx(size_t const idx, size_t const value) {
				*((size_t*)addr_from_idx(idx)) = value;
			}

			void allocate_storage(size_t const num_blocks)
			{
				start = (value_type*)malloc(num_blocks * block_size); // aligned_alloc(num_blocks * block_size, alignof(value_type));
				next_free = 0;
				set_value_at_idx(0, 1);
				total_blocks = num_blocks;
				num_initialized = 0;
				num_free = total_blocks;
			}

			void resize_storage(size_t const num_of_blocks) {
				if (num_of_blocks <= total_blocks) return;

				auto const extra_storage_count = num_of_blocks - total_blocks;
				value_type* new_mem_start = (value_type*)realloc(start, num_of_blocks * block_size); //align on realloc?
				total_blocks = num_of_blocks;
				start = new_mem_start;
				num_free = num_free + extra_storage_count;
			}

			void deallocate_storage() {
				if (start) {
					free(start);
					start = nullptr;
					next_free = 0;
					total_blocks = 0;
					num_free = 0;
					num_initialized = 0;
				}
			}

		public:
			fast_storage(size_t const num_blocks) {
				allocate_storage(num_blocks);
			}

			~fast_storage() {
				deallocate_storage();
			}

			size_t allocate() {
				if (num_free == 0) {
					resize_storage((size_t)(total_blocks * 1.5));
				}

				if (num_initialized < total_blocks) {
					set_value_at_idx(num_initialized, num_initialized + 1);
					++num_initialized;
				}

				if (num_free > 0) {
					size_t ret = next_free;
					--num_free;
					next_free = num_free == 0 ? total_blocks : *(size_t*)addr_from_idx(next_free);
					return ret;
				}

				return std::numeric_limits<size_t>::max();
			}

			size_t allocate(value_type const v) {
				auto ret = allocate();
				*(at(ret)) = v;
				return ret;
			}

			void deallocate(size_t p) {
				set_value_at_idx(p, next_free);
				next_free = p;
				++num_free;
			}

			void deallocate(value_type* p) {
				deallocate(idx_from_addr(p));
			}

			value_type* at(size_t idx) {
				return addr_from_idx(idx);
			}

			value_type* allocate_and_get() {
				return at(allocate());
			}


		};
	}
}



#endif // !__aks_memory_fast_storage_hpp__

