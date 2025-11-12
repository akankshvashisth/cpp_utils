#include <limits>

namespace aks {
namespace memory {

template <typename _T> class fixed_size_memory_pool {
public:
  typedef _T value_type;

private:
  size_t _num_blocks;
  constexpr auto _size_of_each_block() const { return sizeof(value_type); }
  size_t _num_free_blocks;
  size_t _num_initialized;
  value_type *m_mem_start;
  value_type *m_next;

  void allocate_stage(size_t const num_of_blocks) {
    m_mem_start = aligned_alloc(num_of_blocks * _size_of_each_block(),
                                alignof(value_type));
    m_next = m_mem_start;
    _num_blocks = num_of_blocks;
    _num_free_blocks = num_of_blocks;
    _num_initialized = 0;
  }

  void resize_stage(size_t const num_of_blocks) {
    if (num_of_blocks <= _num_blocks)
      return;

    auto next_offset = m_next ? m_next - m_mem_start : _num_blocks;
    auto extra_storage_count = num_of_blocks - _num_blocks;
    value_type *new_mem_start =
        realloc(num_of_blocks * _size_of_each_block()); // align on realloc?
    m_mem_start = new_mem_start;
    m_next = m_mem_start + next_offset;
    _num_blocks = num_of_blocks;
    _num_free_blocks = _num_free_blocks + extra_storage_count;
    _num_initialized = _num_initialized;
  }

  void deallocate_stage() {
    if (m_mem_start) {
      free(m_mem_start);
      m_mem_start = nullptr;
      m_next = nullptr;
      _num_blocks = 0;
      _num_free_blocks = 0;
      _num_initialized = 0;
    }
  }

  constexpr value_type *addr_from_idx(size_t const idx) const {
    return m_mem_start + (idx * _size_of_each_block());
  }

  constexpr size_t idx_from_addr(value_type const *addr) {
    return ((size_t)(addr - m_mem_start)) / _size_of_each_block();
  }

public:
  fixed_size_memory_pool(size_t const num_of_blocks)
      : _num_blocks(0), _num_free_blocks(0), _num_initialized(0),
        m_mem_start(nullptr), m_next(nullptr) {
    allocate_stage(num_of_blocks);
  }

  ~fixed_size_memory_pool() { deallocate_stage(); }

  value_type *allocate() {
    if (_num_free_blocks == 0) {
      resize_stage(_num_blocks * 2);
      return allocate();
    }

    if (_num_initialized < _num_blocks) {
      size_t *p = addr_from_idx(_num_initialized);
      *p = _num_initialized + 1;
      ++_num_initialized;
    }

    void *ret = nullptr;
    ret = (void *)m_next;
    --_num_free_blocks;
    m_next =
        _num_free_blocks == 0 ? nullptr : addr_from_idx(*((size_t *)m_next));
    return ret;
  }

  void deallocate(value_type *p) {
    (*(size_t *)p) = (m_next != nullptr) ? idx_from_addr(m_next) : _num_blocks;
    m_next = (value_type *)p;
    ++_num_free_blocks;
  }
};
} // namespace memory
} // namespace aks
