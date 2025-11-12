#pragma once

#include <memory_resource>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace aks {

template <typename T, size_t UniqueId, size_t MaxBlocksPerChunk = 0,
          size_t LargestRequiredPoolBlock = 0>
struct registry_id {
  constexpr static std::size_t unique_id = UniqueId;
  constexpr static std::size_t max_blocks_per_chunk = MaxBlocksPerChunk;
  constexpr static std::size_t largest_required_pool_block =
      LargestRequiredPoolBlock;

  std::uint64_t id_;

  auto operator<=>(registry_id const &rhs) const = default;
};

template <typename T, size_t UniqueId, size_t MaxBlocksPerChunk = 0,
          size_t LargestRequiredPoolBlock = 0>
struct id_registry {
  constexpr static std::size_t unique_id = UniqueId;
  constexpr static std::size_t max_blocks_per_chunk = MaxBlocksPerChunk;
  constexpr static std::size_t largest_required_pool_block =
      LargestRequiredPoolBlock;
  using value_type = T;
  using registry_id_t = registry_id<value_type, unique_id, max_blocks_per_chunk,
                                    largest_required_pool_block>;

  // static id_registry& instance() {
  //   static id_registry registry;
  //   return registry;
  // }

  auto register_value(auto &&value) {
    std::hash<std::remove_cvref_t<decltype(value)>> hasher;
    std::uint64_t hash = hasher(value);
    registry_id_t id{hash};
    if (registry_.contains(hash)) {
      return id;
    }
    registry_.emplace(hash, std::move(value));
    return id;
  }

  std::optional<std::reference_wrapper<const value_type>>
  get_value(registry_id_t hash) const {
    auto iter = registry_.find(hash.id_);
    if (iter == registry_.end()) {
      return std::nullopt;
    }
    return std::ref(iter->second);
  }

  auto &pool() { return pool_; }
  auto const &pool() const { return pool_; }

  id_registry()
      : pool_(std::pmr::pool_options{max_blocks_per_chunk,
                                     largest_required_pool_block}),
        registry_(&pool_) {}

private:
  std::pmr::synchronized_pool_resource pool_;
  std::pmr::unordered_map<std::uint64_t, value_type> registry_;
};

template <size_t UniqueId, size_t MaxBlocksPerChunk,
          size_t LargestRequiredPoolBlock>
struct id_registry<std::pmr::string, UniqueId, MaxBlocksPerChunk,
                   LargestRequiredPoolBlock> {
  constexpr static std::size_t unique_id = UniqueId;
  constexpr static std::size_t max_blocks_per_chunk = MaxBlocksPerChunk;
  constexpr static std::size_t largest_required_pool_block =
      LargestRequiredPoolBlock;
  using value_type = std::pmr::string;
  using registry_id_t = registry_id<value_type, unique_id, max_blocks_per_chunk,
                                    largest_required_pool_block>;

  // static id_registry &instance() {
  //   static id_registry registry;
  //   return registry;
  // }

  auto register_value(std::string_view value) {
    std::hash<std::string_view> hasher;
    std::uint64_t hash = hasher(value);
    registry_id_t id{hash};
    if (registry_.contains(hash)) {
      return id;
    }
    registry_.emplace(hash, value_type(value, &pool_));
    return id;
  }

  std::optional<std::reference_wrapper<const value_type>>
  get_value(registry_id_t hash) const {
    auto iter = registry_.find(hash.id_);
    if (iter == registry_.end()) {
      return std::nullopt;
    }
    return std::ref(iter->second);
  }

  auto &pool() { return pool_; }
  auto const &pool() const { return pool_; }

  id_registry()
      : pool_(std::pmr::pool_options{max_blocks_per_chunk,
                                     largest_required_pool_block}),
        registry_(&pool_) {}

private:
  std::pmr::synchronized_pool_resource pool_;
  std::pmr::unordered_map<std::uint64_t, value_type> registry_;
};

} // namespace aks
