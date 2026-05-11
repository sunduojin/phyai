#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_set>

#include "radix_cache/atoms.h"
#include "radix_cache/mamba/mamba_allocator.h"
#include "radix_cache/mamba/mamba_slot.h"
#include "radix_cache/prefix_cache.h"
#include "radix_cache/tree/tree_node.h"

namespace phyai_ext::radix_cache {

// Hybrid Mamba + Attention cache. Composes a prefix_cache reference and a
// mamba_allocator on top of the *same* radix_tree. The mamba slot is a
// sibling field of the KV resource on each tree_node, evicted in lock-step
// with the KV resource via an observer hook.
class hybrid_prefix_cache {
 public:
  hybrid_prefix_cache(std::shared_ptr<prefix_cache> kv_cache, std::int32_t num_mamba_slots);

  hybrid_prefix_cache(const hybrid_prefix_cache&) = delete;
  hybrid_prefix_cache& operator=(const hybrid_prefix_cache&) = delete;

  ~hybrid_prefix_cache();

  struct hybrid_match_result {
    prefix_cache::match_result kv;
    tree_node* last_mamba_node = nullptr;  // nearest ancestor carrying a mamba slot
    std::uint32_t mamba_branching_atoms = 0;
  };
  hybrid_match_result match(atom_span query);

  // Slot operations.
  std::optional<mamba_slot> allocate_mamba_slot();
  void attach_mamba(tree_node* node, mamba_slot slot);
  std::optional<mamba_slot> detach_mamba(tree_node* node);
  tree_node* find_last_mamba_node(tree_node* from) const;

  // Free at least `num_slots` mamba slots by evicting LRU mamba leaves.
  // Returns true iff the post-condition is satisfied.
  bool ensure_mamba_capacity_by_evict(std::int32_t num_slots);

  std::int32_t available_slots() const noexcept { return mamba_alloc_.available_slots(); }
  std::int32_t total_slots() const noexcept { return mamba_alloc_.total_slots(); }
  std::int32_t active_slots() const noexcept { return mamba_alloc_.active_slots(); }

  std::shared_ptr<prefix_cache> kv() const noexcept { return kv_cache_; }

 private:
  void on_kv_evict(tree_node* node, tier t);

  std::shared_ptr<prefix_cache> kv_cache_;
  mamba_allocator mamba_alloc_;
  std::unordered_set<tree_node*> mamba_leaves_;
  std::size_t observer_id_ = 0;
};

}  // namespace phyai_ext::radix_cache
