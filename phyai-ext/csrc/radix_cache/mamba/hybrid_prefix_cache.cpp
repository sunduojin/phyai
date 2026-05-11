#include "radix_cache/mamba/hybrid_prefix_cache.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace phyai_ext::radix_cache {

hybrid_prefix_cache::hybrid_prefix_cache(std::shared_ptr<prefix_cache> kv_cache, std::int32_t num_mamba_slots)
    : kv_cache_(std::move(kv_cache)), mamba_alloc_(num_mamba_slots) {
  if (kv_cache_ == nullptr) { throw std::invalid_argument("hybrid_prefix_cache: null kv cache"); }
  observer_id_ = kv_cache_->add_evict_observer([this](tree_node* node, tier t) { this->on_kv_evict(node, t); });
}

hybrid_prefix_cache::~hybrid_prefix_cache() {
  if (kv_cache_ && observer_id_ != 0) { kv_cache_->remove_evict_observer(observer_id_); }
}

hybrid_prefix_cache::hybrid_match_result hybrid_prefix_cache::match(atom_span query) {
  hybrid_match_result out;
  out.kv = kv_cache_->match(query);
  out.last_mamba_node = find_last_mamba_node(out.kv.last_node[tier_index(tier::device)]);
  if (out.last_mamba_node) { out.mamba_branching_atoms = out.last_mamba_node->depth_in_atoms(); }
  return out;
}

std::optional<mamba_slot> hybrid_prefix_cache::allocate_mamba_slot() { return mamba_alloc_.allocate(); }

void hybrid_prefix_cache::attach_mamba(tree_node* node, mamba_slot slot) {
  if (node == nullptr) { throw std::invalid_argument("hybrid_prefix_cache::attach_mamba: null node"); }
  node->attach_mamba(std::move(slot));
  mamba_leaves_.insert(node);
}

std::optional<mamba_slot> hybrid_prefix_cache::detach_mamba(tree_node* node) {
  if (node == nullptr) return std::nullopt;
  auto out = node->detach_mamba();
  mamba_leaves_.erase(node);
  return out;
}

tree_node* hybrid_prefix_cache::find_last_mamba_node(tree_node* from) const {
  for (tree_node* n = from; n != nullptr; n = n->parent()) {
    if (n->has_mamba()) return n;
  }
  return nullptr;
}

bool hybrid_prefix_cache::ensure_mamba_capacity_by_evict(std::int32_t num_slots) {
  if (mamba_alloc_.available_slots() >= num_slots) return true;
  std::vector<tree_node*> cands(mamba_leaves_.begin(), mamba_leaves_.end());
  std::sort(cands.begin(), cands.end(),
            [](tree_node* a, tree_node* b) { return a->last_access_time() < b->last_access_time(); });
  for (tree_node* n : cands) {
    if (mamba_alloc_.available_slots() >= num_slots) return true;
    auto slot = detach_mamba(n);
    (void)slot;
  }
  return mamba_alloc_.available_slots() >= num_slots;
}

void hybrid_prefix_cache::on_kv_evict(tree_node* node, tier /*t*/) {
  // The KV resource on `node` was just evicted; if a mamba slot lived on the
  // same node it is now orphaned (no KV state to reach it from), so drop it.
  if (node != nullptr && node->has_mamba()) { detach_mamba(node); }
}

}  // namespace phyai_ext::radix_cache
