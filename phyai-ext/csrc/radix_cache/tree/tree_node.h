#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>

#include "radix_cache/atoms.h"
#include "radix_cache/mamba/mamba_slot.h"
#include "radix_cache/tree/node_resource.h"
#include "phyai_ext/radix_cache/tier.h"

namespace phyai_ext::radix_cache {

class tree_node {
 public:
  using clock = std::chrono::steady_clock;
  using children_map = std::unordered_map<atom_vec, std::unique_ptr<tree_node>, atom_vec_hash, atom_vec_eq>;

  tree_node(tree_node* parent, atom_vec atoms_bytes, std::uint32_t atom_bytes, std::uint32_t atoms_per_unit,
            std::uint32_t depth_in_atoms);

  tree_node(const tree_node&) = delete;
  tree_node& operator=(const tree_node&) = delete;

  // Topology.
  tree_node* parent() noexcept { return parent_; }
  const tree_node* parent() const noexcept { return parent_; }
  void set_parent(tree_node* p) noexcept { parent_ = p; }
  children_map& children() noexcept { return children_; }
  const children_map& children() const noexcept { return children_; }

  // Atom payload of this node (i.e. the segment between parent and self).
  atom_span atoms() const noexcept { return atom_span{atoms_.data(), atoms_.size()}; }
  const atom_vec& atoms_vec() const noexcept { return atoms_; }
  std::uint32_t atom_count() const noexcept { return atom_count_; }
  std::uint32_t depth_in_atoms() const noexcept { return depth_in_atoms_; }
  std::uint32_t atom_bytes() const noexcept { return atom_bytes_; }
  std::uint32_t atoms_per_unit() const noexcept { return atoms_per_unit_; }

  // KV resource per tier.
  node_resource* resource(tier t) noexcept { return resources_[tier_index(t)].get(); }
  const node_resource* resource(tier t) const noexcept { return resources_[tier_index(t)].get(); }

  void attach_resource(tier t, std::unique_ptr<node_resource> r) { resources_[tier_index(t)] = std::move(r); }
  std::unique_ptr<node_resource> detach_resource(tier t) noexcept { return std::move(resources_[tier_index(t)]); }
  bool has_resource(tier t) const noexcept { return resources_[tier_index(t)] != nullptr; }

  // True iff no KV tier has a resource and no mamba slot is attached and no
  // children — used by `prune_empty`.
  bool is_orphan() const noexcept;

  // Mamba slot.
  bool has_mamba() const noexcept { return mamba_slot_ != nullptr; }
  std::int32_t mamba_index() const noexcept { return mamba_slot_ ? mamba_slot_->index() : -1; }
  void attach_mamba(mamba_slot slot) { mamba_slot_ = std::make_unique<mamba_slot>(std::move(slot)); }
  std::optional<mamba_slot> detach_mamba() noexcept;

  // Time/usage metadata.
  void touch(clock::time_point t) noexcept { last_access_time_ = t; }
  clock::time_point last_access_time() const noexcept { return last_access_time_; }
  std::uint64_t hit_count() const noexcept { return hit_count_; }
  void inc_hit() noexcept { ++hit_count_; }

  std::int64_t user_priority() const noexcept { return user_priority_; }
  void set_user_priority(std::int64_t p) noexcept { user_priority_ = p; }

  std::uint64_t last_access_step() const noexcept { return last_access_step_; }
  void set_last_access_step(std::uint64_t s) noexcept { last_access_step_ = s; }

  // Split this node so the first prefix_atom_count atoms remain here and the
  // suffix becomes a new child node which inherits all this node's children +
  // KV resources (cut proportionally) + mamba_slot_. Returns the new suffix
  // child (now in `children_`).
  tree_node* split_self(std::uint32_t prefix_atom_count);

 private:
  atom_vec atoms_;
  std::uint32_t atom_count_;
  std::uint32_t depth_in_atoms_;
  std::uint32_t atom_bytes_;
  std::uint32_t atoms_per_unit_;

  tree_node* parent_;
  children_map children_;

  std::array<std::unique_ptr<node_resource>, max_tiers> resources_;
  std::unique_ptr<mamba_slot> mamba_slot_;

  clock::time_point last_access_time_{};
  std::uint64_t hit_count_ = 0;
  std::int64_t user_priority_ = 0;
  std::uint64_t last_access_step_ = 0;
};

}  // namespace phyai_ext::radix_cache
