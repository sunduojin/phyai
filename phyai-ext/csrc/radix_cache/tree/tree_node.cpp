#include "radix_cache/tree/tree_node.h"

#include <stdexcept>
#include <utility>

namespace phyai_ext::radix_cache {

tree_node::tree_node(tree_node* parent, atom_vec atoms_bytes, std::uint32_t atom_bytes, std::uint32_t atoms_per_unit,
                     std::uint32_t depth_in_atoms)
    : atoms_(std::move(atoms_bytes)),
      atom_count_(static_cast<std::uint32_t>(atoms_.size() / (atom_bytes == 0 ? 1U : atom_bytes))),
      depth_in_atoms_(depth_in_atoms),
      atom_bytes_(atom_bytes),
      atoms_per_unit_(atoms_per_unit),
      parent_(parent) {}

bool tree_node::is_orphan() const noexcept {
  if (!children_.empty()) return false;
  if (mamba_slot_) return false;
  for (const auto& r : resources_) {
    if (r) return false;
  }
  return true;
}

std::optional<mamba_slot> tree_node::detach_mamba() noexcept {
  if (!mamba_slot_) return std::nullopt;
  auto out = std::optional<mamba_slot>(std::move(*mamba_slot_));
  mamba_slot_.reset();
  return out;
}

tree_node* tree_node::split_self(std::uint32_t prefix_atom_count) {
  if (prefix_atom_count == 0 || prefix_atom_count >= atom_count_) {
    throw std::logic_error("tree_node::split_self: prefix must be in (0, atom_count)");
  }
  if (prefix_atom_count % atoms_per_unit_ != 0) { throw std::logic_error("tree_node::split_self: prefix not aligned to page"); }
  const auto suffix_atoms_count = atom_count_ - prefix_atom_count;
  const auto suffix_byte_off = static_cast<std::size_t>(prefix_atom_count) * atom_bytes_;

  atom_vec suffix_atoms(atoms_.begin() + static_cast<std::ptrdiff_t>(suffix_byte_off), atoms_.end());
  atoms_.resize(suffix_byte_off);

  auto suffix = std::make_unique<tree_node>(this, std::move(suffix_atoms), atom_bytes_, atoms_per_unit_, depth_in_atoms_);
  // After splitting, this node's depth becomes (depth_in_atoms_ -
  // suffix_atoms_count); the suffix node inherits the original full depth.
  depth_in_atoms_ -= suffix_atoms_count;
  atom_count_ = static_cast<std::uint32_t>(atoms_.size() / atom_bytes_);
  // Re-parent existing children to suffix.
  suffix->children_ = std::move(children_);
  for (auto& [_, c] : suffix->children_) { c->set_parent(suffix.get()); }
  children_.clear();

  // Migrate KV resources: split owned_units proportionally; the first
  // (prefix/atoms_per_unit) units stay here, the rest go to the suffix.
  const auto prefix_units = prefix_atom_count / atoms_per_unit_;
  for (std::size_t t = 0; t < max_tiers; ++t) {
    if (!resources_[t]) continue;
    auto suffix_res = std::make_unique<node_resource>(resources_[t]->split_first(prefix_units));
    // After split_first the original holds the suffix units; swap so the
    // prefix stays here.
    std::swap(resources_[t], suffix_res);
    suffix->resources_[t] = std::move(suffix_res);
  }

  // Mamba slot represents state at the deeper boundary, so it follows the
  // suffix when a node is split.
  if (mamba_slot_) { suffix->mamba_slot_ = std::move(mamba_slot_); }

  // Use suffix as the children_ key for the new child (first page of its bytes).
  const std::size_t key_bytes = static_cast<std::size_t>(atoms_per_unit_) * atom_bytes_;
  atom_vec key(suffix->atoms_.begin(), suffix->atoms_.begin() + static_cast<std::ptrdiff_t>(key_bytes));
  auto* raw = suffix.get();
  children_.emplace(std::move(key), std::move(suffix));
  return raw;
}

}  // namespace phyai_ext::radix_cache
