#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>

#include "radix_cache/atoms.h"
#include "radix_cache/tree/tree_node.h"
#include "phyai_ext/radix_cache/tier.h"

namespace phyai_ext::radix_cache {

class radix_tree {
 public:
  using clock = std::chrono::steady_clock;

  radix_tree(std::uint32_t atom_bytes, std::uint32_t atoms_per_unit);

  radix_tree(const radix_tree&) = delete;
  radix_tree& operator=(const radix_tree&) = delete;

  // Per-tier last_node + matched_atoms after a single Walk.
  struct walk_result {
    std::array<tree_node*, max_tiers> last_node{};
    std::array<std::uint32_t, max_tiers> matched_atoms{};
  };

  // Walk down `query` advancing N tiers in lock-step. Once a tier hits a
  // missing or non-Ready node_resource it permanently stops advancing
  // (`tier_alive[t] = false`) — others may continue. Splits child nodes when
  // partial atom matches occur.
  walk_result walk(atom_span query, clock::time_point now);

  // After a Walk that ended at `base`, append `suffix` atoms as new tail nodes
  // under base. Returns the deepest newly-created (or returned-as-is) node.
  // Suffix is split into chunks of `atoms_per_unit` atoms.
  tree_node* insert_suffix(tree_node* base, atom_span suffix);

  tree_node* root() noexcept { return root_.get(); }
  const tree_node* root() const noexcept { return root_.get(); }

  std::uint32_t atom_bytes() const noexcept { return atom_bytes_; }
  std::uint32_t atoms_per_unit() const noexcept { return atoms_per_unit_; }
  std::uint32_t page_bytes() const noexcept { return atom_bytes_ * atoms_per_unit_; }

  // Walk up from `from` removing nodes with no resources / no children.
  void prune_empty(tree_node* from);

 private:
  std::uint32_t atom_bytes_;
  std::uint32_t atoms_per_unit_;
  std::unique_ptr<tree_node> root_;
};

}  // namespace phyai_ext::radix_cache
