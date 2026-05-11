#include "radix_cache/tree/radix_tree.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace phyai_ext::radix_cache {

radix_tree::radix_tree(std::uint32_t atom_bytes, std::uint32_t atoms_per_unit)
    : atom_bytes_(atom_bytes), atoms_per_unit_(atoms_per_unit) {
  if (atom_bytes == 0 || atoms_per_unit == 0) {
    throw std::invalid_argument("radix_tree: atom_bytes and atoms_per_unit must be > 0");
  }
  root_ = std::make_unique<tree_node>(/*parent=*/nullptr, atom_vec{}, atom_bytes, atoms_per_unit,
                                      /*depth=*/0);
}

namespace {

inline std::uint32_t common_atom_prefix(atom_span a, atom_span b, std::uint32_t atom_bytes) noexcept {
  const auto na = static_cast<std::uint32_t>(a.size() / atom_bytes);
  const auto nb = static_cast<std::uint32_t>(b.size() / atom_bytes);
  const auto maxn = std::min(na, nb);
  std::uint32_t i = 0;
  for (; i < maxn; ++i) {
    if (std::memcmp(a.data() + static_cast<std::ptrdiff_t>(i) * atom_bytes,
                    b.data() + static_cast<std::ptrdiff_t>(i) * atom_bytes, atom_bytes)
        != 0) {
      break;
    }
  }
  return i;
}

}  // namespace

radix_tree::walk_result radix_tree::walk(atom_span query, clock::time_point now) {
  walk_result out;
  for (std::size_t t = 0; t < max_tiers; ++t) {
    out.last_node[t] = root_.get();
    out.matched_atoms[t] = 0;
  }
  std::array<bool, max_tiers> tier_alive{};
  tier_alive.fill(true);

  tree_node* current = root_.get();
  std::uint32_t consumed_atoms = 0;
  const std::uint32_t total_query_atoms = static_cast<std::uint32_t>(query.size() / atom_bytes_);

  while (consumed_atoms + atoms_per_unit_ <= total_query_atoms) {
    // Build the page-key for the next branch: atoms_per_unit_ atoms starting at consumed_atoms.
    atom_vec child_key = to_vec(sub_span(query, atom_bytes_, consumed_atoms, atoms_per_unit_));
    auto it = current->children().find(child_key);
    if (it == current->children().end()) break;
    tree_node* child = it->second.get();

    // Compute longest aligned-atom match between the child's stored atoms and the query suffix.
    const std::uint32_t available_atoms = total_query_atoms - consumed_atoms;
    const std::uint32_t check_count = std::min(child->atom_count(), available_atoms);
    atom_span child_atoms = child->atoms();
    atom_span query_tail = sub_span(query, atom_bytes_, consumed_atoms, check_count);
    std::uint32_t matched = common_atom_prefix(child_atoms.subspan(0, check_count * atom_bytes_), query_tail, atom_bytes_);
    matched = (matched / atoms_per_unit_) * atoms_per_unit_;
    if (matched == 0) break;

    if (matched < child->atom_count()) {
      // Partial match — split the child so its first `matched` atoms become its segment.
      tree_node* suffix = child->split_self(matched);
      (void)suffix;  // suffix kept under child as its sole new descendant; we stay at `child`.
    }

    child->touch(now);
    consumed_atoms += matched;
    current = child;

    for (std::size_t t = 0; t < max_tiers; ++t) {
      if (!tier_alive[t]) continue;
      auto* res = current->resource(tier_from_index(t));
      if (res != nullptr && res->state() == resource_state::ready) {
        out.last_node[t] = current;
        out.matched_atoms[t] = consumed_atoms;
      } else {
        tier_alive[t] = false;  // tier breaks at this node (resource missing or pending/failed)
      }
    }

    // If we consumed less than the full child atoms (i.e. a partial split happened), break:
    // the next page-key would not match the new (deeper) child since we've already stopped at `current`.
    if (matched < child->atom_count()) break;
  }
  return out;
}

tree_node* radix_tree::insert_suffix(tree_node* base, atom_span suffix) {
  if (suffix.empty()) return base;
  if (suffix.size() % (static_cast<std::size_t>(atom_bytes_) * atoms_per_unit_) != 0) {
    throw std::invalid_argument("insert_suffix: suffix not page-aligned");
  }
  // The radix tree's branching unit is the page (`atoms_per_unit` atoms);
  // the children-key is the first page worth of bytes. The whole suffix is
  // appended as a single multi-page child node — partial future matches will
  // split it via `tree_node::split_self`.
  const std::size_t page_bytes = static_cast<std::size_t>(atom_bytes_) * atoms_per_unit_;
  const std::uint32_t suffix_atoms = static_cast<std::uint32_t>(suffix.size() / atom_bytes_);
  atom_vec body(suffix.begin(), suffix.end());
  atom_vec key(suffix.begin(), suffix.begin() + static_cast<std::ptrdiff_t>(page_bytes));
  auto child =
      std::make_unique<tree_node>(base, std::move(body), atom_bytes_, atoms_per_unit_, base->depth_in_atoms() + suffix_atoms);
  auto* raw = child.get();
  base->children().emplace(std::move(key), std::move(child));
  return raw;
}

void radix_tree::prune_empty(tree_node* from) {
  while (from != nullptr && from != root_.get() && from->is_orphan()) {
    auto* parent = from->parent();
    if (parent == nullptr) break;
    const std::size_t key_bytes = static_cast<std::size_t>(atoms_per_unit_) * atom_bytes_;
    atom_vec key(from->atoms_vec().begin(), from->atoms_vec().begin() + static_cast<std::ptrdiff_t>(key_bytes));
    auto it = parent->children().find(key);
    tree_node* next = parent;
    if (it != parent->children().end() && it->second.get() == from) { parent->children().erase(it); }
    from = next;
  }
}

}  // namespace phyai_ext::radix_cache
