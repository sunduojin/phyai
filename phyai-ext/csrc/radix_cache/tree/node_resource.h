#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>

#include "radix_cache/async/op_handle.h"
#include "radix_cache/owned_units.h"
#include "phyai_ext/radix_cache/tier.h"

namespace phyai_ext::radix_cache {

// One tier's resource on a tree node: ref-counted owned_units plus a small
// state machine that lets the prefix_cache express "data not yet ready"
// (during async demote/promote) without dropping the unit reservation.
class node_resource {
 public:
  node_resource() = default;
  explicit node_resource(owned_units&& units) : units_(std::move(units)) {}

  node_resource(const node_resource&) = delete;
  node_resource& operator=(const node_resource&) = delete;
  node_resource(node_resource&&) noexcept = default;
  node_resource& operator=(node_resource&&) noexcept = default;

  tier get_tier() const noexcept { return units_.get_tier(); }

  void lock() noexcept { ++ref_count_; }
  void unlock() {
    if (ref_count_ == 0) { throw std::logic_error("node_resource::unlock underflow"); }
    --ref_count_;
  }
  std::uint32_t ref_count() const noexcept { return ref_count_; }

  bool is_evictable() const noexcept { return ref_count_ == 0 && state_ == resource_state::ready; }

  std::size_t size_in_units() const noexcept { return units_.size(); }
  std::span<const unit_id> ids() const noexcept { return units_.ids(); }

  resource_state state() const noexcept { return state_; }
  op_handle pending_handle() const noexcept { return handle_; }

  // Take ownership; afterwards the node_resource is empty and should be
  // destroyed by the caller (typically detach + drop).
  owned_units take_units() { return std::move(units_); }

  // Split off the first prefix_units worth of units into a new node_resource;
  // ref_count and state are duplicated to the new piece.
  node_resource split_first(std::size_t prefix_units) {
    auto pre = units_.take_first(prefix_units);
    node_resource out(std::move(pre));
    out.ref_count_ = ref_count_;
    out.state_ = state_;
    out.handle_ = handle_;
    return out;
  }

 private:
  // friends manipulate state directly through the async op path.
  friend class prefix_cache;
  friend class tree_node;

  void set_pending(op_handle h) noexcept {
    state_ = resource_state::pending;
    handle_ = h;
  }
  void set_ready() noexcept {
    state_ = resource_state::ready;
    handle_ = null_op_handle;
  }
  void set_failed() noexcept { state_ = resource_state::failed; }

  owned_units units_;
  std::uint32_t ref_count_ = 0;
  resource_state state_ = resource_state::ready;
  op_handle handle_ = null_op_handle;
};

}  // namespace phyai_ext::radix_cache
