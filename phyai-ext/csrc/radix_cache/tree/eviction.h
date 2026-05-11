#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include "radix_cache/tree/tree_node.h"
#include "phyai_ext/radix_cache/tier.h"

namespace phyai_ext::radix_cache {

// Three-slot priority key. Lex compare; smaller values are evicted first.
// Three slots cover LRU, LFU, SLRU and Priority policies; the last slot is
// available for tie-breakers (e.g. node-pointer hash).
using eviction_key = std::tuple<std::int64_t, std::int64_t, std::int64_t>;

class eviction_policy {
 public:
  virtual ~eviction_policy() = default;
  virtual eviction_key priority(const tree_node& n, tier t) const = 0;
  virtual std::string name() const = 0;
};

namespace detail {
inline std::int64_t to_ns(tree_node::clock::time_point tp) noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}
}  // namespace detail

class lru_policy : public eviction_policy {
 public:
  eviction_key priority(const tree_node& n, tier /*t*/) const override { return {detail::to_ns(n.last_access_time()), 0, 0}; }
  std::string name() const override { return "lru"; }
};

class lfu_policy : public eviction_policy {
 public:
  eviction_key priority(const tree_node& n, tier /*t*/) const override {
    return {static_cast<std::int64_t>(n.hit_count()), detail::to_ns(n.last_access_time()), 0};
  }
  std::string name() const override { return "lfu"; }
};

class slru_policy : public eviction_policy {
 public:
  explicit slru_policy(std::int64_t protect_threshold) : threshold_(protect_threshold) {}
  std::int64_t threshold() const noexcept { return threshold_; }
  eviction_key priority(const tree_node& n, tier /*t*/) const override {
    const std::int64_t protected_ = (static_cast<std::int64_t>(n.hit_count()) >= threshold_) ? 1 : 0;
    return {protected_, detail::to_ns(n.last_access_time()), 0};
  }
  std::string name() const override { return "slru"; }

 private:
  std::int64_t threshold_;
};

class priority_policy : public eviction_policy {
 public:
  eviction_key priority(const tree_node& n, tier /*t*/) const override {
    return {n.user_priority(), detail::to_ns(n.last_access_time()), 0};
  }
  std::string name() const override { return "priority"; }
};

// Build a policy by name. Optional `slru_threshold` is honoured for the SLRU
// policy (callers that want a non-default protect-threshold pass it through).
std::unique_ptr<eviction_policy> make_eviction_policy(const std::string& name, std::int64_t slru_threshold = 2);

// Convenience predicate factories used by `evict_by_predicate`. They live
// here so callers can build common SWA-style cutoffs without crossing the FFI
// boundary for every candidate, which is much cheaper than a Python callback
// when the candidate set is large.
using node_predicate = std::function<bool(const tree_node&)>;

inline node_predicate predicate_step_le(std::uint64_t cutoff) {
  return [cutoff](const tree_node& n) noexcept { return n.last_access_step() <= cutoff; };
}

inline node_predicate predicate_step_lt(std::uint64_t cutoff) {
  return [cutoff](const tree_node& n) noexcept { return n.last_access_step() < cutoff; };
}

inline node_predicate predicate_hits_lt(std::uint64_t threshold) {
  return [threshold](const tree_node& n) noexcept { return n.hit_count() < threshold; };
}

inline node_predicate predicate_priority_le(std::int64_t cutoff) {
  return [cutoff](const tree_node& n) noexcept { return n.user_priority() <= cutoff; };
}

inline node_predicate predicate_age_ns_ge(std::int64_t age_ns, tree_node::clock::time_point now) {
  const auto cutoff = now - std::chrono::nanoseconds(age_ns);
  return [cutoff](const tree_node& n) noexcept { return n.last_access_time() <= cutoff; };
}

}  // namespace phyai_ext::radix_cache
