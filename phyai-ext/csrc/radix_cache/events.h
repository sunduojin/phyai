#pragma once

#include <cstdint>

#include "radix_cache/async/op_handle.h"
#include "phyai_ext/radix_cache/tier.h"

namespace phyai_ext::radix_cache {

enum class cache_event_kind : std::uint8_t {
  insert = 0,
  evict = 1,
  promote_start = 2,
  promote_done = 3,
  promote_fail = 4,
  demote_start = 5,
  demote_done = 6,
  demote_fail = 7,
  split = 8,
};

struct cache_event {
  cache_event_kind kind = cache_event_kind::insert;
  tier tier_from = tier::device;
  tier tier_to = tier::device;
  op_handle handle = null_op_handle;
  std::uint32_t atom_count = 0;
  std::uint32_t unit_count = 0;
  std::uint64_t ts_ns = 0;
};

}  // namespace phyai_ext::radix_cache
