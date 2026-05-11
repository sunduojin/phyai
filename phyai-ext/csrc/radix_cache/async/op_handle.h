#pragma once

#include <cstdint>

namespace phyai_ext::radix_cache {

// Async op handle. 0 == no pending op.
using op_handle = std::uint64_t;
constexpr op_handle null_op_handle = 0;

// Per-resource state machine for asynchronous transitions between tiers.
enum class resource_state : std::uint8_t {
  ready = 0,    // data is valid; participates in match / lock / evict.
  pending = 1,  // async op in flight; units allocated but contents not ready.
  failed = 2,   // async op failed; caller must clean up.
};

}  // namespace phyai_ext::radix_cache
