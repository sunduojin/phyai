#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "phyai_ext/radix_cache/tier.h"

namespace phyai_ext::radix_cache {

using unit_id = std::int32_t;
constexpr unit_id null_unit = 0;

class unit_allocator;

// RAII handle for a contiguous batch of unit ids. Move-only; destructor
// returns the units back to the originating allocator.
class owned_units {
 public:
  owned_units() = default;
  ~owned_units();

  owned_units(owned_units&& o) noexcept;
  owned_units& operator=(owned_units&& o) noexcept;

  owned_units(const owned_units&) = delete;
  owned_units& operator=(const owned_units&) = delete;

  tier get_tier() const noexcept { return tier_; }
  std::span<const unit_id> ids() const noexcept { return {ids_.data(), ids_.size()}; }
  std::size_t size() const noexcept { return ids_.size(); }
  bool empty() const noexcept { return ids_.empty(); }
  unit_allocator* allocator() const noexcept { return allocator_; }

  // Split the first n units into a new handle (same allocator, same tier);
  // the original keeps the suffix. Throws if n > size().
  owned_units take_first(std::size_t n);

  // Symmetric: split the last n units into a new handle.
  owned_units take_last(std::size_t n);

  // Concatenate; same allocator + tier required (logic_error otherwise).
  void append(owned_units&& other);

  // Renounce RAII; the caller becomes responsible for freeing.
  std::vector<unit_id> release() &&;

 private:
  friend class unit_allocator;
  owned_units(unit_allocator* alloc, tier t, std::vector<unit_id>&& ids) : allocator_(alloc), tier_(t), ids_(std::move(ids)) {}

  void free();

  unit_allocator* allocator_ = nullptr;
  tier tier_ = tier::device;
  std::vector<unit_id> ids_;
};

}  // namespace phyai_ext::radix_cache
