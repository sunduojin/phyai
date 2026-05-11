#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "phyai_ext/radix_cache/tier.h"
#include "radix_cache/owned_units.h"

namespace phyai_ext::radix_cache {

// LIFO free-list allocator for unit ids 1..total_units. Id 0 is reserved as
// the null/dummy sentinel (`null_unit`) so the layer above can safely use
// "0" to mean "no unit assigned".
class unit_allocator {
 public:
  unit_allocator(tier t, std::int32_t total_units);

  unit_allocator(const unit_allocator&) = delete;
  unit_allocator& operator=(const unit_allocator&) = delete;
  unit_allocator(unit_allocator&&) noexcept = default;
  unit_allocator& operator=(unit_allocator&&) noexcept = default;

  // Allocate n units; throws std::runtime_error on shortage.
  owned_units allocate(std::size_t n);

  // Internal: invoked by ~owned_units via friend.
  void deallocate(std::span<const unit_id> ids);

  tier get_tier() const noexcept { return tier_; }
  std::int32_t total_units() const noexcept { return total_units_; }
  std::int32_t available_units() const noexcept { return static_cast<std::int32_t>(free_units_.size()); }
  std::int32_t active_units() const noexcept {
    if (total_units_ == 0) return 0;
    return total_units_ - 1 - available_units();
  }

  // Reset all unit ids back to free. Used during recovery after marking
  // every in-flight op as failed.
  void reset_to_full();

 private:
  friend class owned_units;
  tier tier_;
  std::int32_t total_units_;
  std::vector<unit_id> free_units_;
};

}  // namespace phyai_ext::radix_cache
