#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "radix_cache/mamba/mamba_slot.h"

namespace phyai_ext::radix_cache {

// LIFO free-list allocator for mamba slots. Slot ids are 0..num_slots-1; they
// live in their own namespace, separate from KV unit ids.
class mamba_allocator {
 public:
  explicit mamba_allocator(std::int32_t num_slots);

  mamba_allocator(const mamba_allocator&) = delete;
  mamba_allocator& operator=(const mamba_allocator&) = delete;
  mamba_allocator(mamba_allocator&&) noexcept = default;
  mamba_allocator& operator=(mamba_allocator&&) noexcept = default;

  std::optional<mamba_slot> allocate();
  void free(std::int32_t index);

  std::int32_t total_slots() const noexcept { return total_slots_; }
  std::int32_t available_slots() const noexcept { return static_cast<std::int32_t>(free_list_.size()); }
  std::int32_t active_slots() const noexcept { return total_slots_ - available_slots(); }

 private:
  std::int32_t total_slots_;
  std::vector<std::int32_t> free_list_;
};

}  // namespace phyai_ext::radix_cache
