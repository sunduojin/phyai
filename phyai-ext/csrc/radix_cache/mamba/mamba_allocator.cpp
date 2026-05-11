#include "radix_cache/mamba/mamba_allocator.h"

#include <stdexcept>

namespace phyai_ext::radix_cache {

mamba_allocator::mamba_allocator(std::int32_t num_slots) : total_slots_(num_slots) {
  if (num_slots < 0) { throw std::invalid_argument("mamba_allocator: negative num_slots"); }
  free_list_.reserve(static_cast<std::size_t>(num_slots));
  for (std::int32_t i = num_slots - 1; i >= 0; --i) { free_list_.push_back(i); }
}

std::optional<mamba_slot> mamba_allocator::allocate() {
  if (free_list_.empty()) { return std::nullopt; }
  auto idx = free_list_.back();
  free_list_.pop_back();
  return mamba_slot(idx, this);
}

void mamba_allocator::free(std::int32_t index) {
  if (index < 0 || index >= total_slots_) return;
  free_list_.push_back(index);
}

}  // namespace phyai_ext::radix_cache
