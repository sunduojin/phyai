#include "radix_cache/unit_allocator.h"

#include <stdexcept>

namespace phyai_ext::radix_cache {

unit_allocator::unit_allocator(tier t, std::int32_t total_units) : tier_(t), total_units_(total_units) {
  if (total_units < 0) { throw std::invalid_argument("unit_allocator: negative total_units"); }
  if (total_units > 0) {
    free_units_.reserve(static_cast<std::size_t>(total_units - 1));
    for (std::int32_t i = total_units - 1; i >= 1; --i) { free_units_.push_back(i); }
  }
}

owned_units unit_allocator::allocate(std::size_t n) {
  if (n == 0) { return owned_units(this, tier_, std::vector<unit_id>{}); }
  if (free_units_.size() < n) { throw std::runtime_error("unit_allocator::allocate: not enough units available"); }
  std::vector<unit_id> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(free_units_.back());
    free_units_.pop_back();
  }
  return owned_units(this, tier_, std::move(out));
}

void unit_allocator::deallocate(std::span<const unit_id> ids) {
  for (auto id : ids) {
    if (id <= 0 || id >= total_units_) continue;
    free_units_.push_back(id);
  }
}

void unit_allocator::reset_to_full() {
  free_units_.clear();
  if (total_units_ > 0) {
    free_units_.reserve(static_cast<std::size_t>(total_units_ - 1));
    for (std::int32_t i = total_units_ - 1; i >= 1; --i) { free_units_.push_back(i); }
  }
}

}  // namespace phyai_ext::radix_cache
