#include "radix_cache/owned_units.h"

#include <stdexcept>
#include <utility>

#include "radix_cache/unit_allocator.h"

namespace phyai_ext::radix_cache {

owned_units::~owned_units() {
  if (allocator_ != nullptr && !ids_.empty()) { free(); }
}

owned_units::owned_units(owned_units&& o) noexcept : allocator_(o.allocator_), tier_(o.tier_), ids_(std::move(o.ids_)) {
  o.allocator_ = nullptr;
}

owned_units& owned_units::operator=(owned_units&& o) noexcept {
  if (this != &o) {
    if (allocator_ != nullptr && !ids_.empty()) { free(); }
    allocator_ = o.allocator_;
    tier_ = o.tier_;
    ids_ = std::move(o.ids_);
    o.allocator_ = nullptr;
  }
  return *this;
}

owned_units owned_units::take_first(std::size_t n) {
  if (n > ids_.size()) { throw std::logic_error("owned_units::take_first: n > size"); }
  std::vector<unit_id> head(ids_.begin(), ids_.begin() + static_cast<std::ptrdiff_t>(n));
  ids_.erase(ids_.begin(), ids_.begin() + static_cast<std::ptrdiff_t>(n));
  return owned_units(allocator_, tier_, std::move(head));
}

owned_units owned_units::take_last(std::size_t n) {
  if (n > ids_.size()) { throw std::logic_error("owned_units::take_last: n > size"); }
  std::vector<unit_id> tail(ids_.end() - static_cast<std::ptrdiff_t>(n), ids_.end());
  ids_.erase(ids_.end() - static_cast<std::ptrdiff_t>(n), ids_.end());
  return owned_units(allocator_, tier_, std::move(tail));
}

void owned_units::append(owned_units&& other) {
  if (other.empty()) return;
  if (empty()) {
    allocator_ = other.allocator_;
    tier_ = other.tier_;
  } else if (allocator_ != other.allocator_ || tier_ != other.tier_) {
    throw std::logic_error("owned_units::append: allocator or tier mismatch");
  }
  ids_.insert(ids_.end(), other.ids_.begin(), other.ids_.end());
  other.ids_.clear();
  other.allocator_ = nullptr;
}

std::vector<unit_id> owned_units::release() && {
  allocator_ = nullptr;
  return std::move(ids_);
}

void owned_units::free() {
  allocator_->deallocate({ids_.data(), ids_.size()});
  ids_.clear();
  allocator_ = nullptr;
}

}  // namespace phyai_ext::radix_cache
