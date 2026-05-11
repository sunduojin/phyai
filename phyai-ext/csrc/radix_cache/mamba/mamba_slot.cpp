#include "radix_cache/mamba/mamba_slot.h"

#include <utility>

#include "radix_cache/mamba/mamba_allocator.h"

namespace phyai_ext::radix_cache {

mamba_slot::~mamba_slot() {
  if (allocator_ != nullptr && index_ >= 0) { allocator_->free(index_); }
}

mamba_slot& mamba_slot::operator=(mamba_slot&& o) noexcept {
  if (this != &o) {
    if (allocator_ != nullptr && index_ >= 0) { allocator_->free(index_); }
    index_ = o.index_;
    allocator_ = o.allocator_;
    o.index_ = -1;
    o.allocator_ = nullptr;
  }
  return *this;
}

}  // namespace phyai_ext::radix_cache
