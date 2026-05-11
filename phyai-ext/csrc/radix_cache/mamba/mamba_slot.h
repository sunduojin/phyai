#pragma once

#include <cstdint>

namespace phyai_ext::radix_cache {

class mamba_allocator;

// RAII handle for a single mamba state-slot index. Move-only; the destructor
// pushes the slot back to its originating allocator.
class mamba_slot {
 public:
  mamba_slot() = default;
  mamba_slot(std::int32_t index, mamba_allocator* alloc) noexcept : index_(index), allocator_(alloc) {}
  ~mamba_slot();

  mamba_slot(const mamba_slot&) = delete;
  mamba_slot& operator=(const mamba_slot&) = delete;
  mamba_slot(mamba_slot&& o) noexcept : index_(o.index_), allocator_(o.allocator_) {
    o.index_ = -1;
    o.allocator_ = nullptr;
  }
  mamba_slot& operator=(mamba_slot&& o) noexcept;

  std::int32_t index() const noexcept { return index_; }
  bool valid() const noexcept { return allocator_ != nullptr && index_ >= 0; }
  mamba_allocator* allocator() const noexcept { return allocator_; }

  // Renounce RAII; caller responsible for `allocator->free(index)`.
  std::int32_t release() noexcept {
    auto out = index_;
    allocator_ = nullptr;
    index_ = -1;
    return out;
  }

 private:
  std::int32_t index_ = -1;
  mamba_allocator* allocator_ = nullptr;
};

}  // namespace phyai_ext::radix_cache
