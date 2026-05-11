#pragma once

#include <cstddef>
#include <cstdint>

namespace phyai_ext::radix_cache {

enum class tier : std::uint8_t {
  device = 0,
  host = 1,
  disk = 2,
  remote = 3,
};

constexpr std::size_t max_tiers = 4;

constexpr std::size_t tier_index(tier t) noexcept { return static_cast<std::size_t>(t); }

constexpr tier tier_from_index(std::size_t i) noexcept { return static_cast<tier>(i); }

}  // namespace phyai_ext::radix_cache
