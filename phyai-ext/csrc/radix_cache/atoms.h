#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace phyai_ext::radix_cache {

using atom_span = std::span<const std::byte>;
using atom_vec = std::vector<std::byte>;

inline atom_span sub_span(atom_span whole, std::size_t atom_bytes, std::size_t atom_offset, std::size_t atom_count) noexcept {
  return whole.subspan(atom_offset * atom_bytes, atom_count * atom_bytes);
}

inline atom_vec to_vec(atom_span s) { return atom_vec(s.begin(), s.end()); }

// Hash function for byte vectors used as page keys in the radix tree's
// children map. Implementation is xxh3-64 for high throughput on multi-byte
// keys (text page = 64 B; multimodal page = 256 B).
struct atom_vec_hash {
  std::size_t operator()(const atom_vec& v) const noexcept;
};

struct atom_vec_eq {
  bool operator()(const atom_vec& a, const atom_vec& b) const noexcept {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()) == 0;
  }
};

// xxh3 64-bit hash of an arbitrary byte buffer. Exposed for callers that
// want page-level rolling hashes (e.g. cross-process / storage-backend key).
std::uint64_t xxh3_64(const void* data, std::size_t bytes) noexcept;

inline std::uint64_t xxh3_64(atom_span s) noexcept { return xxh3_64(s.data(), s.size()); }

}  // namespace phyai_ext::radix_cache
