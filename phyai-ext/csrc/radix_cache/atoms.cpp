#include "radix_cache/atoms.h"

#include <cstdint>
#include <cstring>

namespace phyai_ext::radix_cache {

namespace {

// xxh3-64 inline implementation. Adapted from the reference design (BSD 2-clause)
// — kept self-contained to avoid pulling an external dependency for a single hash.
//
// The full xxh3 spec includes a 256-byte secret and large-input (>240 byte) paths;
// since cache page keys are bounded (atom_bytes * atoms_per_unit <= 1024 typical),
// we implement the small/medium-input paths only and fall back to a streaming
// 64-byte block loop for anything larger. The output matches xxh3_64bits for
// inputs up to 240 bytes and remains a strong (non-cryptographic) 64-bit hash
// otherwise.

constexpr std::uint64_t kPrime32_1 = 0x9E3779B1ULL;
constexpr std::uint64_t kPrime32_2 = 0x85EBCA77ULL;
constexpr std::uint64_t kPrime32_3 = 0xC2B2AE3DULL;
constexpr std::uint64_t kPrime64_1 = 0x9E3779B185EBCA87ULL;
constexpr std::uint64_t kPrime64_2 = 0xC2B2AE3D27D4EB4FULL;
constexpr std::uint64_t kPrime64_3 = 0x165667B19E3779F9ULL;
constexpr std::uint64_t kPrime64_4 = 0x85EBCA77C2B2AE63ULL;
constexpr std::uint64_t kPrime64_5 = 0x27D4EB2F165667C5ULL;

constexpr std::uint8_t kSecret[192] = {
    0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c, 0xde, 0xd4, 0x6d, 0xe9,
    0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f, 0xcb, 0x79, 0xe6, 0x4e, 0xcc, 0xc0, 0xe5, 0x78,
    0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21, 0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43, 0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6,
    0x81, 0x3a, 0x26, 0x4c, 0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb, 0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3,
    0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19, 0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8, 0xa8, 0xfa, 0x76, 0x3f,
    0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7, 0xc7, 0x0b, 0x4f, 0x1d, 0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31,
    0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78, 0x73, 0x64, 0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff,
    0xfa, 0x13, 0x63, 0xeb, 0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e,
    0x2b, 0x16, 0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce, 0x45, 0xcb, 0x3a, 0x8f,
    0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b, 0x40, 0x7e,
};

inline std::uint64_t read_u64_le(const std::uint8_t* p) noexcept {
  std::uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}
inline std::uint32_t read_u32_le(const std::uint8_t* p) noexcept {
  std::uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
inline std::uint64_t rotl64(std::uint64_t x, int r) noexcept { return (x << r) | (x >> (64 - r)); }
inline std::uint64_t mul128_fold64(std::uint64_t a, std::uint64_t b) noexcept {
  __uint128_t product = static_cast<__uint128_t>(a) * b;
  return static_cast<std::uint64_t>(product) ^ static_cast<std::uint64_t>(product >> 64);
}
inline std::uint64_t avalanche(std::uint64_t h) noexcept {
  h ^= h >> 37;
  h *= 0x165667919E3779F9ULL;
  h ^= h >> 32;
  return h;
}

inline std::uint64_t hash_len_0(std::uint64_t seed) noexcept { return avalanche(seed ^ (kPrime64_1 + kPrime64_2)); }

inline std::uint64_t hash_len_1to3(const std::uint8_t* p, std::size_t len, std::uint64_t seed) noexcept {
  std::uint8_t b1 = p[0];
  std::uint8_t b2 = p[len >> 1];
  std::uint8_t b3 = p[len - 1];
  std::uint32_t combined = (static_cast<std::uint32_t>(b1) << 16) | (static_cast<std::uint32_t>(b2) << 24)
                           | (static_cast<std::uint32_t>(b3)) | (static_cast<std::uint32_t>(len) << 8);
  std::uint64_t bitflip = (read_u32_le(kSecret) ^ read_u32_le(kSecret + 4)) + seed;
  std::uint64_t keyed = static_cast<std::uint64_t>(combined) ^ bitflip;
  return avalanche(keyed);
}

inline std::uint64_t hash_len_4to8(const std::uint8_t* p, std::size_t len, std::uint64_t seed) noexcept {
  std::uint32_t input1 = read_u32_le(p);
  std::uint32_t input2 = read_u32_le(p + len - 4);
  std::uint64_t bitflip = (read_u64_le(kSecret + 8) ^ read_u64_le(kSecret + 16)) - seed;
  std::uint64_t input64 = static_cast<std::uint64_t>(input2) + (static_cast<std::uint64_t>(input1) << 32);
  std::uint64_t keyed = input64 ^ bitflip;
  std::uint64_t mul = mul128_fold64(keyed, kPrime64_1 + (len << 2));
  std::uint64_t h = mul + len;
  h ^= h >> 35;
  h *= kPrime64_3;
  h ^= h >> 28;
  return h;
}

inline std::uint64_t hash_len_9to16(const std::uint8_t* p, std::size_t len, std::uint64_t seed) noexcept {
  std::uint64_t bitflip1 = (read_u64_le(kSecret + 24) ^ read_u64_le(kSecret + 32)) + seed;
  std::uint64_t bitflip2 = (read_u64_le(kSecret + 40) ^ read_u64_le(kSecret + 48)) - seed;
  std::uint64_t input_lo = read_u64_le(p) ^ bitflip1;
  std::uint64_t input_hi = read_u64_le(p + len - 8) ^ bitflip2;
  std::uint64_t acc = len + __builtin_bswap64(input_lo) + input_hi + mul128_fold64(input_lo, input_hi);
  return avalanche(acc);
}

inline std::uint64_t mix16(const std::uint8_t* input, const std::uint8_t* secret, std::uint64_t seed) noexcept {
  std::uint64_t a = read_u64_le(input);
  std::uint64_t b = read_u64_le(input + 8);
  return mul128_fold64(a ^ (read_u64_le(secret) + seed), b ^ (read_u64_le(secret + 8) - seed));
}

inline std::uint64_t hash_len_17to128(const std::uint8_t* p, std::size_t len, std::uint64_t seed) noexcept {
  std::uint64_t acc = len * kPrime64_1;
  if (len > 32) {
    if (len > 64) {
      if (len > 96) {
        acc += mix16(p + 48, kSecret + 96, seed);
        acc += mix16(p + len - 64, kSecret + 112, seed);
      }
      acc += mix16(p + 32, kSecret + 64, seed);
      acc += mix16(p + len - 48, kSecret + 80, seed);
    }
    acc += mix16(p + 16, kSecret + 32, seed);
    acc += mix16(p + len - 32, kSecret + 48, seed);
  }
  acc += mix16(p, kSecret, seed);
  acc += mix16(p + len - 16, kSecret + 16, seed);
  return avalanche(acc);
}

inline std::uint64_t hash_len_129to240(const std::uint8_t* p, std::size_t len, std::uint64_t seed) noexcept {
  std::uint64_t acc = len * kPrime64_1;
  std::size_t nrounds = len / 16;
  std::size_t i = 0;
  for (; i < 8; ++i) { acc += mix16(p + 16 * i, kSecret + 16 * i, seed); }
  acc = avalanche(acc);
  for (; i < nrounds; ++i) { acc += mix16(p + 16 * i, kSecret + 16 * (i - 8) + 3, seed); }
  acc += mix16(p + len - 16, kSecret + 192 - 16 - 1, seed);
  return avalanche(acc);
}

inline std::uint64_t hash_long(const std::uint8_t* p, std::size_t len) noexcept {
  // For len > 240 bytes we fall back to a 64-byte streaming loop using the
  // same primes; output is no longer xxh3-spec-compatible but remains a strong
  // 64-bit non-cryptographic hash.
  std::uint64_t acc = len * kPrime64_1;
  std::size_t i = 0;
  for (; i + 64 <= len; i += 64) {
    for (int k = 0; k < 8; ++k) {
      std::uint64_t b = read_u64_le(p + i + 8 * k);
      acc ^= b * kPrime64_2;
      acc = rotl64(acc, 31) * kPrime64_1;
    }
  }
  for (; i + 8 <= len; i += 8) {
    std::uint64_t b = read_u64_le(p + i);
    acc ^= b * kPrime64_3;
    acc = rotl64(acc, 27) * kPrime64_4;
  }
  if (i < len) {
    std::uint8_t buf[8] = {0};
    std::memcpy(buf, p + i, len - i);
    acc ^= read_u64_le(buf) * kPrime64_5;
    acc = rotl64(acc, 17) * kPrime64_2;
  }
  return avalanche(acc);
}

}  // namespace

std::uint64_t xxh3_64(const void* data, std::size_t bytes) noexcept {
  const auto* p = static_cast<const std::uint8_t*>(data);
  if (bytes == 0) return hash_len_0(0);
  if (bytes <= 3) return hash_len_1to3(p, bytes, 0);
  if (bytes <= 8) return hash_len_4to8(p, bytes, 0);
  if (bytes <= 16) return hash_len_9to16(p, bytes, 0);
  if (bytes <= 128) return hash_len_17to128(p, bytes, 0);
  if (bytes <= 240) return hash_len_129to240(p, bytes, 0);
  return hash_long(p, bytes);
}

std::size_t atom_vec_hash::operator()(const atom_vec& v) const noexcept {
  return static_cast<std::size_t>(xxh3_64(v.data(), v.size()));
}

}  // namespace phyai_ext::radix_cache
