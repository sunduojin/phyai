#include "radix_cache/tree/eviction.h"

namespace phyai_ext::radix_cache {

std::unique_ptr<eviction_policy> make_eviction_policy(const std::string& name, std::int64_t slru_threshold) {
  if (name == "lru") return std::make_unique<lru_policy>();
  if (name == "lfu") return std::make_unique<lfu_policy>();
  if (name == "slru") return std::make_unique<slru_policy>(slru_threshold);
  if (name == "priority") return std::make_unique<priority_policy>();
  throw std::invalid_argument("unknown eviction policy: " + name);
}

}  // namespace phyai_ext::radix_cache
