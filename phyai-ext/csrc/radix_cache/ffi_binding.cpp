// FFI registration: exposes prefix_cache, hybrid_prefix_cache, storage_backend
// and the small RAII helper objects (owned_units, node_ref, mamba_slot,
// composite_node_ref, cache_event) to Python through tvm-ffi.
#include "radix_cache/ffi_objects.h"

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "radix_cache/atoms.h"
#include "radix_cache/events.h"
#include "radix_cache/owned_units.h"
#include "radix_cache/prefix_cache.h"
#include "radix_cache/storage_backend.h"
#include "radix_cache/tree/tree_node.h"
#include "phyai_ext/radix_cache/tier.h"

namespace phyai_ext::radix_cache::ffi {

namespace {

inline atom_span bytes_to_atom_span(const tvm::ffi::Bytes& b) noexcept {
  return atom_span{reinterpret_cast<const std::byte*>(b.data()), b.size()};
}

inline tvm::ffi::Array<int64_t> ids_to_array(std::span<const unit_id> ids) {
  std::vector<int64_t> v;
  v.reserve(ids.size());
  for (auto i : ids) v.push_back(static_cast<int64_t>(i));
  return tvm::ffi::Array<int64_t>(v);
}

inline tvm::ffi::Array<int64_t> ids_to_array(const std::vector<unit_id>& ids) {
  std::vector<int64_t> v;
  v.reserve(ids.size());
  for (auto i : ids) v.push_back(static_cast<int64_t>(i));
  return tvm::ffi::Array<int64_t>(v);
}

inline ::phyai_ext::radix_cache::tier int_to_tier(int64_t i) {
  if (i < 0 || i >= static_cast<int64_t>(max_tiers)) { throw cache_usage_error("invalid tier index"); }
  return static_cast<::phyai_ext::radix_cache::tier>(i);
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

static prefix_cache_ref prefix_cache_create(int64_t atom_bytes, int64_t atoms_per_unit,
                                            tvm::ffi::Array<int64_t> tier_total_units,
                                            tvm::ffi::Array<int64_t> tier_max_pending_units, tvm::ffi::String policy_name,
                                            int64_t slru_threshold, int64_t max_events_buffered) {
  if (tier_total_units.size() != static_cast<int64_t>(max_tiers)) {
    throw cache_usage_error("prefix_cache.create: tier_total_units must have max_tiers entries");
  }
  if (tier_max_pending_units.size() != static_cast<int64_t>(max_tiers)) {
    throw cache_usage_error("prefix_cache.create: tier_max_pending_units must have max_tiers entries");
  }
  prefix_cache_config cfg;
  cfg.atom_bytes = static_cast<uint32_t>(atom_bytes);
  cfg.atoms_per_unit = static_cast<uint32_t>(atoms_per_unit);
  cfg.eviction_policy = std::string(policy_name.data(), policy_name.size());
  cfg.slru_threshold = slru_threshold;
  cfg.max_events_buffered = max_events_buffered < 0 ? 0 : static_cast<std::size_t>(max_events_buffered);
  for (size_t t = 0; t < max_tiers; ++t) {
    auto n = static_cast<int32_t>(tier_total_units[t]);
    auto p = static_cast<int32_t>(tier_max_pending_units[t]);
    cfg.tiers[t].enabled = (n > 0);
    cfg.tiers[t].total_units = n;
    cfg.tiers[t].max_pending_units = p;
    cfg.tiers[t].is_async =
        (t == tier_index(::phyai_ext::radix_cache::tier::disk) || t == tier_index(::phyai_ext::radix_cache::tier::remote));
  }
  auto impl = std::make_shared<prefix_cache>(std::move(cfg));
  return prefix_cache_ref(std::move(impl));
}

static hybrid_prefix_cache_ref hybrid_prefix_cache_create(prefix_cache_ref kv, int64_t num_mamba_slots) {
  if (!kv.defined() || kv.get() == nullptr || kv.get()->impl == nullptr) {
    throw cache_usage_error("hybrid_prefix_cache.create: kv cache is null");
  }
  auto impl = std::make_shared<hybrid_prefix_cache>(kv.get()->impl, static_cast<int32_t>(num_mamba_slots));
  return hybrid_prefix_cache_ref(std::move(impl), kv);
}

// ---------------------------------------------------------------------------
// prefix_cache methods
// ---------------------------------------------------------------------------

static tvm::ffi::Array<tvm::ffi::Any> prefix_cache_match(prefix_cache_ref self, tvm::ffi::Bytes atoms) {
  auto& pc = *self.get()->impl;
  auto m = pc.match(bytes_to_atom_span(atoms));
  std::vector<int64_t> last_nodes_v;
  std::vector<int64_t> matched_atoms_v;
  last_nodes_v.reserve(max_tiers);
  matched_atoms_v.reserve(max_tiers);
  for (size_t t = 0; t < max_tiers; ++t) {
    last_nodes_v.push_back(reinterpret_cast<int64_t>(m.last_node[t]));
    matched_atoms_v.push_back(static_cast<int64_t>(m.matched_atoms[t]));
  }
  tvm::ffi::Array<tvm::ffi::Any> out;
  out.push_back(tvm::ffi::Any(tvm::ffi::Array<int64_t>(last_nodes_v)));
  out.push_back(tvm::ffi::Any(tvm::ffi::Array<int64_t>(matched_atoms_v)));
  return out;
}

static tvm::ffi::Tensor prefix_cache_collect_units(prefix_cache_ref self, int64_t node_handle, int64_t tier_int) {
  auto* node = reinterpret_cast<tree_node*>(node_handle);
  return self.get()->impl->collect_units_dlpack(node, int_to_tier(tier_int));
}

static tvm::ffi::Array<int64_t> prefix_cache_insert(prefix_cache_ref self, int64_t tier_int, tvm::ffi::Bytes atoms,
                                                    owned_units_ref units_ref) {
  auto& pc = *self.get()->impl;
  auto t = int_to_tier(tier_int);
  owned_units units = std::move(units_ref.get()->impl);
  auto r = pc.insert(t, bytes_to_atom_span(atoms), std::move(units));
  return tvm::ffi::Array<int64_t>(std::vector<int64_t>{
      reinterpret_cast<int64_t>(r.last_node),
      static_cast<int64_t>(r.inserted_atoms),
      static_cast<int64_t>(r.freed_units_due_to_overlap),
  });
}

static owned_units_ref prefix_cache_allocate(prefix_cache_ref self, int64_t tier_int, int64_t n) {
  auto& pc = *self.get()->impl;
  return owned_units_ref(pc.allocate(int_to_tier(tier_int), static_cast<int32_t>(n)));
}

static node_ref_ref prefix_cache_lock(prefix_cache_ref self, int64_t tier_int, int64_t node_handle) {
  auto& pc = *self.get()->impl;
  auto* node = reinterpret_cast<tree_node*>(node_handle);
  return node_ref_ref(pc.lock(int_to_tier(tier_int), node));
}

static composite_node_ref_ref prefix_cache_lock_multi(prefix_cache_ref self, int64_t node_handle,
                                                      tvm::ffi::Array<int64_t> tiers) {
  auto* node = reinterpret_cast<tree_node*>(node_handle);
  std::vector<::phyai_ext::radix_cache::tier> ts;
  ts.reserve(tiers.size());
  for (int64_t v : tiers) ts.push_back(int_to_tier(v));
  (void)self;
  return composite_node_ref_ref(composite_node_ref(node, ts));
}

static void prefix_cache_ensure_capacity(prefix_cache_ref self, int64_t tier_int, int64_t need_units, int64_t promote_to) {
  auto& pc = *self.get()->impl;
  std::optional<::phyai_ext::radix_cache::tier> promote =
      (promote_to < 0) ? std::nullopt : std::optional<::phyai_ext::radix_cache::tier>(int_to_tier(promote_to));
  pc.ensure_capacity(int_to_tier(tier_int), static_cast<int32_t>(need_units), promote);
}

static int64_t prefix_cache_evict_by_predicate(prefix_cache_ref self, int64_t tier_int, tvm::ffi::Function callback) {
  auto& pc = *self.get()->impl;
  return static_cast<int64_t>(pc.evict_by_predicate(int_to_tier(tier_int), [&](const tree_node& n) -> bool {
    int64_t node_addr = reinterpret_cast<int64_t>(&n);
    int64_t depth = static_cast<int64_t>(n.depth_in_atoms());
    int64_t hits = static_cast<int64_t>(n.hit_count());
    int64_t step = static_cast<int64_t>(n.last_access_step());
    int64_t prio = n.user_priority();
    tvm::ffi::Any res = callback(node_addr, depth, hits, step, prio);
    return res.cast<bool>();
  }));
}

static int64_t prefix_cache_evict_by_named(prefix_cache_ref self, int64_t tier_int, tvm::ffi::String predicate, int64_t arg) {
  auto& pc = *self.get()->impl;
  return static_cast<int64_t>(
      pc.evict_by_named_predicate(int_to_tier(tier_int), std::string(predicate.data(), predicate.size()), arg));
}

static int64_t prefix_cache_available(prefix_cache_ref self, int64_t tier_int) {
  return static_cast<int64_t>(self.get()->impl->available(int_to_tier(tier_int)));
}
static int64_t prefix_cache_total(prefix_cache_ref self, int64_t tier_int) {
  return static_cast<int64_t>(self.get()->impl->total(int_to_tier(tier_int)));
}
static int64_t prefix_cache_active(prefix_cache_ref self, int64_t tier_int) {
  return static_cast<int64_t>(self.get()->impl->active(int_to_tier(tier_int)));
}
static int64_t prefix_cache_pending_units(prefix_cache_ref self, int64_t tier_int) {
  return static_cast<int64_t>(self.get()->impl->pending_units(int_to_tier(tier_int)));
}
static int64_t prefix_cache_max_pending_units(prefix_cache_ref self, int64_t tier_int) {
  return static_cast<int64_t>(self.get()->impl->max_pending_units(int_to_tier(tier_int)));
}
static bool prefix_cache_tier_enabled(prefix_cache_ref self, int64_t tier_int) {
  return self.get()->impl->tier_enabled(int_to_tier(tier_int));
}
static int64_t prefix_cache_atom_bytes(prefix_cache_ref self) { return static_cast<int64_t>(self.get()->impl->atom_bytes()); }
static int64_t prefix_cache_atoms_per_unit(prefix_cache_ref self) {
  return static_cast<int64_t>(self.get()->impl->atoms_per_unit());
}
static tvm::ffi::String prefix_cache_policy_name(prefix_cache_ref self) {
  return tvm::ffi::String(self.get()->impl->policy_name());
}
static int64_t prefix_cache_slru_threshold(prefix_cache_ref self) { return self.get()->impl->slru_threshold(); }

static int64_t prefix_cache_start_demote(prefix_cache_ref self, int64_t node_handle, int64_t src, int64_t dst) {
  auto* node = reinterpret_cast<tree_node*>(node_handle);
  return static_cast<int64_t>(self.get()->impl->start_demote(node, int_to_tier(src), int_to_tier(dst)));
}
static int64_t prefix_cache_start_promote(prefix_cache_ref self, int64_t node_handle, int64_t src, int64_t dst) {
  auto* node = reinterpret_cast<tree_node*>(node_handle);
  return static_cast<int64_t>(self.get()->impl->start_promote(node, int_to_tier(src), int_to_tier(dst)));
}
static void prefix_cache_complete_op(prefix_cache_ref self, int64_t handle, bool success) {
  self.get()->impl->complete_op(static_cast<op_handle>(handle), success);
}
static bool prefix_cache_wait_op(prefix_cache_ref self, int64_t handle, int64_t timeout_ms) {
  return self.get()->impl->wait_op(static_cast<op_handle>(handle), timeout_ms);
}
static int64_t prefix_cache_fail_all_inflight(prefix_cache_ref self) {
  return static_cast<int64_t>(self.get()->impl->fail_all_inflight());
}
static tvm::ffi::Array<int64_t> prefix_cache_inflight_ops(prefix_cache_ref self) {
  std::vector<int64_t> v;
  for (auto h : self.get()->impl->inflight_ops()) v.push_back(static_cast<int64_t>(h));
  return tvm::ffi::Array<int64_t>(v);
}

static int64_t prefix_cache_add_observer(prefix_cache_ref self, tvm::ffi::Function callback) {
  auto& pc = *self.get()->impl;
  return static_cast<int64_t>(pc.add_evict_observer([callback](tree_node* node, ::phyai_ext::radix_cache::tier t) mutable {
    callback(reinterpret_cast<int64_t>(node), static_cast<int64_t>(t));
  }));
}
static bool prefix_cache_remove_observer(prefix_cache_ref self, int64_t observer_id) {
  return self.get()->impl->remove_evict_observer(static_cast<std::size_t>(observer_id));
}

static tvm::ffi::Array<cache_event_ref> prefix_cache_take_events(prefix_cache_ref self) {
  std::vector<cache_event_ref> v;
  auto evs = self.get()->impl->take_events();
  v.reserve(evs.size());
  for (auto& e : evs) v.emplace_back(cache_event_ref(e));
  return tvm::ffi::Array<cache_event_ref>(v);
}
static int64_t prefix_cache_dropped_events(prefix_cache_ref self) {
  return static_cast<int64_t>(self.get()->impl->dropped_events());
}

static int64_t prefix_cache_current_step(prefix_cache_ref self) {
  return static_cast<int64_t>(self.get()->impl->current_step());
}
static void prefix_cache_advance_step(prefix_cache_ref self, int64_t n) {
  self.get()->impl->advance_step(static_cast<uint64_t>(n));
}
static void prefix_cache_touch_step(prefix_cache_ref self, int64_t node_handle) {
  auto* node = reinterpret_cast<tree_node*>(node_handle);
  self.get()->impl->touch_step(node);
}
static int64_t prefix_cache_node_path_hash(prefix_cache_ref self, int64_t node_handle) {
  auto* node = reinterpret_cast<const tree_node*>(node_handle);
  return static_cast<int64_t>(self.get()->impl->node_path_hash(node));
}

// ---------------------------------------------------------------------------
// hybrid_prefix_cache methods
// ---------------------------------------------------------------------------

static tvm::ffi::Array<tvm::ffi::Any> hybrid_match(hybrid_prefix_cache_ref self, tvm::ffi::Bytes atoms) {
  auto& hc = *self.get()->impl;
  auto m = hc.match(bytes_to_atom_span(atoms));
  std::vector<int64_t> last_nodes_v;
  std::vector<int64_t> matched_atoms_v;
  for (size_t t = 0; t < max_tiers; ++t) {
    last_nodes_v.push_back(reinterpret_cast<int64_t>(m.kv.last_node[t]));
    matched_atoms_v.push_back(static_cast<int64_t>(m.kv.matched_atoms[t]));
  }
  tvm::ffi::Array<tvm::ffi::Any> kv_pkg;
  kv_pkg.push_back(tvm::ffi::Any(tvm::ffi::Array<int64_t>(last_nodes_v)));
  kv_pkg.push_back(tvm::ffi::Any(tvm::ffi::Array<int64_t>(matched_atoms_v)));
  tvm::ffi::Array<tvm::ffi::Any> out;
  out.push_back(tvm::ffi::Any(kv_pkg));
  out.push_back(tvm::ffi::Any(reinterpret_cast<int64_t>(m.last_mamba_node)));
  out.push_back(tvm::ffi::Any(static_cast<int64_t>(m.mamba_branching_atoms)));
  return out;
}

static prefix_cache_ref hybrid_kv(hybrid_prefix_cache_ref self) { return self.get()->kv_ref; }

static tvm::ffi::Optional<mamba_slot_ref> hybrid_allocate_slot(hybrid_prefix_cache_ref self) {
  auto opt = self.get()->impl->allocate_mamba_slot();
  if (!opt.has_value()) return tvm::ffi::Optional<mamba_slot_ref>(std::nullopt);
  return tvm::ffi::Optional<mamba_slot_ref>(mamba_slot_ref(std::move(*opt)));
}

static void hybrid_attach_mamba(hybrid_prefix_cache_ref self, int64_t node_handle, mamba_slot_ref slot) {
  auto* node = reinterpret_cast<tree_node*>(node_handle);
  self.get()->impl->attach_mamba(node, std::move(slot.get()->impl));
}

static tvm::ffi::Optional<mamba_slot_ref> hybrid_detach_mamba(hybrid_prefix_cache_ref self, int64_t node_handle) {
  auto* node = reinterpret_cast<tree_node*>(node_handle);
  auto opt = self.get()->impl->detach_mamba(node);
  if (!opt.has_value()) return tvm::ffi::Optional<mamba_slot_ref>(std::nullopt);
  return tvm::ffi::Optional<mamba_slot_ref>(mamba_slot_ref(std::move(*opt)));
}

static int64_t hybrid_find_last_mamba_node(hybrid_prefix_cache_ref self, int64_t from_handle) {
  auto* from = reinterpret_cast<tree_node*>(from_handle);
  return reinterpret_cast<int64_t>(self.get()->impl->find_last_mamba_node(from));
}

static bool hybrid_ensure_capacity(hybrid_prefix_cache_ref self, int64_t n) {
  return self.get()->impl->ensure_mamba_capacity_by_evict(static_cast<int32_t>(n));
}

static int64_t hybrid_available_slots(hybrid_prefix_cache_ref self) {
  return static_cast<int64_t>(self.get()->impl->available_slots());
}
static int64_t hybrid_total_slots(hybrid_prefix_cache_ref self) {
  return static_cast<int64_t>(self.get()->impl->total_slots());
}
static int64_t hybrid_active_slots(hybrid_prefix_cache_ref self) {
  return static_cast<int64_t>(self.get()->impl->active_slots());
}

// ---------------------------------------------------------------------------
// owned_units methods
// ---------------------------------------------------------------------------

static int64_t owned_units_tier(owned_units_ref self) { return static_cast<int64_t>(self.get()->impl.get_tier()); }
static int64_t owned_units_size(owned_units_ref self) { return static_cast<int64_t>(self.get()->impl.size()); }
static tvm::ffi::Array<int64_t> owned_units_ids(owned_units_ref self) { return ids_to_array(self.get()->impl.ids()); }
static owned_units_ref owned_units_take_first(owned_units_ref self, int64_t n) {
  return owned_units_ref(self.get()->impl.take_first(static_cast<size_t>(n)));
}
static owned_units_ref owned_units_take_last(owned_units_ref self, int64_t n) {
  return owned_units_ref(self.get()->impl.take_last(static_cast<size_t>(n)));
}
static void owned_units_append(owned_units_ref self, owned_units_ref other) {
  self.get()->impl.append(std::move(other.get()->impl));
}

// ---------------------------------------------------------------------------
// mamba_slot, node_ref, composite_node_ref methods
// ---------------------------------------------------------------------------

static int64_t mamba_slot_index(mamba_slot_ref self) { return static_cast<int64_t>(self.get()->impl.index()); }
static bool mamba_slot_valid(mamba_slot_ref self) { return self.get()->impl.valid(); }

static int64_t node_ref_node(node_ref_ref self) { return reinterpret_cast<int64_t>(self.get()->impl.node()); }
static int64_t node_ref_tier(node_ref_ref self) { return static_cast<int64_t>(self.get()->impl.get_tier()); }
static bool node_ref_valid(node_ref_ref self) { return self.get()->impl.valid(); }

static int64_t composite_node_ref_size(composite_node_ref_ref self) { return static_cast<int64_t>(self.get()->impl.size()); }
static bool composite_node_ref_valid(composite_node_ref_ref self) { return self.get()->impl.valid(); }
static int64_t composite_node_ref_node(composite_node_ref_ref self) {
  return reinterpret_cast<int64_t>(self.get()->impl.node());
}

// ---------------------------------------------------------------------------
// tree_node accessors (telemetry / debugging)
// ---------------------------------------------------------------------------

static int64_t tree_node_depth_in_atoms(int64_t node_handle) {
  auto* n = reinterpret_cast<tree_node*>(node_handle);
  return n == nullptr ? 0 : static_cast<int64_t>(n->depth_in_atoms());
}
static int64_t tree_node_atom_count(int64_t node_handle) {
  auto* n = reinterpret_cast<tree_node*>(node_handle);
  return n == nullptr ? 0 : static_cast<int64_t>(n->atom_count());
}
static int64_t tree_node_hit_count(int64_t node_handle) {
  auto* n = reinterpret_cast<tree_node*>(node_handle);
  return n == nullptr ? 0 : static_cast<int64_t>(n->hit_count());
}
static int64_t tree_node_last_access_step(int64_t node_handle) {
  auto* n = reinterpret_cast<tree_node*>(node_handle);
  return n == nullptr ? 0 : static_cast<int64_t>(n->last_access_step());
}
static void tree_node_set_user_priority(int64_t node_handle, int64_t prio) {
  auto* n = reinterpret_cast<tree_node*>(node_handle);
  if (n != nullptr) n->set_user_priority(prio);
}
static int64_t tree_node_user_priority(int64_t node_handle) {
  auto* n = reinterpret_cast<tree_node*>(node_handle);
  return n == nullptr ? 0 : n->user_priority();
}
static bool tree_node_has_mamba(int64_t node_handle) {
  auto* n = reinterpret_cast<tree_node*>(node_handle);
  return n == nullptr ? false : n->has_mamba();
}
static int64_t tree_node_mamba_index(int64_t node_handle) {
  auto* n = reinterpret_cast<tree_node*>(node_handle);
  return n == nullptr ? -1 : static_cast<int64_t>(n->mamba_index());
}

// ---------------------------------------------------------------------------
// storage_backend constructors and methods
// ---------------------------------------------------------------------------

static storage_backend_ref storage_backend_in_memory(int64_t unit_bytes) {
  return storage_backend_ref(std::make_shared<in_memory_storage_backend>(static_cast<std::size_t>(unit_bytes)));
}
static storage_backend_ref storage_backend_file(tvm::ffi::String path, int64_t unit_bytes) {
  return storage_backend_ref(
      std::make_shared<file_storage_backend>(std::string(path.data(), path.size()), static_cast<std::size_t>(unit_bytes)));
}
static tvm::ffi::String storage_backend_name(storage_backend_ref self) { return tvm::ffi::String(self.get()->impl->name()); }
static int64_t storage_backend_unit_bytes(storage_backend_ref self) {
  return static_cast<int64_t>(self.get()->impl->unit_bytes());
}
static void storage_backend_drain(storage_backend_ref self) { self.get()->impl->drain(); }
static bool storage_backend_in_memory_contains(storage_backend_ref self, int64_t key) {
  auto* mem = dynamic_cast<in_memory_storage_backend*>(self.get()->impl.get());
  return mem != nullptr && mem->contains(static_cast<std::uint64_t>(key));
}
static int64_t storage_backend_in_memory_entries(storage_backend_ref self) {
  auto* mem = dynamic_cast<in_memory_storage_backend*>(self.get()->impl.get());
  return mem == nullptr ? 0 : static_cast<int64_t>(mem->entries());
}

// Synchronous round-trip helpers to drive the backend from Python without
// pulling a real I/O loop. Real deployments would use start_write/start_read
// asynchronously; tests use these.
static bool storage_backend_write_sync(storage_backend_ref self, int64_t op_hdl, int64_t key, tvm::ffi::Array<int64_t> ids) {
  std::vector<unit_id> tmp;
  tmp.reserve(ids.size());
  for (auto v : ids) tmp.push_back(static_cast<unit_id>(v));
  bool ok = false;
  self.get()->impl->start_write(static_cast<op_handle>(op_hdl), static_cast<std::uint64_t>(key),
                                std::span<const unit_id>(tmp.data(), tmp.size()),
                                [&](op_handle, bool success) { ok = success; });
  return ok;
}
static bool storage_backend_read_sync(storage_backend_ref self, int64_t op_hdl, int64_t key, tvm::ffi::Array<int64_t> ids) {
  std::vector<unit_id> tmp;
  tmp.reserve(ids.size());
  for (auto v : ids) tmp.push_back(static_cast<unit_id>(v));
  bool ok = false;
  self.get()->impl->start_read(static_cast<op_handle>(op_hdl), static_cast<std::uint64_t>(key),
                               std::span<const unit_id>(tmp.data(), tmp.size()),
                               [&](op_handle, bool success) { ok = success; });
  return ok;
}

// ---------------------------------------------------------------------------
// xxh3 utility (for callers wanting a path hash without going through the
// node_path_hash method)
// ---------------------------------------------------------------------------

static int64_t xxh3_64_bytes(tvm::ffi::Bytes b) { return static_cast<int64_t>(xxh3_64(b.data(), b.size())); }

}  // namespace phyai_ext::radix_cache::ffi

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  using namespace phyai_ext::radix_cache::ffi;

  refl::ObjectDef<owned_units_obj>();
  refl::ObjectDef<node_ref_obj>();
  refl::ObjectDef<composite_node_ref_obj>();
  refl::ObjectDef<mamba_slot_obj>();
  refl::ObjectDef<cache_event_obj>()
      .def_ro("kind", &cache_event_obj::kind)
      .def_ro("tier_from", &cache_event_obj::tier_from)
      .def_ro("tier_to", &cache_event_obj::tier_to)
      .def_ro("handle", &cache_event_obj::handle)
      .def_ro("atom_count", &cache_event_obj::atom_count)
      .def_ro("unit_count", &cache_event_obj::unit_count)
      .def_ro("ts_ns", &cache_event_obj::ts_ns);
  refl::ObjectDef<prefix_cache_obj>();
  refl::ObjectDef<hybrid_prefix_cache_obj>();
  refl::ObjectDef<storage_backend_obj>();

  refl::GlobalDef()
      .def("phyai_ext.radix_cache.prefix_cache.create", prefix_cache_create)
      .def("phyai_ext.radix_cache.prefix_cache.match", prefix_cache_match)
      .def("phyai_ext.radix_cache.prefix_cache.collect_units", prefix_cache_collect_units)
      .def("phyai_ext.radix_cache.prefix_cache.insert", prefix_cache_insert)
      .def("phyai_ext.radix_cache.prefix_cache.allocate", prefix_cache_allocate)
      .def("phyai_ext.radix_cache.prefix_cache.lock", prefix_cache_lock)
      .def("phyai_ext.radix_cache.prefix_cache.lock_multi", prefix_cache_lock_multi)
      .def("phyai_ext.radix_cache.prefix_cache.ensure_capacity", prefix_cache_ensure_capacity)
      .def("phyai_ext.radix_cache.prefix_cache.evict_by_predicate", prefix_cache_evict_by_predicate)
      .def("phyai_ext.radix_cache.prefix_cache.evict_by_named_predicate", prefix_cache_evict_by_named)
      .def("phyai_ext.radix_cache.prefix_cache.available", prefix_cache_available)
      .def("phyai_ext.radix_cache.prefix_cache.total", prefix_cache_total)
      .def("phyai_ext.radix_cache.prefix_cache.active", prefix_cache_active)
      .def("phyai_ext.radix_cache.prefix_cache.pending_units", prefix_cache_pending_units)
      .def("phyai_ext.radix_cache.prefix_cache.max_pending_units", prefix_cache_max_pending_units)
      .def("phyai_ext.radix_cache.prefix_cache.tier_enabled", prefix_cache_tier_enabled)
      .def("phyai_ext.radix_cache.prefix_cache.atom_bytes", prefix_cache_atom_bytes)
      .def("phyai_ext.radix_cache.prefix_cache.atoms_per_unit", prefix_cache_atoms_per_unit)
      .def("phyai_ext.radix_cache.prefix_cache.policy_name", prefix_cache_policy_name)
      .def("phyai_ext.radix_cache.prefix_cache.slru_threshold", prefix_cache_slru_threshold)
      .def("phyai_ext.radix_cache.prefix_cache.start_demote", prefix_cache_start_demote)
      .def("phyai_ext.radix_cache.prefix_cache.start_promote", prefix_cache_start_promote)
      .def("phyai_ext.radix_cache.prefix_cache.complete_op", prefix_cache_complete_op)
      .def("phyai_ext.radix_cache.prefix_cache.wait_op", prefix_cache_wait_op)
      .def("phyai_ext.radix_cache.prefix_cache.fail_all_inflight", prefix_cache_fail_all_inflight)
      .def("phyai_ext.radix_cache.prefix_cache.inflight_ops", prefix_cache_inflight_ops)
      .def("phyai_ext.radix_cache.prefix_cache.add_evict_observer", prefix_cache_add_observer)
      .def("phyai_ext.radix_cache.prefix_cache.remove_evict_observer", prefix_cache_remove_observer)
      .def("phyai_ext.radix_cache.prefix_cache.take_events", prefix_cache_take_events)
      .def("phyai_ext.radix_cache.prefix_cache.dropped_events", prefix_cache_dropped_events)
      .def("phyai_ext.radix_cache.prefix_cache.current_step", prefix_cache_current_step)
      .def("phyai_ext.radix_cache.prefix_cache.advance_step", prefix_cache_advance_step)
      .def("phyai_ext.radix_cache.prefix_cache.touch_step", prefix_cache_touch_step)
      .def("phyai_ext.radix_cache.prefix_cache.node_path_hash", prefix_cache_node_path_hash);

  refl::GlobalDef()
      .def("phyai_ext.radix_cache.hybrid_prefix_cache.create", hybrid_prefix_cache_create)
      .def("phyai_ext.radix_cache.hybrid_prefix_cache.match", hybrid_match)
      .def("phyai_ext.radix_cache.hybrid_prefix_cache.kv", hybrid_kv)
      .def("phyai_ext.radix_cache.hybrid_prefix_cache.allocate_mamba_slot", hybrid_allocate_slot)
      .def("phyai_ext.radix_cache.hybrid_prefix_cache.attach_mamba", hybrid_attach_mamba)
      .def("phyai_ext.radix_cache.hybrid_prefix_cache.detach_mamba", hybrid_detach_mamba)
      .def("phyai_ext.radix_cache.hybrid_prefix_cache.find_last_mamba_node", hybrid_find_last_mamba_node)
      .def("phyai_ext.radix_cache.hybrid_prefix_cache.ensure_mamba_capacity_by_evict", hybrid_ensure_capacity)
      .def("phyai_ext.radix_cache.hybrid_prefix_cache.available_slots", hybrid_available_slots)
      .def("phyai_ext.radix_cache.hybrid_prefix_cache.total_slots", hybrid_total_slots)
      .def("phyai_ext.radix_cache.hybrid_prefix_cache.active_slots", hybrid_active_slots);

  refl::GlobalDef()
      .def("phyai_ext.radix_cache.owned_units.tier", owned_units_tier)
      .def("phyai_ext.radix_cache.owned_units.size", owned_units_size)
      .def("phyai_ext.radix_cache.owned_units.ids", owned_units_ids)
      .def("phyai_ext.radix_cache.owned_units.take_first", owned_units_take_first)
      .def("phyai_ext.radix_cache.owned_units.take_last", owned_units_take_last)
      .def("phyai_ext.radix_cache.owned_units.append", owned_units_append);

  refl::GlobalDef()
      .def("phyai_ext.radix_cache.mamba_slot.index", mamba_slot_index)
      .def("phyai_ext.radix_cache.mamba_slot.valid", mamba_slot_valid);

  refl::GlobalDef()
      .def("phyai_ext.radix_cache.node_ref.node", node_ref_node)
      .def("phyai_ext.radix_cache.node_ref.tier", node_ref_tier)
      .def("phyai_ext.radix_cache.node_ref.valid", node_ref_valid);

  refl::GlobalDef()
      .def("phyai_ext.radix_cache.composite_node_ref.size", composite_node_ref_size)
      .def("phyai_ext.radix_cache.composite_node_ref.valid", composite_node_ref_valid)
      .def("phyai_ext.radix_cache.composite_node_ref.node", composite_node_ref_node);

  refl::GlobalDef()
      .def("phyai_ext.radix_cache.tree_node.depth_in_atoms", tree_node_depth_in_atoms)
      .def("phyai_ext.radix_cache.tree_node.atom_count", tree_node_atom_count)
      .def("phyai_ext.radix_cache.tree_node.hit_count", tree_node_hit_count)
      .def("phyai_ext.radix_cache.tree_node.last_access_step", tree_node_last_access_step)
      .def("phyai_ext.radix_cache.tree_node.user_priority", tree_node_user_priority)
      .def("phyai_ext.radix_cache.tree_node.set_user_priority", tree_node_set_user_priority)
      .def("phyai_ext.radix_cache.tree_node.has_mamba", tree_node_has_mamba)
      .def("phyai_ext.radix_cache.tree_node.mamba_index", tree_node_mamba_index);

  refl::GlobalDef()
      .def("phyai_ext.radix_cache.storage_backend.in_memory", storage_backend_in_memory)
      .def("phyai_ext.radix_cache.storage_backend.file", storage_backend_file)
      .def("phyai_ext.radix_cache.storage_backend.name", storage_backend_name)
      .def("phyai_ext.radix_cache.storage_backend.unit_bytes", storage_backend_unit_bytes)
      .def("phyai_ext.radix_cache.storage_backend.drain", storage_backend_drain)
      .def("phyai_ext.radix_cache.storage_backend.in_memory_contains", storage_backend_in_memory_contains)
      .def("phyai_ext.radix_cache.storage_backend.in_memory_entries", storage_backend_in_memory_entries)
      .def("phyai_ext.radix_cache.storage_backend.write_sync", storage_backend_write_sync)
      .def("phyai_ext.radix_cache.storage_backend.read_sync", storage_backend_read_sync);

  refl::GlobalDef().def("phyai_ext.radix_cache.xxh3_64", xxh3_64_bytes);
}
