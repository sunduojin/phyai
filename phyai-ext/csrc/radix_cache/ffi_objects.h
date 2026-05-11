// FFI binding wrappers for prefix_cache, hybrid_prefix_cache, owned_units,
// node_ref, mamba_slot, cache_event and storage_backend. Each wrapper is a
// tvm::ffi::Object subclass; the actual logic lives in the STL-style classes
// in the radix_cache directory.
#pragma once

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "radix_cache/events.h"
#include "radix_cache/mamba/hybrid_prefix_cache.h"
#include "radix_cache/mamba/mamba_slot.h"
#include "radix_cache/owned_units.h"
#include "radix_cache/prefix_cache.h"
#include "radix_cache/storage_backend.h"
#include "radix_cache/tree/node_ref.h"

namespace phyai_ext::radix_cache::ffi {

class owned_units_obj : public tvm::ffi::Object {
 public:
  owned_units_obj() = default;
  explicit owned_units_obj(owned_units&& u) : impl(std::move(u)) {}
  owned_units impl;

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("phyai_ext.radix_cache.owned_units", owned_units_obj, tvm::ffi::Object);
};

class owned_units_ref : public tvm::ffi::ObjectRef {
 public:
  explicit owned_units_ref(owned_units&& u) { data_ = tvm::ffi::make_object<owned_units_obj>(std::move(u)); }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(owned_units_ref, tvm::ffi::ObjectRef, owned_units_obj);
};

class node_ref_obj : public tvm::ffi::Object {
 public:
  node_ref_obj() = default;
  explicit node_ref_obj(node_ref&& r) : impl(std::move(r)) {}
  node_ref impl;

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("phyai_ext.radix_cache.node_ref", node_ref_obj, tvm::ffi::Object);
};

class node_ref_ref : public tvm::ffi::ObjectRef {
 public:
  explicit node_ref_ref(node_ref&& r) { data_ = tvm::ffi::make_object<node_ref_obj>(std::move(r)); }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(node_ref_ref, tvm::ffi::ObjectRef, node_ref_obj);
};

class composite_node_ref_obj : public tvm::ffi::Object {
 public:
  composite_node_ref_obj() = default;
  explicit composite_node_ref_obj(composite_node_ref&& r) : impl(std::move(r)) {}
  composite_node_ref impl;

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("phyai_ext.radix_cache.composite_node_ref", composite_node_ref_obj, tvm::ffi::Object);
};

class composite_node_ref_ref : public tvm::ffi::ObjectRef {
 public:
  explicit composite_node_ref_ref(composite_node_ref&& r) {
    data_ = tvm::ffi::make_object<composite_node_ref_obj>(std::move(r));
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(composite_node_ref_ref, tvm::ffi::ObjectRef, composite_node_ref_obj);
};

class mamba_slot_obj : public tvm::ffi::Object {
 public:
  mamba_slot_obj() = default;
  explicit mamba_slot_obj(mamba_slot&& s) : impl(std::move(s)) {}
  mamba_slot impl;

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("phyai_ext.radix_cache.mamba_slot", mamba_slot_obj, tvm::ffi::Object);
};

class mamba_slot_ref : public tvm::ffi::ObjectRef {
 public:
  explicit mamba_slot_ref(mamba_slot&& s) { data_ = tvm::ffi::make_object<mamba_slot_obj>(std::move(s)); }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(mamba_slot_ref, tvm::ffi::ObjectRef, mamba_slot_obj);
};

class cache_event_obj : public tvm::ffi::Object {
 public:
  cache_event_obj() = default;
  explicit cache_event_obj(cache_event ev)
      : kind(static_cast<int64_t>(ev.kind)),
        tier_from(static_cast<int64_t>(ev.tier_from)),
        tier_to(static_cast<int64_t>(ev.tier_to)),
        handle(static_cast<int64_t>(ev.handle)),
        atom_count(static_cast<int64_t>(ev.atom_count)),
        unit_count(static_cast<int64_t>(ev.unit_count)),
        ts_ns(static_cast<int64_t>(ev.ts_ns)) {}

  int64_t kind = 0;
  int64_t tier_from = 0;
  int64_t tier_to = 0;
  int64_t handle = 0;
  int64_t atom_count = 0;
  int64_t unit_count = 0;
  int64_t ts_ns = 0;

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("phyai_ext.radix_cache.cache_event", cache_event_obj, tvm::ffi::Object);
};

class cache_event_ref : public tvm::ffi::ObjectRef {
 public:
  explicit cache_event_ref(cache_event ev) { data_ = tvm::ffi::make_object<cache_event_obj>(ev); }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(cache_event_ref, tvm::ffi::ObjectRef, cache_event_obj);
};

class prefix_cache_obj : public tvm::ffi::Object {
 public:
  std::shared_ptr<prefix_cache> impl;

  prefix_cache_obj() = default;
  explicit prefix_cache_obj(std::shared_ptr<prefix_cache> p) : impl(std::move(p)) {}

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("phyai_ext.radix_cache.prefix_cache", prefix_cache_obj, tvm::ffi::Object);
};

class prefix_cache_ref : public tvm::ffi::ObjectRef {
 public:
  explicit prefix_cache_ref(std::shared_ptr<prefix_cache> p) { data_ = tvm::ffi::make_object<prefix_cache_obj>(std::move(p)); }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(prefix_cache_ref, tvm::ffi::ObjectRef, prefix_cache_obj);
};

class hybrid_prefix_cache_obj : public tvm::ffi::Object {
 public:
  std::shared_ptr<hybrid_prefix_cache> impl;
  // Hold a strong reference to the underlying prefix_cache_obj so Python
  // doesn't drop it.
  prefix_cache_ref kv_ref;

  hybrid_prefix_cache_obj() = default;
  hybrid_prefix_cache_obj(std::shared_ptr<hybrid_prefix_cache> p, prefix_cache_ref kv)
      : impl(std::move(p)), kv_ref(std::move(kv)) {}

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("phyai_ext.radix_cache.hybrid_prefix_cache", hybrid_prefix_cache_obj, tvm::ffi::Object);
};

class hybrid_prefix_cache_ref : public tvm::ffi::ObjectRef {
 public:
  hybrid_prefix_cache_ref(std::shared_ptr<hybrid_prefix_cache> p, prefix_cache_ref kv) {
    data_ = tvm::ffi::make_object<hybrid_prefix_cache_obj>(std::move(p), std::move(kv));
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(hybrid_prefix_cache_ref, tvm::ffi::ObjectRef, hybrid_prefix_cache_obj);
};

class storage_backend_obj : public tvm::ffi::Object {
 public:
  std::shared_ptr<storage_backend> impl;

  storage_backend_obj() = default;
  explicit storage_backend_obj(std::shared_ptr<storage_backend> p) : impl(std::move(p)) {}

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("phyai_ext.radix_cache.storage_backend", storage_backend_obj, tvm::ffi::Object);
};

class storage_backend_ref : public tvm::ffi::ObjectRef {
 public:
  explicit storage_backend_ref(std::shared_ptr<storage_backend> p) {
    data_ = tvm::ffi::make_object<storage_backend_obj>(std::move(p));
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(storage_backend_ref, tvm::ffi::ObjectRef, storage_backend_obj);
};

}  // namespace phyai_ext::radix_cache::ffi
