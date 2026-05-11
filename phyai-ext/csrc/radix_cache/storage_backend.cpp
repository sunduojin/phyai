#include "radix_cache/storage_backend.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace phyai_ext::radix_cache {

void in_memory_storage_backend::start_write(op_handle handle, std::uint64_t key, std::span<const unit_id> ids,
                                            completion_callback cb) {
  {
    std::lock_guard<std::mutex> g(mtx_);
    entry e;
    e.ids.assign(ids.begin(), ids.end());
    store_[key] = std::move(e);
  }
  cb(handle, true);
}

void in_memory_storage_backend::start_read(op_handle handle, std::uint64_t key, std::span<const unit_id> /*ids*/,
                                           completion_callback cb) {
  bool present = false;
  {
    std::lock_guard<std::mutex> g(mtx_);
    present = store_.find(key) != store_.end();
  }
  cb(handle, present);
}

bool in_memory_storage_backend::contains(std::uint64_t key) const {
  std::lock_guard<std::mutex> g(mtx_);
  return store_.find(key) != store_.end();
}

std::size_t in_memory_storage_backend::entries() const {
  std::lock_guard<std::mutex> g(mtx_);
  return store_.size();
}

// ---------------------------------------------------------------------------
// file_storage_backend
// ---------------------------------------------------------------------------

file_storage_backend::file_storage_backend(std::string path, std::size_t unit_bytes)
    : path_(std::move(path)), unit_bytes_(unit_bytes) {
  fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd_ < 0) { throw std::runtime_error("file_storage_backend: cannot open " + path_); }
  end_offset_ = ::lseek(fd_, 0, SEEK_END);
  if (end_offset_ < 0) end_offset_ = 0;
}

file_storage_backend::~file_storage_backend() {
  if (fd_ >= 0) ::close(fd_);
}

void file_storage_backend::start_write(op_handle handle, std::uint64_t key, std::span<const unit_id> ids,
                                       completion_callback cb) {
  if (unit_bytes_ == 0) {
    cb(handle, false);
    return;
  }
  const std::size_t bytes = ids.size() * unit_bytes_;
  std::vector<std::uint8_t> buf(bytes, 0);
  // Encode the unit-id list deterministically into the leading bytes so a
  // matching read can verify content-addressing without external state.
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (unit_bytes_ >= sizeof(unit_id)) { std::memcpy(buf.data() + i * unit_bytes_, &ids[i], sizeof(unit_id)); }
  }
  std::int64_t off;
  {
    std::lock_guard<std::mutex> g(mtx_);
    off = end_offset_;
    end_offset_ += static_cast<std::int64_t>(bytes);
    offsets_[key] = off;
  }
  ssize_t w = ::pwrite(fd_, buf.data(), buf.size(), off);
  cb(handle, w == static_cast<ssize_t>(buf.size()));
}

void file_storage_backend::start_read(op_handle handle, std::uint64_t key, std::span<const unit_id> ids,
                                      completion_callback cb) {
  std::int64_t off = -1;
  {
    std::lock_guard<std::mutex> g(mtx_);
    auto it = offsets_.find(key);
    if (it == offsets_.end()) {
      cb(handle, false);
      return;
    }
    off = it->second;
  }
  if (unit_bytes_ == 0) {
    cb(handle, true);
    return;
  }
  std::vector<std::uint8_t> buf(ids.size() * unit_bytes_, 0);
  ssize_t r = ::pread(fd_, buf.data(), buf.size(), off);
  cb(handle, r == static_cast<ssize_t>(buf.size()));
}

void file_storage_backend::drain() {
  if (fd_ >= 0) ::fdatasync(fd_);
}

}  // namespace phyai_ext::radix_cache
