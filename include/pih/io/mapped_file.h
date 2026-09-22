#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

#include "pih/core/result.h"

namespace pih {

class MappedFile final {
 public:
  static Result<MappedFile> OpenReadOnly(const std::filesystem::path& path,
                                         std::uint64_t maximum_bytes);
  ~MappedFile();

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;

  [[nodiscard]] const std::byte* data() const noexcept { return data_; }
  [[nodiscard]] std::uint64_t size_bytes() const noexcept { return size_bytes_; }
  Result<std::span<const std::byte>> slice(std::uint64_t offset,
                                          std::uint64_t bytes) const;

 private:
  struct Impl;
  MappedFile(std::shared_ptr<Impl> impl, const std::byte* data,
             std::uint64_t size_bytes)
      : impl_(std::move(impl)), data_(data), size_bytes_(size_bytes) {}

  std::shared_ptr<Impl> impl_;
  const std::byte* data_ = nullptr;
  std::uint64_t size_bytes_ = 0;
};

}  // namespace pih
