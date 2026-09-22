#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "pih/core/result.h"
#include "pih/io/controller_file_lease.h"

namespace pih {

class DescriptorMappedRange final {
 public:
  static Result<DescriptorMappedRange> MapReadOnly(
      const ArtifactWorkerDescriptor& descriptor, std::uint64_t file_offset,
      std::uint64_t bytes);

  ~DescriptorMappedRange();
  DescriptorMappedRange(const DescriptorMappedRange&) = delete;
  DescriptorMappedRange& operator=(const DescriptorMappedRange&) = delete;
  DescriptorMappedRange(DescriptorMappedRange&&) noexcept;
  DescriptorMappedRange& operator=(DescriptorMappedRange&&) noexcept;

  [[nodiscard]] const std::byte* data() const noexcept { return data_; }
  [[nodiscard]] std::uint64_t file_offset() const noexcept {
    return file_offset_;
  }
  [[nodiscard]] std::uint64_t size_bytes() const noexcept { return size_bytes_; }
  Result<std::span<const std::byte>> slice(std::uint64_t offset,
                                          std::uint64_t bytes) const;

 private:
  struct Impl;
  DescriptorMappedRange(std::unique_ptr<Impl> impl, const std::byte* data,
                        std::uint64_t file_offset, std::uint64_t size_bytes);

  std::unique_ptr<Impl> impl_;
  const std::byte* data_ = nullptr;
  std::uint64_t file_offset_ = 0;
  std::uint64_t size_bytes_ = 0;
};

}  // namespace pih
