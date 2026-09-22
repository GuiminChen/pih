#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>

#include "pih/io/mapped_file.h"
#include "pih/io/controller_file_lease.h"
#include "pih/core/sha256.h"
#include "pih/model/safetensors_header.h"
#include "pih/model/safetensors_header_receipt.h"

namespace pih {

struct ImmutableTensorBytes final {
  DType dtype;
  std::span<const std::uint64_t> shape;
  std::span<const std::byte> bytes;
};

class SafetensorsFile final {
 public:
  static Result<SafetensorsFile> Open(const std::filesystem::path& path,
                                      std::uint64_t maximum_file_bytes);
  static Result<SafetensorsFile> Borrow(std::span<const std::byte> bytes);

  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;
  SafetensorsFile(SafetensorsFile&&) noexcept = default;
  SafetensorsFile& operator=(SafetensorsFile&&) noexcept = default;

  [[nodiscard]] const SafetensorsHeader& header() const noexcept { return header_; }
  [[nodiscard]] std::uint64_t size_bytes() const noexcept {
    return bytes_.size();
  }
  [[nodiscard]] std::span<const std::byte> file_bytes() const noexcept {
    return bytes_;
  }
  Result<ImmutableTensorBytes> tensor(std::string_view name) const;

 private:
  SafetensorsFile(MappedFile mapping, SafetensorsHeader header)
      : mapping_(std::move(mapping)),
        bytes_(mapping_->data(), static_cast<std::size_t>(mapping_->size_bytes())),
        header_(std::move(header)) {}
  SafetensorsFile(std::span<const std::byte> bytes, SafetensorsHeader header)
      : bytes_(bytes), header_(std::move(header)) {}

  std::optional<MappedFile> mapping_;
  std::span<const std::byte> bytes_;
  SafetensorsHeader header_;
};

// Reads only the length prefix and bounded JSON header. It never maps or
// touches tensor payload pages and is suitable for controller preflight.
Result<SafetensorsHeaderFileReceipt> load_safetensors_header_file(
    const std::filesystem::path& path, std::uint64_t maximum_file_bytes);

// Reads the controller's already-opened master descriptor. No pathname lookup
// occurs, so the receipt and subsequent worker duplicates share one identity.
Result<SafetensorsHeaderFileReceipt> load_safetensors_header_descriptor(
    const ControllerFileLease& lease);

}  // namespace pih
