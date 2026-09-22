#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pih/core/dtype.h"

namespace pih {

struct SafetensorRecord final {
  std::string name;
  DType dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t file_begin;
  std::uint64_t file_end;
};

class SafetensorsHeader final {
 public:
  static constexpr std::uint64_t kMaxHeaderBytes = 16ULL * 1024 * 1024;
  static constexpr std::size_t kMaxTensorCount = 4096;
  static constexpr std::size_t kMaxTensorNameBytes = 512;
  static constexpr std::size_t kMaxRank = 8;

  static Result<SafetensorsHeader> ParsePrefix(
      std::span<const std::byte> prefix, std::uint64_t file_bytes);

  [[nodiscard]] const std::vector<SafetensorRecord>& tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] std::uint64_t header_bytes() const noexcept { return header_bytes_; }
  [[nodiscard]] std::uint64_t data_bytes() const noexcept { return data_bytes_; }
  [[nodiscard]] const SafetensorRecord* tensor(std::string_view name) const noexcept;

 private:
  SafetensorsHeader(std::uint64_t header_bytes, std::uint64_t data_bytes,
                    std::vector<SafetensorRecord> tensors)
      : header_bytes_(header_bytes), data_bytes_(data_bytes),
        tensors_(std::move(tensors)) {}

  std::uint64_t header_bytes_;
  std::uint64_t data_bytes_;
  std::vector<SafetensorRecord> tensors_;
};

}  // namespace pih
