#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "pih/core/result.h"

namespace pih {

class QwenInt4CanonicalTensor final {
 public:
  static constexpr std::uint64_t kGroupSize = 128;
  static constexpr std::uint64_t kMaximumElements = UINT64_C(1) << 28;

  static Result<QwenInt4CanonicalTensor> Verify(
      std::span<const std::byte> packed_values,
      std::span<const std::uint16_t> scale_bits, std::uint64_t rows,
      std::uint64_t columns);

  Result<std::vector<float>> dequantize() const;

  [[nodiscard]] std::uint64_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::uint64_t columns() const noexcept { return columns_; }
  [[nodiscard]] std::uint64_t groups_per_row() const noexcept {
    return groups_per_row_;
  }
  [[nodiscard]] std::span<const std::byte> packed_values() const noexcept {
    return packed_values_;
  }
  [[nodiscard]] std::span<const std::uint16_t> scale_bits() const noexcept {
    return scale_bits_;
  }

 private:
  QwenInt4CanonicalTensor(std::uint64_t rows, std::uint64_t columns,
                          std::uint64_t groups_per_row,
                          std::vector<std::byte> packed_values,
                          std::vector<std::uint16_t> scale_bits)
      : rows_(rows), columns_(columns), groups_per_row_(groups_per_row),
        packed_values_(std::move(packed_values)),
        scale_bits_(std::move(scale_bits)) {}

  std::int8_t value(std::uint64_t row, std::uint64_t column) const noexcept;

  std::uint64_t rows_;
  std::uint64_t columns_;
  std::uint64_t groups_per_row_;
  std::vector<std::byte> packed_values_;
  std::vector<std::uint16_t> scale_bits_;
};

}  // namespace pih
