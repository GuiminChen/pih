#pragma once
// Private to native offline Qwen conversion; not a model/runtime SDK interface.

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "pih/core/bfloat16.h"
#include "pih/core/result.h"

namespace pih {

struct QwenInt4ArtifactPayload final {
  std::vector<std::byte> packed_values;
  std::vector<std::uint16_t> scale_bits;
};

class QwenInt4QuantizedTensor final {
 public:
  static constexpr std::uint64_t kGroupSize = 128;
  static constexpr std::uint64_t kMaximumElements = UINT64_C(1) << 28;

  static Result<QwenInt4QuantizedTensor> Quantize(
      std::span<const BFloat16> source, std::uint64_t rows,
      std::uint64_t columns);

  [[nodiscard]] std::uint64_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::uint64_t columns() const noexcept { return columns_; }
  [[nodiscard]] std::uint64_t groups_per_row() const noexcept {
    return groups_per_row_;
  }
  [[nodiscard]] std::span<const std::int8_t> logical_values() const noexcept {
    return logical_values_;
  }
  [[nodiscard]] std::span<const std::uint16_t> scale_bits() const noexcept {
    return scale_bits_;
  }
  [[nodiscard]] std::span<const std::byte> packed_values() const noexcept {
    return packed_values_;
  }
  QwenInt4ArtifactPayload release_artifact_payload() && noexcept {
    std::vector<std::int8_t>().swap(logical_values_);
    return {std::move(packed_values_), std::move(scale_bits_)};
  }

 private:
  QwenInt4QuantizedTensor(std::uint64_t rows, std::uint64_t columns,
                          std::uint64_t groups_per_row,
                          std::vector<std::int8_t> logical_values,
                          std::vector<std::uint16_t> scale_bits,
                          std::vector<std::byte> packed_values)
      : rows_(rows), columns_(columns), groups_per_row_(groups_per_row),
        logical_values_(std::move(logical_values)),
        scale_bits_(std::move(scale_bits)),
        packed_values_(std::move(packed_values)) {}

  std::uint64_t rows_;
  std::uint64_t columns_;
  std::uint64_t groups_per_row_;
  std::vector<std::int8_t> logical_values_;
  std::vector<std::uint16_t> scale_bits_;
  std::vector<std::byte> packed_values_;
};

}  // namespace pih
