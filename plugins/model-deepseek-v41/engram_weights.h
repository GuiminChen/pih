#pragma once
#include "config.h"
#include <span>

namespace pih::deepseek_v41 {
enum class EngramStorage { kE4M3FN, kE8M0, kBF16, kF32 };
struct EngramMatrix final {
  // Suffix relative to layers.<layer>.engram. No aliases or transposes.
  std::string_view name;
  EngramStorage storage;
  std::uint64_t rows, columns, bytes;
};
struct EngramRowLocation final {
  bool owned = false;
  // Zero is the safe gather row for a non-owner; its output must be masked out.
  std::uint32_t local_row = 0;
};
class EngramWeightPlan final {
 public:
  // Gate storage is explicit: the reference creates q/k in the active default
  // floating dtype. This API does not infer checkpoint dtype from that default.
  static Result<EngramWeightPlan> Create(const FlashConfig& config,
      std::uint32_t layer, std::uint32_t world_size, std::uint32_t rank,
      EngramStorage gate_storage);
  const std::array<EngramMatrix, 6>& matrices() const noexcept { return matrices_; }
  std::uint32_t global_rows() const noexcept { return global_rows_; }
  std::uint32_t first_row() const noexcept { return first_row_; }
  std::uint32_t valid_rows() const noexcept { return valid_rows_; }
  std::uint32_t padded_rows() const noexcept { return padded_rows_; }
  bool requires_sum_collective() const noexcept { return world_size_ > 1; }
  // Rejects invalid global IDs, including padding in the last partition.
  Result<EngramRowLocation> Locate(std::uint32_t global_row) const;
  // Metadata admission only. Does not authenticate payloads or validate scales.
  Status Validate(std::span<const EngramMatrix> supplied) const;
 private:
  EngramWeightPlan() = default;
  std::array<EngramMatrix, 6> matrices_{};
  std::uint32_t global_rows_ = 0, first_row_ = 0, valid_rows_ = 0;
  std::uint32_t padded_rows_ = 0, world_size_ = 0;
};
}  // namespace pih::deepseek_v41
