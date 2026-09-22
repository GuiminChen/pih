#include "engram_weights.h"
#include <algorithm>

namespace pih::deepseek_v41 {
Result<EngramWeightPlan> EngramWeightPlan::Create(const FlashConfig& config,
    std::uint32_t layer, std::uint32_t world_size, std::uint32_t rank,
    EngramStorage gate_storage) {
  // Both backbone and MTP expert partitions must remain integral. This is a
  // geometry constraint, not a claim that any particular TP topology can run.
  if (config.config_sha256() == Sha256Digest{} || (layer != 1 && layer != 14) ||
      world_size == 0 || world_size > 128 || 384 % world_size != 0 ||
      128 % world_size != 0 || rank >= world_size)
    return Status::InvalidArgument("Engram layer or V4.1 TP geometry invalid");
  if (gate_storage != EngramStorage::kBF16 && gate_storage != EngramStorage::kF32)
    return Status::InvalidArgument("Engram gate storage must be explicit BF16 or F32");
  EngramWeightPlan plan;
  plan.global_rows_ = layer == 1 ? 384006168U : 384016682U;
  plan.world_size_ = world_size;
  const std::uint32_t partition = (plan.global_rows_ + world_size - 1) / world_size;
  plan.first_row_ = rank * partition;
  plan.valid_rows_ = std::min(partition, plan.global_rows_ - plan.first_row_);
  plan.padded_rows_ = partition - plan.valid_rows_;
  const std::uint64_t rows = partition;
  const std::uint64_t gate_bytes = gate_storage == EngramStorage::kBF16 ? 2 : 4;
  plan.matrices_ = {{
      {"embed.weight", EngramStorage::kE4M3FN, rows, 256, rows * 256},
      {"embed.scale", EngramStorage::kE8M0, rows, 8, rows * 8},
      {"wkv.weight", EngramStorage::kE4M3FN, 25600, 6144, 25600ULL * 6144},
      {"wkv.scale", EngramStorage::kE8M0, 800, 192, 800ULL * 192},
      {"q_weight", gate_storage, 4, 5120, 4 * 5120 * gate_bytes},
      {"k_weight", gate_storage, 4, 5120, 4 * 5120 * gate_bytes}}};
  return plan;
}
Result<EngramRowLocation> EngramWeightPlan::Locate(std::uint32_t global_row) const {
  if (global_row >= global_rows_)
    return Status::InvalidArgument("Engram hash row outside the unpadded global table");
  const bool owned = global_row >= first_row_ && global_row - first_row_ < valid_rows_;
  return EngramRowLocation{owned, owned ? global_row - first_row_ : 0};
}
Status EngramWeightPlan::Validate(std::span<const EngramMatrix> supplied) const {
  if (supplied.size() != matrices_.size())
    return Status::InvalidArgument("Engram requires exactly six weight members");
  std::array<bool, 6> seen{};
  for (const auto& actual : supplied) {
    std::size_t slot = 0;
    while (slot < matrices_.size() && matrices_[slot].name != actual.name) ++slot;
    if (slot == matrices_.size() || seen[slot])
      return Status::InvalidArgument("Engram weight name unknown or duplicated");
    seen[slot] = true;
    const auto& expected = matrices_[slot];
    if (actual.storage != expected.storage || actual.rows != expected.rows ||
        actual.columns != expected.columns || actual.bytes != expected.bytes)
      return Status::InvalidArgument("Engram weight dtype, layout or byte length differs");
  }
  return Status::Ok();
}
}  // namespace pih::deepseek_v41
