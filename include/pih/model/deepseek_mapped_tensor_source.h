#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_tensor_record.h"
#include "pih/model/deepseek_stage_mapped_inventory.h"

namespace pih {

struct DeepSeekMappedTensorSpan final {
  const DeepSeekRankTensorRecord* record = nullptr;
  std::span<const std::byte> bytes;
};

class DeepSeekWeightByteSource {
 public:
  virtual ~DeepSeekWeightByteSource() = default;
  virtual Result<std::span<const std::byte>> resolve_weight_bytes(
      std::string_view tensor_name) const = 0;
};

class DeepSeekMappedTensorSource final : public DeepSeekWeightByteSource {
 public:
  static Result<DeepSeekMappedTensorSource> Create(
      const DeepSeekStageMappedInventory& inventory,
      std::span<const DeepSeekRankTensorRecord> records);

  Result<DeepSeekMappedTensorSpan> resolve(std::string_view tensor_name) const;
  Result<std::span<const std::byte>> resolve_weight_bytes(
      std::string_view tensor_name) const override;
  [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }

 private:
  DeepSeekMappedTensorSource(const DeepSeekStageMappedInventory& inventory,
                             std::vector<DeepSeekRankTensorRecord> records)
      : inventory_(&inventory), records_(std::move(records)) {}

  const DeepSeekStageMappedInventory* inventory_ = nullptr;
  std::vector<DeepSeekRankTensorRecord> records_;
};

}  // namespace pih
