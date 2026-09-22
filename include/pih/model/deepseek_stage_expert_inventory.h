#pragma once

#include <cstdint>
#include <vector>

#include "pih/model/deepseek_expert_transfer_driver.h"
#include "pih/model/deepseek_v4_config.h"

namespace pih {

struct DeepSeekStageExpertExtentRecord final {
  DeepSeekExpertIdentity identity;
  DeepSeekPinnedExpertExtent extent;
};

class DeepSeekStageExpertInventory final : public DeepSeekExpertHostSource {
 public:
  static Result<DeepSeekStageExpertInventory> Create(
      DeepSeekStageRange owned_layers,
      std::vector<DeepSeekStageExpertExtentRecord> records);

  Result<DeepSeekPinnedExpertExtent> resolve(
      DeepSeekExpertIdentity identity, std::uint64_t bytes) override;
  [[nodiscard]] DeepSeekStageRange owned_layers() const noexcept {
    return owned_layers_;
  }
  [[nodiscard]] std::uint32_t record_count() const noexcept {
    return static_cast<std::uint32_t>(extents_.size());
  }

 private:
  DeepSeekStageRange owned_layers_;
  std::vector<DeepSeekPinnedExpertExtent> extents_;
};

}  // namespace pih
