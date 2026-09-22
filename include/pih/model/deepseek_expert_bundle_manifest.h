#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/deepseek_mapped_expert_source.h"
#include "pih/model/deepseek_rank_tensor_record.h"

namespace pih {

class DeepSeekExpertBundleManifest final {
 public:
  static Result<DeepSeekExpertBundleManifest> Create(
      DeepSeekStageRange owned_layers,
      std::span<const DeepSeekRankTensorRecord> rank_tensors);

  [[nodiscard]] const std::vector<DeepSeekExpertMappedBundleRecord>& bundles()
      const noexcept { return bundles_; }
  [[nodiscard]] std::uint64_t payload_bytes() const noexcept {
    return payload_bytes_;
  }

 private:
  std::vector<DeepSeekExpertMappedBundleRecord> bundles_;
  std::uint64_t payload_bytes_ = 0;
};

}  // namespace pih
