#pragma once

#include <optional>

#include "pih/model/deepseek_rank_compute_bundle.h"

namespace pih {

class DeepSeekDeferredRankComputePlanCompiler {
 public:
  virtual ~DeepSeekDeferredRankComputePlanCompiler() = default;
  [[nodiscard]] virtual std::uint32_t world_size() const noexcept = 0;
  virtual Result<DeepSeekRankComputePlanWork> compile(
      std::uint32_t rank,
      std::uintptr_t incoming_activation_bf16) = 0;
};

}  // namespace pih
