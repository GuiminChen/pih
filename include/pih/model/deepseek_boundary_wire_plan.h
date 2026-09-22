#pragma once

#include <cstdint>
#include <span>

#include "pih/model/deepseek_rank_compute_bundle.h"

namespace pih {

class DeepSeekBoundaryWirePlan final {
 public:
  static Result<DeepSeekBoundaryWirePlan> Create(
      DeepSeekPipelinePlanDescriptor descriptor,
      std::uint32_t world_size,
      std::span<const DeepSeekRankComputePlanWork> rank_work);

  [[nodiscard]] std::uint32_t wire_token_count() const noexcept {
    return wire_token_count_;
  }
  [[nodiscard]] std::uint32_t boundary_count() const noexcept {
    return boundary_count_;
  }

 private:
  DeepSeekBoundaryWirePlan(std::uint32_t wire_token_count,
                           std::uint32_t boundary_count) noexcept
      : wire_token_count_(wire_token_count),
        boundary_count_(boundary_count) {}

  std::uint32_t wire_token_count_ = 0;
  std::uint32_t boundary_count_ = 0;
};

}  // namespace pih
