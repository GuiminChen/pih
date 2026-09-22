#pragma once

#include <array>
#include <optional>
#include <span>

#include "pih/model/deepseek_attention_stage_backend.h"

namespace pih {

struct DeepSeekBoundDecodeAttentionLayerWork final {
  std::uint32_t layer = 0;
  DeepSeekDecodeAttentionWork work;
};

struct DeepSeekBoundChunkAttentionLayerWork final {
  std::uint32_t layer = 0;
  DeepSeekChunkAttentionWork work;
};

class DeepSeekBoundDecodeAttentionWorkProvider final
    : public DeepSeekDecodeAttentionWorkProvider {
 public:
  static Result<DeepSeekBoundDecodeAttentionWorkProvider> Create(
      DeepSeekStageRange owned_layers);
  Status bind(
      const DeepSeekPipelinePlanDescriptor& descriptor,
      std::span<const DeepSeekBoundDecodeAttentionLayerWork> layers);
  void clear() noexcept { descriptor_.reset(); }
  Status reset_completed();
  Result<const DeepSeekDecodeAttentionWork*> resolve(
      std::uint32_t layer,
      const DeepSeekPipelinePlanDescriptor& plan) override;

 private:
  DeepSeekStageRange owned_layers_;
  std::array<std::array<DeepSeekDecodeAttentionWork, 43>, 2> banks_{};
  std::size_t active_bank_ = 0;
  std::optional<DeepSeekPipelinePlanDescriptor> descriptor_;
};

class DeepSeekBoundChunkAttentionWorkProvider final
    : public DeepSeekChunkAttentionWorkProvider {
 public:
  static Result<DeepSeekBoundChunkAttentionWorkProvider> Create(
      DeepSeekStageRange owned_layers);
  Status bind(
      const DeepSeekPipelinePlanDescriptor& descriptor,
      std::span<const DeepSeekBoundChunkAttentionLayerWork> layers);
  void clear() noexcept { descriptor_.reset(); }
  Status reset_completed();
  Result<const DeepSeekChunkAttentionWork*> resolve(
      std::uint32_t layer,
      const DeepSeekPipelinePlanDescriptor& plan) override;

 private:
  DeepSeekStageRange owned_layers_;
  std::array<std::array<DeepSeekChunkAttentionWork, 43>, 2> banks_{};
  std::size_t active_bank_ = 0;
  std::optional<DeepSeekPipelinePlanDescriptor> descriptor_;
};

}  // namespace pih
