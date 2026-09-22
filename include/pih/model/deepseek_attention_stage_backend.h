#pragma once

#include "pih/model/deepseek_attention_layer_coordinator.h"
#include "pih/model/deepseek_prefill_layer_coordinator.h"
#include "pih/model/deepseek_stage_compute_driver.h"

namespace pih {

struct DeepSeekDecodeAttentionWork final {
  DeepSeekRecentStateSubmission recent;
  DeepSeekCompressedLayerUpdateSubmission update;
  DeepSeekAttentionLayerSubmission attention;
  DeepSeekRecentStateWriter* recent_writer = nullptr;
  DeepSeekCompressedLayerUpdateCoordinator* update_coordinator = nullptr;
  DeepSeekAttentionLayerCoordinator* coordinator = nullptr;
  DeepSeekAttentionSequenceTransaction* transaction = nullptr;
};

class DeepSeekDecodeAttentionWorkProvider {
 public:
  virtual ~DeepSeekDecodeAttentionWorkProvider() = default;
  // The work and all span-backed storage must remain valid until poll succeeds.
  virtual Result<const DeepSeekDecodeAttentionWork*> resolve(
      std::uint32_t layer,
      const DeepSeekPipelinePlanDescriptor& plan) = 0;
};

struct DeepSeekChunkAttentionWork final {
  DeepSeekPrefillLayerSubmission submission;
  DeepSeekPrefillLayerCoordinator* chunk_coordinator = nullptr;
  DeepSeekAttentionLayerCoordinator* attention_coordinator = nullptr;
  DeepSeekAttentionSequenceTransaction* transaction = nullptr;
};

class DeepSeekChunkAttentionWorkProvider {
 public:
  virtual ~DeepSeekChunkAttentionWorkProvider() = default;
  // The work and all span-backed storage must remain valid until poll succeeds.
  virtual Result<const DeepSeekChunkAttentionWork*> resolve(
      std::uint32_t layer,
      const DeepSeekPipelinePlanDescriptor& plan) = 0;
};

class DeepSeekAttentionStageOperatorBackend final
    : public DeepSeekStageOperatorBackend {
 public:
  static Result<DeepSeekAttentionStageOperatorBackend> Create(
      DeepSeekStageOperatorBackend& fallback,
      DeepSeekDecodeAttentionWorkProvider& provider);
  static Result<DeepSeekAttentionStageOperatorBackend> Create(
      DeepSeekStageOperatorBackend& fallback,
      DeepSeekDecodeAttentionWorkProvider& decode_provider,
      DeepSeekChunkAttentionWorkProvider& chunk_provider);

  Status launch(const DeepSeekStageOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor& plan) override;
  Result<DeepSeekStageComputeStatus> poll() override;

 private:
  enum class Mode : std::uint8_t { kIdle, kFallback, kAttention, kPoisoned };
  Status poison(Status status);

  DeepSeekStageOperatorBackend* fallback_ = nullptr;
  DeepSeekDecodeAttentionWorkProvider* provider_ = nullptr;
  DeepSeekChunkAttentionWorkProvider* chunk_provider_ = nullptr;
  const DeepSeekDecodeAttentionWork* work_ = nullptr;
  DeepSeekAttentionLayerCoordinator* active_coordinator_ = nullptr;
  Mode mode_ = Mode::kIdle;
};

}  // namespace pih
