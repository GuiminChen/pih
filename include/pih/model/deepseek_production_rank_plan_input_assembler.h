#pragma once

#include "pih/model/deepseek_deferred_native_plan_compiler.h"
#include "pih/model/deepseek_dense_mhc_stage_submission_assembler.h"
#include "pih/model/deepseek_request_input_staging_resources.h"
#include "pih/model/deepseek_attention_plan_input_shape_assembler.h"
#include "pih/model/deepseek_sampler.h"

namespace pih {

class DeepSeekEngineResources;

struct DeepSeekProductionRankPlanSeed final {
  std::vector<std::uint32_t> token_ids;
  std::vector<std::uint32_t> positions;
  std::vector<DeepSeekDecodeAttentionLayerPlanInput> decode_attention;
  std::vector<DeepSeekChunkAttentionLayerPlanInput> chunk_attention;
  std::vector<DeepSeekAttentionLayerPlanShapeSeed> attention_shape_seeds;
  DeepSeekAttentionSequenceBinding attention_binding;
  std::optional<DeepSeekPreparedSamplingInput> sampling;
  std::uint32_t table_position_count = 0;
};

struct DeepSeekProductionRankResolvedInput final {
  DeepSeekStagePlan stage;
  DeepSeekRankAttentionStatePool* attention_state_pool = nullptr;
  DeepSeekRequestInputStagingLease request_input_lease;
  std::vector<DeepSeekLearnedRouterLayerPlanWork> learned_router;
  std::vector<DeepSeekHashRouterLayerPlanWork> hash_router;
  DeepSeekDenseMhcStageSubmissions dense_mhc;
  std::optional<DeepSeekEndpointStageSequenceWork> endpoint;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::optional<DeepSeekDsparkStageWork> dspark;
  std::vector<DeepSeekBoundDsparkMtpStageWork> dspark_mtp;
#endif
  std::vector<std::shared_ptr<const void>> additional_lifetime_backings;
};

class DeepSeekProductionRankPlanInputAssembler final
    : public DeepSeekDeferredRankPlanInputAssembler {
 public:
  static Result<std::unique_ptr<DeepSeekProductionRankPlanInputAssembler>>
  Create(DeepSeekEngineResources& resources, std::uint32_t rank,
         DeepSeekPipelinePlanDescriptor descriptor,
         DeepSeekProductionRankPlanSeed seed);

  Result<DeepSeekRankPlanCompilerInput> assemble(
      std::uintptr_t incoming_activation_bf16) override;
  static Result<DeepSeekRankPlanCompilerInput> PublishResolved(
      const DeepSeekPipelinePlanDescriptor& descriptor,
      DeepSeekProductionRankPlanSeed seed,
      DeepSeekProductionRankResolvedInput resolved);
  static Status AssembleSchedulerAttentionShapes(
      const DeepSeekPipelinePlanDescriptor& descriptor,
      DeepSeekProductionRankPlanSeed& seed,
      DeepSeekProductionRankResolvedInput& resolved);

 private:
  DeepSeekProductionRankPlanInputAssembler(
      DeepSeekEngineResources& resources, std::uint32_t rank,
      DeepSeekPipelinePlanDescriptor descriptor,
      DeepSeekProductionRankPlanSeed seed
      ) noexcept
      : resources_(&resources), rank_(rank), descriptor_(descriptor),
        seed_(std::move(seed))
        {}

  DeepSeekEngineResources* resources_ = nullptr;
  std::uint32_t rank_ = 0;
  DeepSeekPipelinePlanDescriptor descriptor_;
  DeepSeekProductionRankPlanSeed seed_;
  bool assembled_ = false;
};

}  // namespace pih
