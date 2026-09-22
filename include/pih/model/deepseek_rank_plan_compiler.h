#pragma once

#include "pih/model/deepseek_attention_work_factory.h"
#include "pih/model/deepseek_dense_mhc_work_factory.h"
#include "pih/model/deepseek_endpoint_work_factory.h"
#include "pih/model/deepseek_hash_router_work_factory.h"
#include "pih/model/deepseek_learned_router_work_factory.h"
#include "pih/model/deepseek_rank_attention_transaction_factory.h"
#include "pih/model/deepseek_dense_mhc_plan_input_assembler.h"

namespace pih {

struct DeepSeekRankPlanCompilerInput final {
  std::vector<std::uint32_t> token_ids;
  std::vector<DeepSeekHashRouterLayerScores> hash_router_scores;
  std::vector<DeepSeekHashRouterLayerPlanWork> projected_hash_router;
  std::vector<DeepSeekLearnedRouterLayerPlanWork> learned_router;
  std::vector<DeepSeekDecodeAttentionLayerPlanInput> decode_attention;
  std::vector<DeepSeekChunkAttentionLayerPlanInput> chunk_attention;
  std::vector<DeepSeekDenseAttentionLayerPlanInput> dense_attention;
  std::vector<DeepSeekMhcLayerPlanInput> mhc_attention;
  std::vector<DeepSeekMhcLayerPlanInput> mhc_feed_forward;
  std::vector<DeepSeekDenseMhcLayerSubmissionInput> dense_mhc_layers;
  std::vector<DeepSeekEndpointStageSequenceWork> endpoint;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::optional<DeepSeekDsparkStageWork> dspark;
  std::vector<DeepSeekBoundDsparkMtpStageWork> dspark_mtp;
#endif
  DeepSeekRankAttentionStatePool* attention_state_pool = nullptr;
  std::vector<DeepSeekAttentionSequenceBinding> attention_bindings;
  std::vector<std::shared_ptr<const void>> lifetime_backings;
};

class DeepSeekRankPlanCompiler final {
 public:
  static Result<DeepSeekRankPlanCompiler> Create(
      DeepSeekStagePlan stage,
      DeepSeekHashRouterWorkFactory hash_router,
      DeepSeekLearnedRouterWorkFactory learned_router,
      DeepSeekAttentionWorkFactory attention,
      DeepSeekDenseMhcWorkFactory dense_mhc,
      DeepSeekEndpointWorkFactory endpoint,
      DeepSeekDenseMhcRuntimeResources* dense_mhc_runtime = nullptr);

  Result<DeepSeekRankComputePlanWork> compile(
      const DeepSeekPipelinePlanDescriptor& descriptor,
      DeepSeekRankPlanCompilerInput input);
  [[nodiscard]] const DeepSeekStagePlan& stage() const noexcept {
    return stage_;
  }

 private:
  DeepSeekStagePlan stage_;
  DeepSeekHashRouterWorkFactory hash_router_;
  DeepSeekLearnedRouterWorkFactory learned_router_;
  DeepSeekAttentionWorkFactory attention_;
  DeepSeekDenseMhcWorkFactory dense_mhc_;
  DeepSeekEndpointWorkFactory endpoint_;
  DeepSeekDenseMhcRuntimeResources* dense_mhc_runtime_ = nullptr;
};

}  // namespace pih
