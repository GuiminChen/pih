#include "pih/backend/cuda/nvidia_deepseek_plan_operation_set.h"

namespace pih {

Result<NvidiaDeepSeekPlanOperationSet>
NvidiaDeepSeekPlanOperationSet::Create(
    const DeepSeekOptimizationPolicy& policy,
    const pih_deepseek_kernels_api_v1& kernels,
    std::uintptr_t retained_context,
    const pih_nvidia_cuda_async_api_v1& async_api) {
  auto plan = DeepSeekFusedKernelPlan::Compile(policy, 1, nullptr, nullptr);
  if (!plan.ok()) return plan.status();
  return Create(std::move(*plan), policy.dspark(), &kernels, retained_context,
                &async_api);
}

Result<NvidiaDeepSeekPlanOperationSet>
NvidiaDeepSeekPlanOperationSet::Create(DeepSeekFusedKernelPlan kernel_plan,
                                       bool enable_dspark,
                                       const pih_deepseek_kernels_api_v1*
                                           kernels,
                                       std::uintptr_t retained_context,
                                       const pih_nvidia_cuda_async_api_v1*
                                           async_api) {
  auto learned_router = NvidiaDeepSeekLearnedRouterOperations::Create(
      retained_context, *async_api, *kernels);
  if (!learned_router.ok()) return learned_router.status();
  auto attention_projection =
      NvidiaDeepSeekAttentionProjectionOperations::Create(
          retained_context, *async_api, *kernels);
  if (!attention_projection.ok()) return attention_projection.status();
  auto attention_output_projection =
      NvidiaDeepSeekAttentionOutputProjectionOperations::Create(
          retained_context, *async_api, *kernels);
  if (!attention_output_projection.ok()) {
    return attention_output_projection.status();
  }
  auto endpoint = NvidiaDeepSeekEndpointSequenceOperations::Create(
      retained_context, *async_api, *kernels);
  if (!endpoint.ok()) return endpoint.status();
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::optional<NvidiaDeepSeekDsparkEmbedOperations> dspark_embed;
  std::optional<NvidiaDeepSeekDsparkHeadOperations> dspark_head;
  std::optional<NvidiaDeepSeekDsparkPrefillStageOperations> dspark_prefill;
  std::optional<NvidiaDeepSeekDsparkDecodeStageOperations> dspark_decode;
  std::optional<NvidiaDeepSeekDsparkMoeStageOperations> dspark_moe;
  std::optional<NvidiaDeepSeekDsparkStateDigestOperations>
      dspark_state_digest;
  if (enable_dspark) {
    auto embed = NvidiaDeepSeekDsparkEmbedOperations::Create();
    if (!embed.ok()) return embed.status();
    dspark_embed.emplace(std::move(*embed));
    auto head = NvidiaDeepSeekDsparkHeadOperations::Create();
    if (!head.ok()) return head.status();
    dspark_head.emplace(std::move(*head));
    auto prefill = NvidiaDeepSeekDsparkPrefillStageOperations::Create();
    if (!prefill.ok()) return prefill.status();
    dspark_prefill.emplace(std::move(*prefill));
    auto decode = NvidiaDeepSeekDsparkDecodeStageOperations::Create();
    if (!decode.ok()) return decode.status();
    dspark_decode.emplace(std::move(*decode));
    auto moe = NvidiaDeepSeekDsparkMoeStageOperations::Create();
    if (!moe.ok()) return moe.status();
    dspark_moe.emplace(std::move(*moe));
    auto digest = NvidiaDeepSeekDsparkStateDigestOperations::Create();
    if (!digest.ok()) return digest.status();
    dspark_state_digest.emplace(std::move(*digest));
  }
#else
  if (enable_dspark) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark is not present in this CUDA backend bundle");
  }
#endif
  auto mhc_branch = std::make_unique<RejectStandaloneMhcBranch>();
  auto mhc = NvidiaDeepSeekMhcSequenceOperations::Create(
      retained_context, *async_api, *mhc_branch, *kernels);
  if (!mhc.ok()) return mhc.status();
  return NvidiaDeepSeekPlanOperationSet(
      std::move(kernel_plan),
      std::move(*learned_router), std::move(*attention_projection),
      std::move(*attention_output_projection), std::move(*endpoint),
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      std::move(dspark_embed), std::move(dspark_head),
      std::move(dspark_prefill), std::move(dspark_decode),
      std::move(dspark_moe), std::move(dspark_state_digest),
#endif
      std::move(mhc_branch),
      std::move(*mhc));
}

}  // namespace pih
