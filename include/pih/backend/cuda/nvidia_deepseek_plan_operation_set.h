#pragma once

#include <memory>
#include <optional>
#include <utility>

#include "pih/backend/cuda/nvidia_deepseek_attention_output_projection_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_attention_projection_operations.h"
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
#include "pih/backend/cuda/nvidia_deepseek_dspark_embed_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_dspark_head_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_dspark_prefill_stage_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_dspark_decode_stage_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_dspark_moe_stage_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_dspark_state_digest_operations.h"
#endif
#include "pih/backend/cuda/nvidia_deepseek_endpoint_sequence_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_learned_router_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_mhc_sequence_operations.h"
#include "pih/model/deepseek_fused_kernel_plan.h"
#include "pih/contracts/deepseek_kernels_v1.h"

namespace pih {

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
class DeepSeekDsparkDecodeStageOperations;
class DeepSeekDsparkEmbedOperations;
class DeepSeekDsparkGpuStateDigestOperations;
class DeepSeekDsparkHeadOperations;
class DeepSeekDsparkMoeStageOperations;
class DeepSeekDsparkPrefillStageOperations;
#endif

class NvidiaDeepSeekPlanOperationSet final {
 public:
  // The caller supplies the exact, already-admitted optimization policy.
  // In particular, this operation set must not infer SM89/PP1 defaults for
  // an H100 or a multi-rank pipeline.
  static Result<NvidiaDeepSeekPlanOperationSet> Create(
      const DeepSeekOptimizationPolicy& policy,
      const pih_deepseek_kernels_api_v1& kernels,
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1& async_api);

  [[nodiscard]] const DeepSeekFusedKernelPlan& kernel_plan() const noexcept {
    return kernel_plan_;
  }

  [[nodiscard]] DeepSeekLearnedRouterOperations& learned_router() noexcept {
    return learned_router_;
  }
  [[nodiscard]] DeepSeekAttentionProjectionOperations&
  attention_projection() noexcept { return attention_projection_; }
  [[nodiscard]] DeepSeekAttentionOutputProjectionOperations&
  attention_output_projection() noexcept {
    return attention_output_projection_;
  }
  [[nodiscard]] DeepSeekEndpointSequenceOperations& endpoint() noexcept {
    return endpoint_;
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  [[nodiscard]] DeepSeekDsparkEmbedOperations* dspark_embed() noexcept {
    return dspark_embed_ ? &*dspark_embed_ : nullptr;
  }
  [[nodiscard]] DeepSeekDsparkHeadOperations* dspark_head() noexcept {
    return dspark_head_ ? &*dspark_head_ : nullptr;
  }
  [[nodiscard]] DeepSeekDsparkPrefillStageOperations*
  dspark_prefill() noexcept {
    return dspark_prefill_ ? &*dspark_prefill_ : nullptr;
  }
  [[nodiscard]] DeepSeekDsparkDecodeStageOperations*
  dspark_decode() noexcept {
    return dspark_decode_ ? &*dspark_decode_ : nullptr;
  }
  [[nodiscard]] DeepSeekDsparkMoeStageOperations* dspark_moe() noexcept {
    return dspark_moe_ ? &*dspark_moe_ : nullptr;
  }
  [[nodiscard]] DeepSeekDsparkGpuStateDigestOperations*
  dspark_state_digest() noexcept {
    return dspark_state_digest_ ? &*dspark_state_digest_ : nullptr;
  }
#endif
  [[nodiscard]] DeepSeekMhcSequenceOperations& mhc() noexcept {
    return mhc_;
  }

 private:
  static Result<NvidiaDeepSeekPlanOperationSet> Create(
      DeepSeekFusedKernelPlan kernel_plan, bool enable_dspark,
      const pih_deepseek_kernels_api_v1* kernels,
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1* async_api);
  class RejectStandaloneMhcBranch final
      : public NvidiaDeepSeekMhcBranchDriver {
   public:
    Status launch(DeepSeekMhcBranchLaunch) override {
      return Status::FailedPrecondition(
          "DeepSeek production mHC branch is driven by the stage backend");
    }
  };
  NvidiaDeepSeekPlanOperationSet(
      DeepSeekFusedKernelPlan kernel_plan,
      NvidiaDeepSeekLearnedRouterOperations learned_router,
      NvidiaDeepSeekAttentionProjectionOperations attention_projection,
      NvidiaDeepSeekAttentionOutputProjectionOperations
          attention_output_projection,
      NvidiaDeepSeekEndpointSequenceOperations endpoint,
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      std::optional<NvidiaDeepSeekDsparkEmbedOperations> dspark_embed,
      std::optional<NvidiaDeepSeekDsparkHeadOperations> dspark_head,
      std::optional<NvidiaDeepSeekDsparkPrefillStageOperations> dspark_prefill,
      std::optional<NvidiaDeepSeekDsparkDecodeStageOperations> dspark_decode,
      std::optional<NvidiaDeepSeekDsparkMoeStageOperations> dspark_moe,
      std::optional<NvidiaDeepSeekDsparkStateDigestOperations>
          dspark_state_digest,
#endif
      std::unique_ptr<RejectStandaloneMhcBranch> mhc_branch,
      NvidiaDeepSeekMhcSequenceOperations mhc) noexcept
      : kernel_plan_(std::move(kernel_plan)),
        learned_router_(std::move(learned_router)),
        attention_projection_(std::move(attention_projection)),
        attention_output_projection_(std::move(attention_output_projection)),
        endpoint_(std::move(endpoint)),
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
        dspark_embed_(std::move(dspark_embed)),
        dspark_head_(std::move(dspark_head)),
        dspark_prefill_(std::move(dspark_prefill)),
        dspark_decode_(std::move(dspark_decode)),
        dspark_moe_(std::move(dspark_moe)),
        dspark_state_digest_(std::move(dspark_state_digest)),
#endif
        mhc_branch_(std::move(mhc_branch)),
        mhc_(std::move(mhc)) {}

  DeepSeekFusedKernelPlan kernel_plan_;
  NvidiaDeepSeekLearnedRouterOperations learned_router_;
  NvidiaDeepSeekAttentionProjectionOperations attention_projection_;
  NvidiaDeepSeekAttentionOutputProjectionOperations
      attention_output_projection_;
  NvidiaDeepSeekEndpointSequenceOperations endpoint_;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::optional<NvidiaDeepSeekDsparkEmbedOperations> dspark_embed_;
  std::optional<NvidiaDeepSeekDsparkHeadOperations> dspark_head_;
  std::optional<NvidiaDeepSeekDsparkPrefillStageOperations> dspark_prefill_;
  std::optional<NvidiaDeepSeekDsparkDecodeStageOperations> dspark_decode_;
  std::optional<NvidiaDeepSeekDsparkMoeStageOperations> dspark_moe_;
  std::optional<NvidiaDeepSeekDsparkStateDigestOperations>
      dspark_state_digest_;
#endif
  std::unique_ptr<RejectStandaloneMhcBranch> mhc_branch_;
  NvidiaDeepSeekMhcSequenceOperations mhc_;
};

}  // namespace pih
