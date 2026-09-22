#pragma once

#include <memory>

#include "pih/model/deepseek_attention_stage_backend.h"
#include "pih/model/deepseek_dense_attention_stage_backend.h"
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
#include "pih/model/deepseek_dspark_stage_backend.h"
#include "pih/model/deepseek_dspark_mtp_block_executor.h"
#endif
#include "pih/model/deepseek_endpoint_stage_backend.h"
#include "pih/model/deepseek_mhc_stage_backend.h"
#include "pih/model/deepseek_router_stage_backend.h"
#include "pih/model/deepseek_routed_stage_backend.h"
#include "pih/model/deepseek_shared_expert_driver.h"

namespace pih {

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
class DeepSeekDsparkMtpBlockExecutor;
class DeepSeekDsparkMtpOperatorBackend;
class DeepSeekDsparkStageOperatorBackend;
class DeepSeekDsparkStageWorkProvider;
#endif

struct DeepSeekStageComputeStackDependencies final {
  DeepSeekDecodeAttentionWorkProvider* decode_attention = nullptr;
  DeepSeekChunkAttentionWorkProvider* chunk_attention = nullptr;
  DeepSeekDenseAttentionStageWorkProvider* dense_attention = nullptr;
  DeepSeekRouterStageWorkProvider* router = nullptr;
  DeepSeekExpertPlanProvider* expert_plan = nullptr;
  DeepSeekExpertPager* expert_pager = nullptr;
  DeepSeekExpertTransferDriver* expert_transfer = nullptr;
  const DeepSeekResidentExpertBindings* resident_experts = nullptr;
  DeepSeekExpertKernelDriver* expert_kernel = nullptr;
  DeepSeekSharedExpertProvider* shared_experts = nullptr;
  DeepSeekMhcStageWorkProvider* mhc = nullptr;
  DeepSeekEndpointStageWorkProvider* endpoint = nullptr;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  DeepSeekDsparkStageWorkProvider* dspark = nullptr;
  DeepSeekDsparkMtpOperatorBackend* dspark_mtp = nullptr;
#endif
};

// Owns the canonical V1 operator backend chain. Every internal backend lives on
// the heap so the non-owning links used by the individual wrappers remain
// stable when this driver is moved into a rank runtime.
class DeepSeekStageComputeStack final : public DeepSeekStageComputeDriver {
 public:
  static Result<DeepSeekStageComputeStack> Create(
      DeepSeekStagePlan stage,
      DeepSeekStageComputeStackDependencies dependencies);

  DeepSeekStageComputeStack(const DeepSeekStageComputeStack&) = delete;
  DeepSeekStageComputeStack& operator=(const DeepSeekStageComputeStack&) = delete;
  DeepSeekStageComputeStack(DeepSeekStageComputeStack&&) noexcept;
  DeepSeekStageComputeStack& operator=(DeepSeekStageComputeStack&&) noexcept;
  ~DeepSeekStageComputeStack() override;
  // Startup only; provider must outlive this stack. Launch is forbidden until bound.
  Status bind_shared_experts(DeepSeekSharedExpertProvider& provider);

  Status launch(const DeepSeekPipelinePlanDescriptor& plan,
                const DeepSeekStagePlan& stage) override;
  Result<DeepSeekStageComputeStatus> poll() override;

 private:
  class RejectBackend;
  class SharedProviderSlot;
  DeepSeekStageComputeStack() = default;

  std::unique_ptr<RejectBackend> reject_;
  std::unique_ptr<DeepSeekRoutedStageOperatorBackend> routed_;
  std::unique_ptr<SharedProviderSlot> shared_provider_;
  std::unique_ptr<DeepSeekSharedExpertStageBackend> shared_;
  std::unique_ptr<DeepSeekRouterStageOperatorBackend> router_;
  std::unique_ptr<DeepSeekAttentionStageOperatorBackend> attention_;
  std::unique_ptr<DeepSeekDenseAttentionStageOperatorBackend> dense_attention_;
  std::unique_ptr<DeepSeekMhcStageOperatorBackend> mhc_;
  std::unique_ptr<DeepSeekEndpointStageOperatorBackend> endpoint_;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::unique_ptr<DeepSeekDsparkStageOperatorBackend> dspark_;
  std::unique_ptr<DeepSeekDsparkMtpBlockExecutor> dspark_block_;
#endif
  std::unique_ptr<DeepSeekModelStageComputeDriver> driver_;
  bool started_ = false;
};

}  // namespace pih
