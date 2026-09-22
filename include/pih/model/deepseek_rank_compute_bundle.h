#pragma once

#include <memory>
#include <optional>
#include <span>
#include <utility>

#include "pih/model/deepseek_bound_attention_work_provider.h"
#include "pih/model/deepseek_bound_dense_attention_work_provider.h"
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
#include "pih/model/deepseek_bound_dspark_stage_work_provider.h"
#include "pih/model/deepseek_bound_dspark_mtp_stage_work_provider.h"
#endif
#include "pih/model/deepseek_bound_endpoint_stage_work_provider.h"
#include "pih/model/deepseek_bound_expert_plan_provider.h"
#include "pih/model/deepseek_bound_mhc_stage_work_provider.h"
#include "pih/model/deepseek_bound_router_stage_work_provider.h"
#include "pih/model/deepseek_attention_plan_compute_driver.h"
#include "pih/model/deepseek_stage_compute_stack.h"

namespace pih {

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
class DeepSeekBoundDsparkMtpOperatorBackend;
class DeepSeekBoundDsparkMtpStageWorkProvider;
class DeepSeekBoundDsparkStageWorkProvider;
struct DeepSeekBoundDsparkMtpStageWork;
struct DeepSeekDsparkStageWork;
#endif
class DeepSeekRankComputeWorkBuilder;

class DeepSeekRankComputePlanWorkOwner final {
 private:
  DeepSeekRankComputePlanWorkOwner(
      const DeepSeekRankComputePlanWorkOwner&) = delete;
  DeepSeekRankComputePlanWorkOwner& operator=(
      const DeepSeekRankComputePlanWorkOwner&) = delete;
  DeepSeekRankComputePlanWorkOwner(
      DeepSeekRankComputePlanWorkOwner&&) = delete;
  DeepSeekRankComputePlanWorkOwner& operator=(
      DeepSeekRankComputePlanWorkOwner&&) = delete;
  explicit DeepSeekRankComputePlanWorkOwner(
      std::shared_ptr<const void> backing) noexcept
      : backing_(std::move(backing)) {}

  std::shared_ptr<const void> backing_;
  friend class DeepSeekRankComputeWorkBuilder;
};

struct DeepSeekRankComputePlanWork final {
  std::shared_ptr<const DeepSeekRankComputePlanWorkOwner> lifetime_owner;
  std::span<const DeepSeekBoundHashRouterWork> hash_router;
  std::span<const DeepSeekBoundLearnedRouterWork> learned_router;
  std::span<const DeepSeekBoundDecodeAttentionLayerWork> decode_attention;
  std::span<const DeepSeekBoundChunkAttentionLayerWork> chunk_attention;
  std::span<const DeepSeekBoundDenseAttentionLayerWork> dense_attention;
  std::span<const DeepSeekBoundMhcLayerWork> mhc_attention;
  std::span<const DeepSeekBoundMhcLayerWork> mhc_feed_forward;
  std::span<const DeepSeekEndpointStageSequenceWork> endpoint;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  const DeepSeekDsparkStageWork* dspark = nullptr;
  std::span<const DeepSeekBoundDsparkMtpStageWork> dspark_mtp;
#endif
};

class DeepSeekRankComputeBundle final : public DeepSeekStageComputeDriver {
 public:
  static Result<DeepSeekRankComputeBundle> Create(
      DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
      std::uint32_t maximum_tokens, DeepSeekExpertPager& pager,
      DeepSeekExpertTransferDriver& transfer,
      DeepSeekExpertKernelDriver& kernel,
      std::uintptr_t attention_compute_stream = 0);
  static Result<DeepSeekRankComputeBundle> CreateResident(
      DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
      std::uint32_t maximum_tokens, DeepSeekExpertKernelDriver& kernel,
      const DeepSeekResidentExpertBindings& resident_experts,
      std::uintptr_t attention_compute_stream = 0);

  DeepSeekRankComputeBundle(const DeepSeekRankComputeBundle&) = delete;
  DeepSeekRankComputeBundle& operator=(const DeepSeekRankComputeBundle&) = delete;
  DeepSeekRankComputeBundle(DeepSeekRankComputeBundle&&) noexcept = default;
  DeepSeekRankComputeBundle& operator=(
      DeepSeekRankComputeBundle&& other) noexcept;

  Status bind_plan(const DeepSeekPipelinePlanDescriptor& descriptor,
                   DeepSeekRankComputePlanWork work);
  Status bind_shared_experts(DeepSeekSharedExpertProvider& provider) {
    if (stack_ == nullptr) return Status::FailedPrecondition("Rank compute stack is unavailable");
    return stack_->bind_shared_experts(provider);
  }
  Status abort_bound_plan(
      const DeepSeekPipelinePlanDescriptor& descriptor) noexcept;
  [[nodiscard]] bool can_abort_bound_plan(
      const DeepSeekPipelinePlanDescriptor& descriptor) const noexcept;
  Status launch(const DeepSeekPipelinePlanDescriptor& plan,
                const DeepSeekStagePlan& stage) override;
  Result<DeepSeekStageComputeStatus> poll() override;

  [[nodiscard]] DeepSeekStagePlan stage() const noexcept { return stage_; }
  [[nodiscard]] std::uintptr_t attention_compute_stream() const noexcept {
    return attention_compute_stream_;
  }
  [[nodiscard]] bool attention_transaction_driver_bound() const noexcept {
    return attention_driver_ != nullptr;
  }
  [[nodiscard]] DeepSeekStageComputeDriver& driver() noexcept { return *this; }
  [[nodiscard]] DeepSeekBoundExpertPlanProvider& expert_store() noexcept {
    return *expert_store_;
  }
  [[nodiscard]] DeepSeekBoundRouterStageWorkProvider& router() noexcept {
    return *router_;
  }
  [[nodiscard]] DeepSeekBoundDecodeAttentionWorkProvider& decode_attention()
      noexcept { return *decode_attention_; }
  [[nodiscard]] DeepSeekBoundChunkAttentionWorkProvider& chunk_attention()
      noexcept { return *chunk_attention_; }
  [[nodiscard]] DeepSeekBoundDenseAttentionWorkProvider& dense_attention()
      noexcept { return *dense_attention_; }
  [[nodiscard]] DeepSeekBoundMhcStageWorkProvider& mhc() noexcept {
    return *mhc_;
  }
  [[nodiscard]] DeepSeekBoundEndpointStageWorkProvider* endpoint() noexcept {
    return endpoint_.get();
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  [[nodiscard]] DeepSeekBoundDsparkStageWorkProvider* dspark() noexcept {
    return dspark_.get();
  }
  [[nodiscard]] DeepSeekBoundDsparkMtpStageWorkProvider* dspark_mtp()
      noexcept { return dspark_mtp_provider_.get(); }
#endif

 private:
  static Result<DeepSeekRankComputeBundle> CreateConfigured(
      DeepSeekStagePlan stage, std::uint32_t maximum_sequences,
      std::uint32_t maximum_tokens, DeepSeekExpertPager* pager,
      DeepSeekExpertTransferDriver* transfer,
      DeepSeekExpertKernelDriver& kernel,
      const DeepSeekResidentExpertBindings* resident_experts,
      std::uintptr_t attention_compute_stream);
  DeepSeekRankComputeBundle() = default;
  void clear_plan() noexcept;

  DeepSeekStagePlan stage_;
  std::unique_ptr<DeepSeekBoundExpertPlanProvider> expert_store_;
  std::unique_ptr<DeepSeekBoundRouterStageWorkProvider> router_;
  std::unique_ptr<DeepSeekBoundDecodeAttentionWorkProvider> decode_attention_;
  std::unique_ptr<DeepSeekBoundChunkAttentionWorkProvider> chunk_attention_;
  std::unique_ptr<DeepSeekBoundDenseAttentionWorkProvider> dense_attention_;
  std::unique_ptr<DeepSeekBoundMhcStageWorkProvider> mhc_;
  std::unique_ptr<DeepSeekBoundEndpointStageWorkProvider> endpoint_;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::unique_ptr<DeepSeekBoundDsparkStageWorkProvider> dspark_;
  std::unique_ptr<DeepSeekBoundDsparkMtpStageWorkProvider>
      dspark_mtp_provider_;
  std::unique_ptr<DeepSeekBoundDsparkMtpOperatorBackend>
      dspark_mtp_backend_;
#endif
  // Destroyed first so every non-owning backend link remains valid.
  std::unique_ptr<DeepSeekStageComputeStack> stack_;
  // Plan-scoped wrapper borrows stack_ and is destroyed before it.
  std::unique_ptr<DeepSeekAttentionPlanComputeDriver> attention_driver_;
  std::uintptr_t attention_compute_stream_ = 0;
  std::optional<DeepSeekPipelinePlanDescriptor> bound_plan_;
  bool launched_ = false;
};

}  // namespace pih
