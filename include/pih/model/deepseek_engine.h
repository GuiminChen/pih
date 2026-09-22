#pragma once

#include <cstdint>
#include <cstddef>
#include <chrono>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pih/model/deepseek_engine_resources.h"
#include "pih/model/deepseek_engine_artifact_poller.h"
#include "pih/model/deepseek_control_plane.h"
#include "pih/model/deepseek_native_rank_compute_plan.h"
#include "pih/model/deepseek_rank_plan_runtime.h"
#include "pih/model/deepseek_deferred_rank_compute_plan_compiler.h"

namespace pih {

enum class DeepSeekEngineState : std::uint8_t {
  kReady,
  kClosing,
  kFailed,
  kClosed,
};

enum class DeepSeekEngineFailure : std::uint8_t {
  kNone,
  kArtifactIntegrity,
  kMappingFault,
  kWorkerLost,
};

class DeepSeekEngine final {
 public:
  static Result<DeepSeekEngine> Create(
      std::unique_ptr<DeepSeekEngineResources> resources,
      std::unique_ptr<DeepSeekEngineArtifactPoller> artifact_poller,
      DeepSeekPipelineCapacity pipeline_capacity);

  DeepSeekEngine(const DeepSeekEngine&) = delete;
  DeepSeekEngine& operator=(const DeepSeekEngine&) = delete;
  DeepSeekEngine(DeepSeekEngine&&) noexcept = default;
  DeepSeekEngine& operator=(DeepSeekEngine&& other) noexcept;

  Status admit_request();
  Status submit_request(std::uint64_t request_id,
                        std::uint64_t request_generation);
  Status cancel_request(std::uint64_t request_id,
                        std::uint64_t request_generation);
  Status finish_request(std::uint64_t request_id,
                        std::uint64_t request_generation);
  Status retire_request(std::uint64_t request_id,
                        std::uint64_t request_generation);
  Status configure_ledger(std::uint64_t request_id,
                          std::uint64_t request_generation,
                          std::uint64_t prompt_token_count,
                          std::uint32_t maximum_completion_tokens,
                          std::uint32_t minimum_completion_tokens);
  Status configure_sampling(std::uint64_t request_id,
                            std::uint64_t request_generation,
                            DeepSeekRequestSamplingConfig config);
  Result<DeepSeekRequestState> request_state(
      std::uint64_t request_id, std::uint64_t request_generation) const;
  Result<DeepSeekAcceptedTokenSnapshot> accepted_token_snapshot(
      std::uint64_t request_id, std::uint64_t request_generation) const;
  Result<DeepSeekAttentionSequenceBinding> request_attention_binding(
      std::uint64_t request_id, std::uint64_t request_generation) const;
  Status prepare_pipeline(
      DeepSeekPipelinePlanDescriptor descriptor,
      std::vector<DeepSeekRequestIdentity> request_identities);
  Status prepare_bound_pipeline(
      DeepSeekPipelinePlanDescriptor descriptor,
      std::vector<DeepSeekRequestIdentity> request_identities,
      std::vector<DeepSeekRankComputePlanWork> rank_work);
  Status prepare_native_pipeline(
      DeepSeekNativeRankComputePlan plan,
      std::vector<DeepSeekRequestIdentity> request_identities);
  Status prepare_deferred_native_pipeline(
      DeepSeekPipelinePlanDescriptor descriptor,
      std::vector<DeepSeekRequestIdentity> request_identities,
      DeepSeekDeferredRankComputePlanCompiler& compiler);
  Status prepare_production_pipeline(
      DeepSeekPipelinePlanDescriptor descriptor,
      std::vector<DeepSeekRequestIdentity> request_identities,
      std::vector<DeepSeekProductionRankPlanSeed> rank_seeds,
      std::uint32_t maximum_sequences, std::uint32_t maximum_tokens,
      std::uint32_t vocabulary_size,
      bool publish_sampled_token = false);
  Status prepare_bound_drain_pipeline(
      std::uint64_t plan_sequence,
      std::vector<DeepSeekRequestIdentity> request_identities);
  Status pipeline_stage_ready(std::uint32_t rank);
  Status commit_pipeline();
  Status stage_accepted_tokens(
      std::uint64_t plan_sequence,
      std::vector<DeepSeekAcceptedTokenBatch> batches);
  Status pipeline_stage_complete(std::uint32_t rank);
  Status advance_bound_pipeline();
  Status acknowledge_output_plan(std::uint64_t plan_sequence);
  Status pipeline_stage_failed(std::uint32_t rank, Status reason);
  Status poll_artifacts();
  Status report_mapping_fault(std::uint32_t rank);
  Status report_worker_lost(std::uint32_t rank);
  void fail_all() noexcept;
  Status begin_close() noexcept;
  Status advance_close() noexcept;
  Status close() noexcept;

  [[nodiscard]] DeepSeekEngineState state() const noexcept { return state_; }
  [[nodiscard]] DeepSeekEngineFailure failure() const noexcept {
    return failure_;
  }
  [[nodiscard]] std::uint64_t epoch() const noexcept {
    return resources_->epoch();
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return resources_->world_size();
  }
  [[nodiscard]] bool execution_live() const noexcept {
    return control_plane_->execution_live();
  }
  [[nodiscard]] std::size_t active_request_count() const noexcept {
    return control_plane_->active_request_count();
  }
  [[nodiscard]] std::string_view public_error() const noexcept;

 private:
  DeepSeekEngine(std::unique_ptr<DeepSeekEngineResources> resources,
                 std::unique_ptr<DeepSeekEngineArtifactPoller> artifact_poller,
                 std::unique_ptr<DeepSeekControlPlane> control_plane,
                 std::vector<DeepSeekPipelineResourceSet> rank_runtime_resources)
      : resources_(std::move(resources)),
        artifact_poller_(std::move(artifact_poller)),
        control_plane_(std::move(control_plane)),
        rank_runtime_resources_(std::move(rank_runtime_resources)) {}

  Status fail(DeepSeekEngineFailure failure, Status cause);
  Status abort_bound_compute_plans() noexcept;
  Status poll_artifacts_if_due();
  [[nodiscard]] Status validate_release_attention_binding(
      std::uint64_t request_id, std::uint64_t request_generation) const;
  Status release_attention_binding(std::uint64_t request_id,
                                   std::uint64_t request_generation);

  std::unique_ptr<DeepSeekEngineResources> resources_;
  std::unique_ptr<DeepSeekEngineArtifactPoller> artifact_poller_;
  std::unique_ptr<DeepSeekControlPlane> control_plane_;
  std::vector<DeepSeekPipelineResourceSet> rank_runtime_resources_;
  std::vector<DeepSeekRankPlanReservation> current_rank_plan_resources_;
  std::vector<DeepSeekRankPlanRuntime> current_rank_runtimes_;
  std::vector<std::shared_ptr<const DeepSeekRankComputePlanWorkOwner>>
      current_rank_work_owners_;
  std::vector<bool> rank_completion_reported_;
  struct RequestAttentionBinding final {
    std::uint64_t request_generation = 0;
    DeepSeekAttentionSequenceBinding binding;
  };
  std::unordered_map<std::uint64_t, RequestAttentionBinding>
      request_attention_bindings_;
  std::vector<std::uint32_t> free_attention_state_slots_;
  std::uint32_t next_attention_sequence_ = 1;
  bool bound_execution_ = false;
  bool publish_bound_sampled_token_ = false;
  bool bound_sampled_token_staged_ = false;
  std::optional<DeepSeekRequestSamplingConfig> bound_sampling_config_;
  std::optional<DeepSeekPreparedSamplingInput> bound_sampling_;
  DeepSeekEngineState state_ = DeepSeekEngineState::kReady;
  DeepSeekEngineFailure failure_ = DeepSeekEngineFailure::kNone;
  std::chrono::steady_clock::time_point next_artifact_poll_{};
};

}  // namespace pih
