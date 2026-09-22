#pragma once

#include <memory>
#include <vector>

#include "pih/model/deepseek_rank_engine_resources.h"
#include "pih/model/deepseek_rank_router_factory_set.h"
#include "pih/model/deepseek_learned_router_plan_input_assembler.h"
#include "pih/model/deepseek_hash_router_plan_input_assembler.h"
#include "pih/model/deepseek_endpoint_plan_input_assembler.h"
#include "pih/model/deepseek_endpoint_runtime_resources.h"
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
#include "pih/model/deepseek_dspark_plan_input_assembler.h"
#include "pih/model/deepseek_dspark_runtime_resources.h"
#endif
#include "pih/model/deepseek_native_plan_compiler.h"
#include "pih/model/deepseek_dense_mhc_stage_weight_bindings.h"
#include "pih/model/deepseek_request_input_staging_resources.h"
#include "pih/model/deepseek_production_rank_plan_input_assembler.h"

namespace pih {


class DeepSeekEngineResources final {
 public:
  static Result<DeepSeekEngineResources> Create(
      std::uint64_t epoch, std::uint32_t artifact_poll_interval_ms,
      std::vector<DeepSeekRankEngineResources> ranks);
  static Result<DeepSeekEngineResources> CreateOwned(
      std::uint64_t epoch, std::uint32_t artifact_poll_interval_ms,
      std::vector<std::unique_ptr<Allocator>> device_allocators,
      std::unique_ptr<RegisteredPinnedAllocator> pinned_allocator,
      std::vector<DeepSeekRankEngineResources> ranks);
  static Result<DeepSeekEngineResources> CreateOwnedWithRuntimes(
      std::uint64_t epoch, std::uint32_t artifact_poll_interval_ms,
      std::vector<std::unique_ptr<Allocator>> device_allocators,
      std::unique_ptr<RegisteredPinnedAllocator> pinned_allocator,
      std::vector<std::unique_ptr<DeepSeekRankRuntimeOwner>> runtime_owners,
      std::vector<DeepSeekRankEngineResources> ranks);

  DeepSeekEngineResources(const DeepSeekEngineResources&) = delete;
  DeepSeekEngineResources& operator=(const DeepSeekEngineResources&) = delete;
  DeepSeekEngineResources(DeepSeekEngineResources&&) noexcept = default;
  DeepSeekEngineResources& operator=(DeepSeekEngineResources&& other) noexcept;

  [[nodiscard]] bool admission_allowed() const noexcept {
    return guard_->admission_allowed() && barrier_->ready() &&
           all_compute_ready();
  }
  [[nodiscard]] std::uint64_t epoch() const noexcept { return guard_->epoch(); }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return guard_->world_size();
  }
  [[nodiscard]] std::uint32_t artifact_poll_interval_ms() const noexcept {
    return guard_->poll_interval_ms();
  }
  [[nodiscard]] DeepSeekRankEngineResources& rank(std::uint32_t rank) {
    return ranks_.at(rank);
  }
  [[nodiscard]] DeepSeekArtifactEpochGuard& artifact_guard() noexcept {
    return *guard_;
  }
  Status verify_weight_seals() const;
  Result<std::vector<DeepSeekRankPlanReservation>> prepare_rank_plan_resources(
      DeepSeekPipelinePlanDescriptor descriptor);
  Status bind_rank_compute_plans(
      DeepSeekPipelinePlanDescriptor descriptor,
      std::vector<DeepSeekRankComputePlanWork> work);
  Status abort_rank_compute_plans(
      const DeepSeekPipelinePlanDescriptor& descriptor) noexcept;
  Status activate_rank_runtime(std::uint32_t rank);
  Result<DeepSeekLearnedRouterOperations*> learned_router_operations(
      std::uint32_t rank) noexcept;
  Result<DeepSeekRankRouterFactorySet> assemble_rank_router_factories(
      std::uint32_t rank, std::uint32_t maximum_tokens,
      std::uint32_t vocabulary_size);
  Result<DeepSeekNativePlanCompiler> assemble_native_plan_compiler(
      std::uint32_t maximum_sequences, std::uint32_t maximum_tokens,
      std::uint32_t vocabulary_size);
  Result<DeepSeekDeferredNativePlanCompiler>
  assemble_production_deferred_plan_compiler(
      DeepSeekPipelinePlanDescriptor descriptor,
      std::vector<DeepSeekProductionRankPlanSeed> rank_seeds,
      std::uint32_t maximum_sequences, std::uint32_t maximum_tokens,
      std::uint32_t vocabulary_size);
  Result<std::vector<DeepSeekLearnedRouterLayerPlanWork>>
  assemble_rank_learned_router_plan_input(
      std::uint32_t rank, std::uint32_t token_count,
      std::span<const DeepSeekLearnedRouterLayerInput> layer_inputs);
  Result<std::vector<DeepSeekHashRouterLayerPlanWork>>
  assemble_rank_hash_router_plan_input(
      std::uint32_t rank, std::uint32_t token_count,
      std::span<const DeepSeekLearnedRouterLayerInput> layer_inputs);
  Result<DeepSeekDenseMhcStageSubmissions>
  assemble_rank_dense_mhc_stage_submissions(
      std::uint32_t rank, std::uint32_t token_count,
      std::uintptr_t initial_residual_bf16,
      std::uint32_t table_position_count);
  Result<DeepSeekRequestInputStagingLease> stage_rank_request_inputs(
      std::uint32_t rank, std::span<const std::uint32_t> token_ids,
      std::span<const std::uint32_t> positions);
  Result<std::uintptr_t> rank_request_token_ids_u32(std::uint32_t rank);
  Result<std::uintptr_t> rank_initial_residual_bf16(std::uint32_t rank);
  Result<DeepSeekEndpointStageSequenceWork>
  assemble_rank_endpoint_plan_input(
      std::uint32_t rank, std::uint32_t token_count,
      std::uintptr_t token_ids_u32, std::uintptr_t final_hc_bf16,
      std::optional<DeepSeekPreparedSamplingInput> sampling = std::nullopt);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  Result<DeepSeekDsparkStageWork> assemble_rank_dspark_plan_input(
      std::uint32_t rank, DeepSeekPlanPhase phase,
      std::uint32_t token_count, std::uintptr_t input_token_ids_u32);
  Result<std::vector<DeepSeekBoundDsparkMtpStageWork>>
  assemble_rank_dspark_prefill_mtp_plan_input(
      std::uint32_t rank, std::uint32_t token_count,
      std::uint32_t position_table_count);
  Result<std::vector<DeepSeekBoundDsparkMtpStageWork>>
  assemble_rank_dspark_decode_mtp_plan_input(
      std::uint32_t rank, std::uint32_t current_position,
      std::uint32_t position_table_count);
#endif
  [[nodiscard]] bool execution_topology_ready() const noexcept {
    return world_size() == 1;
  }

 private:
  [[nodiscard]] bool all_compute_ready() const noexcept;
  DeepSeekEngineResources(
      std::unique_ptr<DeepSeekArtifactEpochGuard> guard,
      std::unique_ptr<DeepSeekEngineBootstrapBarrier> barrier,
      std::vector<std::unique_ptr<Allocator>> device_allocators,
      std::unique_ptr<RegisteredPinnedAllocator> pinned_allocator,
      std::vector<std::unique_ptr<DeepSeekRankRuntimeOwner>> runtime_owners,
      std::vector<DeepSeekRankEngineResources> ranks)
      : guard_(std::move(guard)), barrier_(std::move(barrier)),
        device_allocators_(std::move(device_allocators)),
        pinned_allocator_(std::move(pinned_allocator)),
        runtime_owners_(std::move(runtime_owners)),
        ranks_(std::move(ranks)) {}

  static Result<DeepSeekEngineResources> CreateImpl(
      std::uint64_t epoch, std::uint32_t artifact_poll_interval_ms,
      std::vector<std::unique_ptr<Allocator>> device_allocators,
      std::unique_ptr<RegisteredPinnedAllocator> pinned_allocator,
      bool require_owned_allocators,
      std::vector<std::unique_ptr<DeepSeekRankRuntimeOwner>> runtime_owners,
      bool require_runtime_owners,
      std::vector<DeepSeekRankEngineResources> ranks);

  // Declaration order ensures ranks release before the barrier and guard.
  std::unique_ptr<DeepSeekArtifactEpochGuard> guard_;
  std::unique_ptr<DeepSeekEngineBootstrapBarrier> barrier_;
  // Allocators outlive ranks because members are destroyed in reverse order.
  std::vector<std::unique_ptr<Allocator>> device_allocators_;
  std::unique_ptr<RegisteredPinnedAllocator> pinned_allocator_;
  // Runtime contexts outlive every rank-owned device allocation.
  std::vector<std::unique_ptr<DeepSeekRankRuntimeOwner>> runtime_owners_;
  std::vector<DeepSeekRankEngineResources> ranks_;
};

}  // namespace pih
