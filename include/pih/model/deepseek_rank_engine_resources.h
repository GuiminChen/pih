#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "pih/model/deepseek_engine_bootstrap_barrier.h"
#include "pih/model/deepseek_expert_bundle_staging_pool.h"
#include "pih/model/deepseek_mapped_expert_source.h"
#include "pih/model/deepseek_mapped_tensor_source.h"
#include "pih/model/deepseek_rank_plan_resource_pool.h"
#include "pih/model/deepseek_rank_compute_lane_set.h"
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
#include "pih/model/deepseek_dspark_weight_bindings.h"
#endif

namespace pih {

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
class DeepSeekDsparkWeightBindings;
#endif

class DeepSeekRankEngineResources final {
 public:
  // Native PP1 resources retain verified-artifact capability leases. This
  // owner does not accept cross-process materialization grants or raw FDs.
  static Result<DeepSeekRankEngineResources>
  BuildForInProcessDevelopment(
      std::uint64_t epoch, std::uint32_t world_size,
      DeepSeekStageRange owned_layers,
      DeepSeekRoutedExpertResidency expert_residency,
      DeepSeekRankMappingPlan mapping_plan,
      std::vector<DeepSeekCapabilityShardLease> leases,
      std::vector<DeepSeekRankTensorRecord> rank_tensors,
      std::uint32_t expert_slot_count, std::uint32_t staging_extent_count,
      const DeepSeekPipelineCapacity& pipeline_capacity,
      Allocator& device_allocator,
      RegisteredPinnedAllocator& pinned_allocator, MemoryCopier& copier,
      DeepSeekWeightConsumerDryRun& dry_run,
      std::uint64_t host_spill_pinned_bytes = 0,
      std::uint64_t host_spill_device_bytes = 0,
      std::uint32_t transfer_reservation_window = 0,
      RegisteredPinnedAllocator* host_spill_pinned_allocator = nullptr);

  DeepSeekRankEngineResources(const DeepSeekRankEngineResources&) = delete;
  DeepSeekRankEngineResources& operator=(const DeepSeekRankEngineResources&) = delete;
  DeepSeekRankEngineResources(DeepSeekRankEngineResources&&) noexcept = default;
  DeepSeekRankEngineResources& operator=(
      DeepSeekRankEngineResources&& other) noexcept;

  [[nodiscard]] std::uint32_t rank() const noexcept {
    return mapped_inventory().rank();
  }
  [[nodiscard]] const DeepSeekRankBootstrapReceipt& receipt() const noexcept {
    return receipt_;
  }
  [[nodiscard]] const DeepSeekResidentWeightArena& resident_weights()
      const noexcept { return *arena_; }
  [[nodiscard]] DeepSeekExpertPager* expert_pager() noexcept {
    return pager_.get();
  }
  [[nodiscard]] const DeepSeekExpertPager* expert_pager() const noexcept {
    return pager_.get();
  }
  [[nodiscard]] DeepSeekRoutedExpertResidency expert_residency()
      const noexcept {
    return expert_residency_;
  }
  [[nodiscard]] std::uint64_t pinned_staging_bytes() const noexcept {
    return staging_pool_ == nullptr ? 0 : staging_pool_->bytes();
  }
  [[nodiscard]] std::uint64_t pinned_staging_generation() const noexcept {
    return staging_pool_ == nullptr ? 0 : staging_pool_->generation();
  }
  [[nodiscard]] std::uint32_t staging_extent_count() const noexcept {
    return staging_pool_ == nullptr ? 0 : staging_pool_->extent_count();
  }
  [[nodiscard]] DeepSeekMappedExpertSource* expert_source() noexcept {
    return expert_source_.get();
  }
  [[nodiscard]] const DeepSeekWeightByteSource& weight_source() const noexcept {
    return mapped_tensor_source();
  }
  Status verify_weight_seal() const { return finalizer_->verify_unchanged(); }
  Result<DeepSeekRankPlanReservation> prepare_plan_resources(
      DeepSeekPipelinePlanDescriptor descriptor);
  Status attach_compute_lanes(
      std::unique_ptr<DeepSeekRankComputeLaneSet> lanes);
  [[nodiscard]] bool compute_ready() const noexcept {
    return compute_lanes_ != nullptr;
  }
  [[nodiscard]] DeepSeekRankComputeBundle* compute_bundle() noexcept {
    return compute_lanes_ == nullptr ? nullptr : &compute_lanes_->bundle();
  }
  [[nodiscard]] DeepSeekRankAttentionResources* attention_resources()
      noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->attention_resources();
  }
  [[nodiscard]] DeepSeekRankAttentionStatePool* attention_state_pool()
      noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->attention_state_pool();
  }
  [[nodiscard]] DeepSeekLearnedRouterDeviceResources*
  learned_router_device_resources() noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->learned_router_device_resources();
  }
  [[nodiscard]] DeepSeekLearnedRouterStagingPool*
  learned_router_staging_pool() noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->learned_router_staging_pool();
  }
  [[nodiscard]] DeepSeekHashRouterStagingPool*
  hash_router_staging_pool() noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->hash_router_staging_pool();
  }
  [[nodiscard]] DeepSeekEndpointRuntimeResources*
  endpoint_runtime_resources() noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->endpoint_runtime_resources();
  }
  [[nodiscard]] DeepSeekEndpointDeviceResources*
  endpoint_device_resources() noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->endpoint_device_resources();
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  [[nodiscard]] DeepSeekDsparkRuntimeResources*
  dspark_runtime_resources() noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->dspark_runtime_resources();
  }
  [[nodiscard]] DeepSeekDsparkDeviceResources*
  dspark_device_resources() noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->dspark_device_resources();
  }
#endif
  [[nodiscard]] DeepSeekDenseMhcRuntimeResources*
  dense_mhc_runtime_resources() noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->dense_mhc_runtime_resources();
  }
  [[nodiscard]] DeepSeekAttentionProjectionDeviceResources*
  attention_projection_device_resources() noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->attention_projection_device_resources();
  }
  [[nodiscard]] DeepSeekMhcDeviceResources* mhc_device_resources() noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->mhc_device_resources();
  }
  [[nodiscard]] DeepSeekRopeTableDeviceResources*
  rope_table_device_resources() noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->rope_table_device_resources();
  }
  [[nodiscard]] DeepSeekRequestInputStagingResources*
  request_input_staging_resources() noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->request_input_staging_resources();
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  [[nodiscard]] DeepSeekDsparkExpertKernelDriver*
  dspark_expert_kernel() noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->dspark_expert_kernel();
  }
#endif
  [[nodiscard]] DeepSeekExpertComputeArena expert_compute_arena()
      const noexcept {
    return compute_lanes_ == nullptr
               ? DeepSeekExpertComputeArena{}
               : compute_lanes_->expert_compute_arena();
  }
  [[nodiscard]] std::uintptr_t expert_accumulator_f32() const noexcept {
    return compute_lanes_ == nullptr
               ? 0
               : compute_lanes_->expert_accumulator_f32();
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  [[nodiscard]] const DeepSeekDsparkResidentExpertBindings*
  dspark_resident_expert_bindings() const noexcept {
    return compute_lanes_ == nullptr
               ? nullptr
               : compute_lanes_->dspark_resident_expert_bindings();
  }
  // The pointee is heap-owned so its address remains stable if the rank or
  // enclosing engine resource is moved while a compiled plan is retained.
  Result<const DeepSeekDsparkWeightBindings*> dspark_weight_bindings();
#endif
  Status bind_compute_plan(DeepSeekPipelinePlanDescriptor descriptor,
                           DeepSeekRankComputePlanWork work);
  Status abort_compute_plan(
      const DeepSeekPipelinePlanDescriptor& descriptor) noexcept;
  [[nodiscard]] bool can_abort_compute_plan(
      const DeepSeekPipelinePlanDescriptor& descriptor) const noexcept;

 private:
  static Result<DeepSeekRankEngineResources> BuildFromMappedSources(
      std::uint64_t epoch, std::uint32_t world_size,
      DeepSeekStageRange owned_layers,
      DeepSeekRoutedExpertResidency expert_residency,
      std::unique_ptr<DeepSeekStageMappedInventory> development_inventory,
      std::unique_ptr<DeepSeekMappedTensorSource>
          development_tensor_source,
      std::vector<DeepSeekRankTensorRecord> rank_tensors,
      std::uint32_t expert_slot_count, std::uint32_t staging_extent_count,
      const DeepSeekPipelineCapacity& pipeline_capacity,
      Allocator& device_allocator,
      RegisteredPinnedAllocator& pinned_allocator, MemoryCopier& copier,
      DeepSeekWeightConsumerDryRun& dry_run,
      std::uint64_t host_spill_pinned_bytes,
      std::uint64_t host_spill_device_bytes,
      std::uint32_t transfer_reservation_window,
      RegisteredPinnedAllocator* host_spill_pinned_allocator);

  DeepSeekRankEngineResources(
      std::unique_ptr<DeepSeekStageMappedInventory> inventory,
      std::unique_ptr<DeepSeekMappedTensorSource> tensor_source,
      std::unique_ptr<DeepSeekExpertBundleManifest> expert_manifest,
      std::unique_ptr<DeepSeekRankWeightDispositionManifest> disposition,
      std::unique_ptr<DeepSeekWeightMaterializationPlan> materialization,
      std::unique_ptr<DeepSeekResidentWeightArena> arena,
      std::unique_ptr<DeepSeekWeightFinalizer> finalizer,
      std::unique_ptr<DeepSeekExpertBundleStagingPool> staging_pool,
      std::unique_ptr<DeepSeekMappedExpertSource> expert_source,
      std::unique_ptr<DeepSeekExpertPager> pager,
      std::unique_ptr<DeepSeekRankPlanResourcePool> plan_resource_pool,
      DeepSeekStagePlan stage,
      DeepSeekRoutedExpertResidency expert_residency,
      std::uint64_t verify_prefix_workspace_bytes,
      std::uint64_t operator_workspace_bytes,
      std::uint64_t expert_workspace_bytes,
      DeepSeekRankBootstrapReceipt receipt)
      : inventory_(std::move(inventory)),
        tensor_source_(std::move(tensor_source)),
        expert_manifest_(std::move(expert_manifest)),
        disposition_(std::move(disposition)),
        materialization_(std::move(materialization)),
        arena_(std::move(arena)),
        finalizer_(std::move(finalizer)), staging_pool_(std::move(staging_pool)),
        expert_source_(std::move(expert_source)), pager_(std::move(pager)),
        plan_resource_pool_(std::move(plan_resource_pool)),
        stage_(stage),
        expert_residency_(expert_residency),
        verify_prefix_workspace_bytes_(verify_prefix_workspace_bytes),
        operator_workspace_bytes_(operator_workspace_bytes),
        expert_workspace_bytes_(expert_workspace_bytes),
        receipt_(std::move(receipt)) {}

  [[nodiscard]] const DeepSeekStageMappedInventory& mapped_inventory()
      const noexcept {
    return *inventory_;
  }
  [[nodiscard]] const DeepSeekMappedTensorSource& mapped_tensor_source()
      const noexcept {
    return *tensor_source_;
  }

  std::unique_ptr<DeepSeekStageMappedInventory> inventory_;
  std::unique_ptr<DeepSeekMappedTensorSource> tensor_source_;
  std::unique_ptr<DeepSeekExpertBundleManifest> expert_manifest_;
  std::unique_ptr<DeepSeekRankWeightDispositionManifest> disposition_;
  std::unique_ptr<DeepSeekWeightMaterializationPlan> materialization_;
  std::unique_ptr<DeepSeekResidentWeightArena> arena_;
  std::unique_ptr<DeepSeekWeightFinalizer> finalizer_;
  std::unique_ptr<DeepSeekExpertBundleStagingPool> staging_pool_;
  std::unique_ptr<DeepSeekMappedExpertSource> expert_source_;
  std::unique_ptr<DeepSeekExpertPager> pager_;
  std::unique_ptr<DeepSeekRankPlanResourcePool> plan_resource_pool_;
  DeepSeekStagePlan stage_;
  DeepSeekRoutedExpertResidency expert_residency_ =
      DeepSeekRoutedExpertResidency::kHostSpill;
  std::uint64_t verify_prefix_workspace_bytes_ = 0;
  std::uint64_t operator_workspace_bytes_ = 0;
  std::uint64_t expert_workspace_bytes_ = 0;
  DeepSeekRankBootstrapReceipt receipt_;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::unique_ptr<DeepSeekDsparkWeightBindings> dspark_weight_bindings_;
#endif
  // Declared last so compute drivers are destroyed before pager/weights.
  std::unique_ptr<DeepSeekRankComputeLaneSet> compute_lanes_;
};

}  // namespace pih
