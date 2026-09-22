#include "pih/model/deepseek_rank_engine_resources.h"

#include <new>
#include <utility>

namespace pih {

DeepSeekRankEngineResources& DeepSeekRankEngineResources::operator=(
    DeepSeekRankEngineResources&& other) noexcept {
  if (this != &other) {
    // Releasing members one-by-one in declaration order would drop mapping
    // and allocator owners before the objects that borrow from them. Preserve
    // reverse destruction order, then rebuild the complete ownership graph.
    this->~DeepSeekRankEngineResources();
    ::new (static_cast<void*>(this))
        DeepSeekRankEngineResources(std::move(other));
  }
  return *this;
}

Result<DeepSeekRankEngineResources>
DeepSeekRankEngineResources::BuildForInProcessDevelopment(
    std::uint64_t epoch, std::uint32_t world_size,
    DeepSeekStageRange owned_layers,
    DeepSeekRoutedExpertResidency expert_residency,
    DeepSeekRankMappingPlan mapping_plan,
    std::vector<DeepSeekCapabilityShardLease> leases,
    std::vector<DeepSeekRankTensorRecord> rank_tensors,
    std::uint32_t expert_slot_count, std::uint32_t staging_extent_count,
    const DeepSeekPipelineCapacity& pipeline_capacity,
    Allocator& device_allocator, RegisteredPinnedAllocator& pinned_allocator,
    MemoryCopier& copier, DeepSeekWeightConsumerDryRun& dry_run,
    std::uint64_t host_spill_pinned_bytes,
    std::uint64_t host_spill_device_bytes,
    std::uint32_t transfer_reservation_window,
    RegisteredPinnedAllocator* host_spill_pinned_allocator) {
  if (epoch == 0 || world_size != 1 || pipeline_capacity.dspark_enabled ||
      mapping_plan.rank >= world_size || rank_tensors.empty() ||
      pipeline_capacity.world_size != world_size) {
    return Status::InvalidArgument(
        "DeepSeek rank engine build identity is invalid");
  }
  auto inventory_value =
      DeepSeekStageMappedInventory::CreateCapabilityBacked(
          mapping_plan, std::move(leases));
  if (!inventory_value.ok()) return inventory_value.status();
  auto inventory = std::make_unique<DeepSeekStageMappedInventory>(
      std::move(*inventory_value));

  auto tensor_source_value = DeepSeekMappedTensorSource::Create(
      *inventory, rank_tensors);
  if (!tensor_source_value.ok()) return tensor_source_value.status();
  auto tensor_source = std::make_unique<DeepSeekMappedTensorSource>(
      std::move(*tensor_source_value));

  return BuildFromMappedSources(
      epoch, world_size, owned_layers, expert_residency,
      std::move(inventory), std::move(tensor_source),
      std::move(rank_tensors), expert_slot_count, staging_extent_count,
      pipeline_capacity, device_allocator, pinned_allocator, copier,
      dry_run, host_spill_pinned_bytes, host_spill_device_bytes,
      transfer_reservation_window, host_spill_pinned_allocator);
}

Result<DeepSeekRankEngineResources>
DeepSeekRankEngineResources::BuildFromMappedSources(
    std::uint64_t epoch, std::uint32_t world_size,
    DeepSeekStageRange owned_layers,
    DeepSeekRoutedExpertResidency expert_residency,
    std::unique_ptr<DeepSeekStageMappedInventory> development_inventory,
    std::unique_ptr<DeepSeekMappedTensorSource> development_tensor_source,
    std::vector<DeepSeekRankTensorRecord> rank_tensors,
    std::uint32_t expert_slot_count, std::uint32_t staging_extent_count,
    const DeepSeekPipelineCapacity& pipeline_capacity,
    Allocator& device_allocator, RegisteredPinnedAllocator& pinned_allocator,
    MemoryCopier& copier, DeepSeekWeightConsumerDryRun& dry_run,
    std::uint64_t host_spill_pinned_bytes,
    std::uint64_t host_spill_device_bytes,
    std::uint32_t transfer_reservation_window,
    RegisteredPinnedAllocator* host_spill_pinned_allocator) {
  const bool host_spill =
      expert_residency == DeepSeekRoutedExpertResidency::kHostSpill;
  const auto effective_transfer_reservation_window =
      transfer_reservation_window;
  const auto expected_pinned_bytes =
      static_cast<std::uint64_t>(staging_extent_count) *
      DeepSeekExpertPager::kBundleBytes;
  const auto expected_device_bytes =
      static_cast<std::uint64_t>(expert_slot_count) *
      DeepSeekExpertPager::kBundleBytes;
  const bool host_spill_budget_valid =
      !host_spill ||
      (host_spill_pinned_bytes == expected_pinned_bytes &&
       host_spill_device_bytes == expected_device_bytes &&
       host_spill_pinned_allocator != nullptr);
  if (development_inventory == nullptr || development_tensor_source == nullptr ||
      epoch == 0 || world_size != 1 || pipeline_capacity.dspark_enabled ||
      rank_tensors.empty() ||
      pipeline_capacity.world_size != world_size ||
      !host_spill_budget_valid ||
      (!host_spill &&
       (host_spill_pinned_bytes != 0 || host_spill_device_bytes != 0 ||
        transfer_reservation_window != 0))) {
    return Status::InvalidArgument(
        "DeepSeek mapped rank materialization identity is invalid");
  }
  const auto& inventory = *development_inventory;
  const auto& tensor_source = *development_tensor_source;
  const auto rank = inventory.rank();
  if (rank != 0) {
    return Status::FailedPrecondition("DeepSeek PP1 rank must be zero");
  }
  const DeepSeekStagePlan stage{
      rank, owned_layers, rank == 0, rank + 1 == world_size,
      pipeline_capacity.dspark_enabled && rank + 1 == world_size};
  if ((host_spill &&
       (effective_transfer_reservation_window == 0 ||
        expert_slot_count < effective_transfer_reservation_window ||
        staging_extent_count < effective_transfer_reservation_window)) ||
      (!host_spill &&
       (expert_slot_count != 0 || staging_extent_count != 0))) {
    return Status::InvalidArgument(
        "DeepSeek rank expert resource topology is invalid");
  }

  auto expert_manifest_value = DeepSeekExpertBundleManifest::Create(
      owned_layers, rank_tensors);
  if (!expert_manifest_value.ok()) return expert_manifest_value.status();
  auto expert_manifest = std::make_unique<DeepSeekExpertBundleManifest>(
      std::move(*expert_manifest_value));

  auto disposition_value = DeepSeekRankWeightDispositionManifest::Create(
      rank, owned_layers, expert_residency, rank_tensors);
  if (!disposition_value.ok()) return disposition_value.status();
  auto disposition =
      std::make_unique<DeepSeekRankWeightDispositionManifest>(
          std::move(*disposition_value));

  auto materialization_value = DeepSeekWeightMaterializationPlan::Create(
      *disposition, *expert_manifest, rank_tensors);
  if (!materialization_value.ok()) return materialization_value.status();
  auto materialization = std::make_unique<DeepSeekWeightMaterializationPlan>(
      std::move(*materialization_value));

  auto arena_value = DeepSeekResidentWeightArena::Load(
      *materialization, tensor_source, device_allocator, copier);
  if (!arena_value.ok()) return arena_value.status();
  if (arena_value->device().type() != DeviceType::kCuda ||
      arena_value->device().index() < 0) {
    return Status::FailedPrecondition(
        "DeepSeek resident weights are not on a CUDA device");
  }
  auto arena = std::make_unique<DeepSeekResidentWeightArena>(
      std::move(*arena_value));

  auto finalizer_value = DeepSeekWeightFinalizer::Create(
      *materialization, *arena);
  if (!finalizer_value.ok()) return finalizer_value.status();
  auto finalizer = std::make_unique<DeepSeekWeightFinalizer>(
      std::move(*finalizer_value));
  auto status = finalizer->run_dry_run(dry_run);
  if (!status.ok()) return status;
  status = finalizer->seal();
  if (!status.ok()) return status;

  std::unique_ptr<DeepSeekExpertBundleStagingPool> staging_pool;
  std::unique_ptr<DeepSeekMappedExpertSource> expert_source;
  std::unique_ptr<DeepSeekExpertPager> pager;
  if (host_spill) {
    auto& staging_allocator = host_spill_pinned_allocator == nullptr
        ? pinned_allocator
        : *host_spill_pinned_allocator;
    auto staging_value = DeepSeekExpertBundleStagingPool::Allocate(
        staging_extent_count, staging_allocator,
        effective_transfer_reservation_window);
    if (!staging_value.ok()) return staging_value.status();
    staging_pool = std::make_unique<DeepSeekExpertBundleStagingPool>(
        std::move(*staging_value));
    std::vector<DeepSeekPinnedExpertExtent> extents(
        staging_pool->extents().begin(), staging_pool->extents().end());
    auto source_value = DeepSeekMappedExpertSource::Create(
        inventory, owned_layers, expert_manifest->bundles(),
        std::move(extents), effective_transfer_reservation_window);
    if (!source_value.ok()) return source_value.status();
    expert_source = std::make_unique<DeepSeekMappedExpertSource>(
        std::move(*source_value));
    auto pager_value = DeepSeekExpertPager::Create(
        owned_layers, expert_slot_count, staging_extent_count,
        effective_transfer_reservation_window);
    if (!pager_value.ok()) return pager_value.status();
    pager = std::make_unique<DeepSeekExpertPager>(std::move(*pager_value));
  }

  auto receipt = DeepSeekRankBootstrapReceipt::Create(
      epoch, world_size, inventory, *materialization, *finalizer,
      expert_residency, pager.get());
  if (!receipt.ok()) return receipt.status();
  std::array<DeepSeekRankPlanResourceCapacity,
             DeepSeekRankPlanResourcePool::kKindCount> plan_capacities{{
      {DeepSeekRankPlanResourceKind::kIncomingBoundary,
       rank == 0 ? 0 : DeepSeekPipelineCapacity::kBoundaryCredits},
      {DeepSeekRankPlanResourceKind::kOutgoingActivation,
       rank + 1 == world_size ? 0 : DeepSeekPipelineCapacity::kBoundaryCredits},
      {DeepSeekRankPlanResourceKind::kControlSlot,
       DeepSeekPipelineCapacity::kBoundaryCredits},
      {DeepSeekRankPlanResourceKind::kAttentionStateTransaction,
       pipeline_capacity.max_sequences},
      {DeepSeekRankPlanResourceKind::kVerifyPrefixWorkspaceBytes,
       pipeline_capacity.boundary_slot_bytes},
      {DeepSeekRankPlanResourceKind::kOperatorWorkspaceBytes,
       pipeline_capacity.boundary_slot_bytes},
      {DeepSeekRankPlanResourceKind::kExpertWorkspaceBytes,
       pipeline_capacity.expert_workspace_bytes},
      {DeepSeekRankPlanResourceKind::kPagerForwardProgressSlot,
       expert_slot_count},
      {DeepSeekRankPlanResourceKind::kPinnedStagingCredit,
       staging_extent_count},
      {DeepSeekRankPlanResourceKind::kOutputSamplingCredit,
       rank + 1 == world_size ? pipeline_capacity.max_sequences : 0},
  }};
  auto plan_resource_pool = DeepSeekRankPlanResourcePool::Create(
      epoch, rank, world_size, plan_capacities);
  if (!plan_resource_pool.ok()) return plan_resource_pool.status();
  return DeepSeekRankEngineResources(
      std::move(development_inventory),
      std::move(development_tensor_source),
      std::move(expert_manifest), std::move(disposition),
      std::move(materialization),
      std::move(arena), std::move(finalizer),
      std::move(staging_pool), std::move(expert_source), std::move(pager),
      std::move(*plan_resource_pool), stage, expert_residency,
      pipeline_capacity.boundary_slot_bytes,
      pipeline_capacity.boundary_slot_bytes,
      pipeline_capacity.expert_workspace_bytes,
      std::move(*receipt));
}

Result<DeepSeekRankPlanReservation>
DeepSeekRankEngineResources::prepare_plan_resources(
    DeepSeekPipelinePlanDescriptor descriptor) {
  return DeepSeekRankPlanReservation::Prepare(
      descriptor, stage_,
      receipt_.world_size(), expert_residency_,
      descriptor.phase == DeepSeekPlanPhase::kVerify
          ? verify_prefix_workspace_bytes_
          : 0,
      operator_workspace_bytes_, expert_workspace_bytes_,
      *plan_resource_pool_);
}

Status DeepSeekRankEngineResources::attach_compute_lanes(
    std::unique_ptr<DeepSeekRankComputeLaneSet> lanes) {
  const bool pager_identity_matches =
      expert_residency_ == DeepSeekRoutedExpertResidency::kHostSpill
          ? pager_ != nullptr && lanes != nullptr &&
                lanes->pager_identity() == pager_.get()
          : pager_ == nullptr && lanes != nullptr &&
                lanes->pager_identity() == nullptr;
  if (lanes == nullptr || compute_lanes_ != nullptr ||
      !pager_identity_matches) {
    return Status::FailedPrecondition(
        "DeepSeek rank compute lanes cannot be attached to these resources");
  }
  const auto lane_stage = lanes->stage();
  if (lane_stage.rank != stage_.rank ||
      lane_stage.layers.first_layer != stage_.layers.first_layer ||
      lane_stage.layers.last_layer != stage_.layers.last_layer ||
      lane_stage.owns_embedding != stage_.owns_embedding ||
      lane_stage.owns_lm_head != stage_.owns_lm_head ||
      lane_stage.owns_dspark != stage_.owns_dspark) {
    return Status::FailedPrecondition(
        "DeepSeek rank compute lane stage identity differs from resources");
  }
  compute_lanes_ = std::move(lanes);
  return Status::Ok();
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Result<const DeepSeekDsparkWeightBindings*>
DeepSeekRankEngineResources::dspark_weight_bindings() {
  if (!stage_.owns_dspark || arena_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek rank does not own DSpark weights");
  }
  if (dspark_weight_bindings_ == nullptr) {
    auto resolved = DeepSeekDsparkWeightBindings::Resolve(stage_, *arena_);
    if (!resolved.ok()) return resolved.status();
    dspark_weight_bindings_ =
        std::make_unique<DeepSeekDsparkWeightBindings>(
            std::move(*resolved));
  }
  return dspark_weight_bindings_.get();
}
#endif

Status DeepSeekRankEngineResources::bind_compute_plan(
    DeepSeekPipelinePlanDescriptor descriptor,
    DeepSeekRankComputePlanWork work) {
  if (compute_lanes_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek rank has no compute lanes for plan binding");
  }
  return compute_lanes_->bundle().bind_plan(descriptor, work);
}

Status DeepSeekRankEngineResources::abort_compute_plan(
    const DeepSeekPipelinePlanDescriptor& descriptor) noexcept {
  if (compute_lanes_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek rank has no compute lanes for plan abort");
  }
  return compute_lanes_->bundle().abort_bound_plan(descriptor);
}

bool DeepSeekRankEngineResources::can_abort_compute_plan(
    const DeepSeekPipelinePlanDescriptor& descriptor) const noexcept {
  return compute_lanes_ != nullptr &&
         compute_lanes_->bundle().can_abort_bound_plan(descriptor);
}

}  // namespace pih
