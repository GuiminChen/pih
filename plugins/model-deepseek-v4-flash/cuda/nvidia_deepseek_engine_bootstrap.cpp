#include "pih/backend/cuda/nvidia_deepseek_engine_bootstrap.h"

#include <algorithm>
#include <optional>
#include <utility>

#include "pih/backend/cuda/nvidia_deepseek_rank_runtime.h"
#include "pih/model/deepseek_rank_compute_infrastructure.h"
#include "pih/model/deepseek_weight_binding_dry_run.h"

namespace pih {
namespace {

class CatalogArtifactPoller final : public DeepSeekEngineArtifactPoller {
 public:
  explicit CatalogArtifactPoller(
      DeepSeekCapabilityArtifactCatalog catalog,
      std::shared_ptr<const RuntimeEngineAdmission> admission_anchor = {})
      : catalog_(std::move(catalog)),
        admission_anchor_(std::move(admission_anchor)) {}
  std::uint32_t world_size() const noexcept override {
    return catalog_.mapping_plan().world_size();
  }
  Status poll() override { return catalog_.poll_leases(); }

 private:
  DeepSeekCapabilityArtifactCatalog catalog_;
  // Retains immutable profile/reference authority for the complete engine
  // epoch. The catalog independently retains target-generation descriptors.
  std::shared_ptr<const RuntimeEngineAdmission> admission_anchor_;
};

Status validate_dspark_runtime_gate(bool requested, bool capacity_enabled,
                                    bool permitted = false) {
  if (requested != capacity_enabled) {
    return Status::InvalidArgument(
        "DeepSeek DSpark request and capacity disagree");
  }
  if (requested && !permitted) {
    return Status::FailedPrecondition(
        "deepseek_dspark_runtime_permit_required");
  }
  return Status::Ok();
}

bool nonzero_digest(const Sha256Digest& value) {
  return std::ranges::any_of(
      value.bytes, [](std::byte byte) { return byte != std::byte{0}; });
}

Result<DeepSeekGpuArchitecture> admitted_deepseek_architecture(
    RuntimeProfileGpuFamily family) {
  switch (family) {
    case RuntimeProfileGpuFamily::kRtx4090D24GiB:
      return DeepSeekGpuArchitecture::kSm89;
    case RuntimeProfileGpuFamily::kH100Pcie80GiB:
      return DeepSeekGpuArchitecture::kSm90;
  }
  return Status::FailedPrecondition(
      "DeepSeek rank runtime requires an admitted SM89 or SM90 GPU family");
}

Status validate_production_admission_axes(
    const RuntimeEngineAdmission& admission,
    const DeepSeekPipelinePlan& pipeline,
    const NvidiaDeepSeekBootstrapConfig& config, bool dspark_enabled) {
  const auto residency =
      config.expert_residency() == DeepSeekRoutedExpertResidency::kHostSpill
          ? RuntimeProfileResidency::kHostSpill
          : RuntimeProfileResidency::kFullResident;
  if (admission.model() != RuntimeProfileModel::kDeepSeekV4Flash0731 ||
      admission.weight_format() !=
          RuntimeProfileWeightFormat::kDeepSeekNative ||
      !admission.evidence_projection_bound() ||
      admission.dspark_enabled() != dspark_enabled ||
      (dspark_enabled && admission.gpu_family() !=
                              RuntimeProfileGpuFamily::kH100Pcie80GiB) ||
      admission.device_ordinals().size() != config.world_size() ||
      pipeline.world_size() != config.world_size() ||
      !std::ranges::equal(admission.device_ordinals(),
                          config.device_ordinals()) ||
      admission.residency() != residency) {
    return Status::FailedPrecondition(
        "DeepSeek production bootstrap differs from admitted runtime profile");
  }
  return Status::Ok();
}

}  // namespace

Result<std::unique_ptr<DeepSeekEngine>> BuildCore(
    DeepSeekCapabilityArtifactCatalog catalog, const DeepSeekPipelinePlan& pipeline,
    DeepSeekPipelineCapacity capacity,
    const NvidiaDeepSeekBootstrapConfig& config,
    bool dspark_permitted,
    std::shared_ptr<const RuntimeEngineAdmission> admission_anchor,
    NvidiaDeepSeekRankRuntimeFactory& runtime_factory);

Result<NvidiaDeepSeekBootstrapConfig> NvidiaDeepSeekBootstrapConfig::Create(
    std::uint64_t epoch, std::uint32_t world_size,
    std::vector<std::int32_t> device_ordinals,
    DeepSeekRoutedExpertResidency expert_residency,
    std::uint32_t expert_slot_count,
    std::uint32_t staging_extent_count,
    std::uint32_t artifact_poll_interval_ms,
    std::uint32_t attention_reserved_tokens_per_sequence,
    DeepSeekExecutionTopology execution_topology,
    std::uint64_t host_spill_pinned_bytes,
    std::uint64_t host_spill_device_bytes,
    std::uint32_t transfer_reservation_window) {
  if (epoch == 0 || world_size != 1 ||
      device_ordinals.size() != world_size || artifact_poll_interval_ms == 0 ||
      attention_reserved_tokens_per_sequence == 0 ||
      attention_reserved_tokens_per_sequence > 1048576 ||
      deepseek_execution_topology_name(execution_topology).empty() ||
      std::ranges::any_of(device_ordinals,
                          [](std::int32_t ordinal) { return ordinal < 0; })) {
    return Status::InvalidArgument("DeepSeek NVIDIA bootstrap topology is invalid");
  }
  auto ordered = device_ordinals;
  std::ranges::sort(ordered);
  if (std::adjacent_find(ordered.begin(), ordered.end()) != ordered.end()) {
    return Status::InvalidArgument("DeepSeek NVIDIA device ordinal is duplicated");
  }
  const bool spill =
      expert_residency == DeepSeekRoutedExpertResidency::kHostSpill;
  const bool resident =
      expert_residency == DeepSeekRoutedExpertResidency::kFullResident;
  const auto effective_transfer_reservation_window =
      transfer_reservation_window;
  const auto expected_pinned_bytes =
      static_cast<std::uint64_t>(staging_extent_count) *
      DeepSeekExpertPager::kBundleBytes;
  const auto expected_device_bytes =
      static_cast<std::uint64_t>(expert_slot_count) *
      DeepSeekExpertPager::kBundleBytes;
  const bool spill_budget_valid =
      !spill || (host_spill_pinned_bytes == expected_pinned_bytes &&
                 host_spill_device_bytes == expected_device_bytes);
  if ((!spill && !resident) ||
      (spill &&
       (effective_transfer_reservation_window == 0 ||
        expert_slot_count < effective_transfer_reservation_window ||
        staging_extent_count < effective_transfer_reservation_window)) ||
      (resident && (expert_slot_count != 0 || staging_extent_count != 0 ||
                    host_spill_pinned_bytes != 0 ||
                    host_spill_device_bytes != 0 ||
                    transfer_reservation_window != 0)) ||
      !spill_budget_valid) {
    return Status::InvalidArgument(
        "DeepSeek NVIDIA expert residency resources are invalid");
  }
  NvidiaDeepSeekBootstrapConfig result;
  result.epoch_ = epoch;
  result.world_size_ = world_size;
  result.device_ordinals_ = std::move(device_ordinals);
  result.expert_residency_ = expert_residency;
  result.expert_slot_count_ = expert_slot_count;
  result.staging_extent_count_ = staging_extent_count;
  result.artifact_poll_interval_ms_ = artifact_poll_interval_ms;
  result.attention_reserved_tokens_per_sequence_ =
      attention_reserved_tokens_per_sequence;
  result.host_spill_pinned_bytes_ = spill ? expected_pinned_bytes : 0;
  result.host_spill_device_bytes_ = spill ? expected_device_bytes : 0;
  result.transfer_reservation_window_ =
      spill ? effective_transfer_reservation_window : 0;
  result.execution_topology_ = execution_topology;
  return result;
}


Result<std::unique_ptr<DeepSeekEngine>> BuildCore(
    DeepSeekCapabilityArtifactCatalog catalog, const DeepSeekPipelinePlan& pipeline,
    DeepSeekPipelineCapacity capacity,
    const NvidiaDeepSeekBootstrapConfig& config,
    bool dspark_permitted,
    std::shared_ptr<const RuntimeEngineAdmission> admission_anchor,
    NvidiaDeepSeekRankRuntimeFactory& runtime_factory) {
  if (config.execution_topology() !=
      DeepSeekExecutionTopology::kInProcessRankSetDevelopment) {
    return Status::FailedPrecondition(
        "DeepSeek in-process CUDA bootstrap received a process-per-rank topology");
  }
  if (pipeline.world_size() != config.world_size() ||
      catalog.mapping_plan().world_size() != config.world_size() ||
      capacity.world_size != config.world_size()) {
    return Status::InvalidArgument(
        "DeepSeek NVIDIA bootstrap authorities disagree on world size");
  }
  if (config.expert_residency() ==
          DeepSeekRoutedExpertResidency::kHostSpill &&
      (config.host_spill_pinned_bytes() !=
           static_cast<std::uint64_t>(config.staging_extent_count()) *
               DeepSeekExpertPager::kBundleBytes ||
       config.host_spill_device_bytes() !=
           static_cast<std::uint64_t>(config.expert_slot_count()) *
               DeepSeekExpertPager::kBundleBytes)) {
    return Status::FailedPrecondition(
        "DeepSeek host-spill budget differs from rank resources");
  }
  
  const auto stable_authority = [&]() {
    {
      return catalog.poll_leases();
    }
  }();
  if (!stable_authority.ok()) return stable_authority;
  bool pipeline_dspark = false;
  for (std::uint32_t rank = 0; rank < pipeline.world_size(); ++rank) {
    pipeline_dspark = pipeline_dspark || pipeline.rank(rank).owns_dspark;
  }
  const auto dspark_gate = validate_dspark_runtime_gate(
      pipeline_dspark, capacity.dspark_enabled, dspark_permitted);
  if (!dspark_gate.ok()) return dspark_gate;
  {
    if (pipeline_dspark || admission_anchor != nullptr ||
        config.world_size() != 1) {
      return Status::FailedPrecondition(
          "DeepSeek capability catalog is restricted to development PP1");
    }
  }
  const bool development_sm89_pp1 =
      admission_anchor == nullptr && config.world_size() == 1 &&
      !pipeline_dspark;
  if (admission_anchor == nullptr && !development_sm89_pp1) {
    return Status::FailedPrecondition(
        "DeepSeek rank runtime requires an admitted hardware profile");
  }
  auto architecture = development_sm89_pp1
      ? Result<DeepSeekGpuArchitecture>(DeepSeekGpuArchitecture::kSm89)
      : admitted_deepseek_architecture(admission_anchor->gpu_family());
  if (!architecture.ok()) return architecture.status();
  auto optimization_policy = DeepSeekOptimizationPolicy::Create({
      .architecture = *architecture,
      .world_size = config.world_size(),
      .fused_kernel_set = DeepSeekFusedKernelSet::kUnfusedControl,
      .dspark = pipeline_dspark,
  });
  if (!optimization_policy.ok()) return optimization_policy.status();
  std::unique_ptr<RegisteredPinnedAllocator> pinned;
  std::vector<std::unique_ptr<Allocator>> allocators;
  std::vector<DeepSeekRankEngineResources> ranks;
  std::vector<std::unique_ptr<DeepSeekRankRuntimeOwner>> runtimes;
  allocators.reserve(config.world_size());
  ranks.reserve(config.world_size());
  runtimes.reserve(config.world_size());
  DeepSeekWeightBindingDryRun dry_run;
  for (std::uint32_t rank = 0; rank < config.world_size(); ++rank) {
    const auto stage = pipeline.rank(rank);
    Result<std::unique_ptr<NvidiaDeepSeekRankRuntimeView>> runtime = [&]()
        -> Result<std::unique_ptr<NvidiaDeepSeekRankRuntimeView>> {
      return runtime_factory.Create(
          config.device_ordinals()[rank], rank, config.epoch(),
          *optimization_policy);
    }();
    if (!runtime.ok()) return runtime.status();
    auto* runtime_view = runtime->get();
    if (pinned == nullptr) {
      auto created = runtime_view->CreateSharedPinnedHostAllocator();
      if (!created.ok()) return created.status();
      pinned = std::move(*created);
    }
    auto allocator = runtime_view->CreateDeviceAllocator();
    if (!allocator.ok()) return allocator.status();
    auto copier = runtime_view->CreateDeviceMemoryCopier();
    if (!copier.ok()) return copier.status();
    auto records = catalog.rank_tensor_records(rank);
    if (!records.ok()) return records.status();
    std::vector<DeepSeekRankTensorRecord> owned_records(
        records->begin(), records->end());
    auto built = [&]() -> Result<DeepSeekRankEngineResources> {
      {
        auto leases = catalog.rank_leases(rank);
        if (!leases.ok()) return leases.status();
        return DeepSeekRankEngineResources::BuildForInProcessDevelopment(
            config.epoch(), config.world_size(), pipeline.rank(rank).layers,
            config.expert_residency(), catalog.mapping_plan().rank(rank),
            std::move(*leases), std::move(owned_records),
            config.expert_slot_count(), config.staging_extent_count(),
            capacity, **allocator, *pinned, **copier, dry_run,
            config.host_spill_pinned_bytes(),
            config.host_spill_device_bytes(),
            config.transfer_reservation_window(),
            runtime_view->host_spill_pinned_allocator());
      }
    }();
    if (!built.ok()) return built.status();
    const bool host_spill = config.expert_residency() ==
                            DeepSeekRoutedExpertResidency::kHostSpill;
    if (host_spill &&
        (built->expert_pager() == nullptr || built->expert_source() == nullptr ||
         built->expert_pager()->transfer_reservation_window() !=
             config.transfer_reservation_window())) {
      return Status::Internal(
          "DeepSeek host-spill rank differs from compiled pager plan");
    }
    if (!host_spill &&
        (built->expert_pager() != nullptr || built->expert_source() != nullptr)) {
      return Status::Internal(
          "DeepSeek full-resident rank unexpectedly created pager resources");
    }
    auto device_resources = DeepSeekRankExpertDeviceResources::Allocate(
        **allocator, config.expert_slot_count(), capacity.expert_tokens,
        runtime_view->identity().context, config.device_ordinals()[rank],
        runtime_view->host_spill_device_allocator());
    if (!device_resources.ok()) return device_resources.status();
    auto attention_state = DeepSeekRankAttentionStatePool::Allocate(
        **allocator, pipeline.rank(rank), capacity.max_sequences,
        config.attention_reserved_tokens_per_sequence(),
        runtime_view->identity().event,
        runtime_view->fixed_state_operations(),
        runtime_view->identity().context, config.device_ordinals()[rank]);
    if (!attention_state.ok()) return attention_state.status();
    auto attention = DeepSeekRankAttentionResources::Allocate(
        stage.layers, capacity.max_pipeline_tokens,
        attention_state->ratio4_page_pairs_per_sequence(),
        attention_state->fixed_layout(), attention_state->page_arena(),
        runtime_view->attention_operations(), *pinned, **allocator,
        runtime_view->identity().event, runtime_view->identity().context,
        config.device_ordinals()[rank]);
    if (!attention.ok()) return attention.status();
    auto learned_router_device =
        DeepSeekLearnedRouterDeviceResources::Allocate(
            capacity.expert_tokens, device_resources->arena(), **allocator,
            runtime_view->identity().context, config.device_ordinals()[rank]);
    if (!learned_router_device.ok()) return learned_router_device.status();
    auto hash_router_staging =
        DeepSeekHashRouterStagingPool::Allocate(
            stage.layers, capacity.expert_tokens, *pinned);
    if (!hash_router_staging.ok()) return hash_router_staging.status();
    auto learned_router_staging =
        DeepSeekLearnedRouterStagingPool::Allocate(
            stage.layers, capacity.expert_tokens, *pinned);
    if (!learned_router_staging.ok()) return learned_router_staging.status();
    auto endpoint = DeepSeekEndpointRuntimeResources::Allocate(
        pipeline.rank(rank), runtime_view->endpoint_operations(),
        *pinned);
    if (!endpoint.ok()) return endpoint.status();
    auto endpoint_device = DeepSeekEndpointDeviceResources::Allocate(
        pipeline.rank(rank), capacity.max_pipeline_tokens, **allocator,
        runtime_view->identity().context, config.device_ordinals()[rank]);
    if (!endpoint_device.ok()) return endpoint_device.status();
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
    std::unique_ptr<DeepSeekDsparkRuntimeResources> dspark;
    std::unique_ptr<DeepSeekDsparkDeviceResources> dspark_device;
    if (stage.owns_dspark) {
      auto allocated = DeepSeekDsparkRuntimeResources::Allocate(
          stage, runtime_view->dspark_embed_operations(),
          runtime_view->dspark_head_operations(),
          runtime_view->dspark_prefill_operations(),
          runtime_view->dspark_decode_operations(),
          runtime_view->dspark_moe_operations(), *pinned);
      if (!allocated.ok()) return allocated.status();
      dspark = std::make_unique<DeepSeekDsparkRuntimeResources>(
          std::move(*allocated));
      auto device = DeepSeekDsparkDeviceResources::Allocate(
          stage, capacity.max_pipeline_tokens, **allocator,
          runtime_view->identity().context, config.device_ordinals()[rank]);
      if (!device.ok()) return device.status();
      dspark_device = std::make_unique<DeepSeekDsparkDeviceResources>(
          std::move(*device));
    }
#else
    if (stage.owns_dspark) {
      return Status::FailedPrecondition(
          "DeepSeek DSpark is not present in this CUDA backend bundle");
    }
#endif
    auto dense_mhc = DeepSeekDenseMhcRuntimeResources::Allocate(
        stage.layers,
        {&runtime_view->attention_projection_operations(),
         &runtime_view->attention_output_projection_operations(),
         &runtime_view->mhc_operations()},
        *pinned);
    if (!dense_mhc.ok()) return dense_mhc.status();
    auto attention_projection_device =
        DeepSeekAttentionProjectionDeviceResources::Allocate(
            **allocator, capacity.max_pipeline_tokens,
            runtime_view->identity().context,
            config.device_ordinals()[rank]);
    if (!attention_projection_device.ok()) {
      return attention_projection_device.status();
    }
    auto mhc_device = DeepSeekMhcDeviceResources::Allocate(
        **allocator, capacity.max_pipeline_tokens,
        runtime_view->identity().context, config.device_ordinals()[rank]);
    if (!mhc_device.ok()) return mhc_device.status();
    auto rope_tables =
        DeepSeekRopeTableDeviceResources::AllocateDeferred(
            **allocator, stage,
            config.attention_reserved_tokens_per_sequence(),
            runtime_view->identity().stream,
            runtime_view->rope_table_operations(),
            runtime_view->identity().context,
            config.device_ordinals()[rank]);
    if (!rope_tables.ok()) return rope_tables.status();
    auto request_input_staging =
        DeepSeekRequestInputStagingResources::Allocate(
            capacity.max_pipeline_tokens, *pinned);
    if (!request_input_staging.ok()) {
      return request_input_staging.status();
    }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
    std::unique_ptr<DeepSeekDsparkResidentExpertBindings>
        dspark_resident_experts;
    if (stage.owns_dspark) {
      auto resolved = DeepSeekDsparkResidentExpertBindings::Resolve(
          stage, built->resident_weights());
      if (!resolved.ok()) return resolved.status();
      dspark_resident_experts =
          std::make_unique<DeepSeekDsparkResidentExpertBindings>(
              std::move(*resolved));
    }
#endif
    std::unique_ptr<DeepSeekResidentExpertBindings> resident_experts;
    if (!host_spill) {
      auto resolved = DeepSeekResidentExpertBindings::Resolve(
          stage, built->resident_weights());
      if (!resolved.ok()) return resolved.status();
      resident_experts = std::make_unique<DeepSeekResidentExpertBindings>(
          std::move(*resolved));
    }
    auto kernel_lane = runtime_view->CreateExpertKernelLane(
        *pinned, device_resources->slot_table(), device_resources->arena(),
        capacity.expert_tokens, device_resources->source_hidden_bf16(),
        device_resources->accumulator_f32()
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
        , stage.owns_dspark ? mhc_device->view().layer_input_bf16 : 0
#endif
        );
    if (!kernel_lane.ok()) return kernel_lane.status();
    auto shared_operations = runtime_view->CreateSharedExpertOperations();
    if (!shared_operations.ok()) return shared_operations.status();
    auto shared_experts = DeepSeekSharedExpertRuntimeResources::Create(
        stage.layers.first_layer, stage.layers.last_layer, capacity.expert_tokens,
        built->resident_weights(), device_resources->shared_arena(),
        mhc_device->view().layer_input_bf16, device_resources->source_hidden_bf16(),
        device_resources->accumulator_f32(), mhc_device->view().ffn_branch_output_bf16,
        runtime_view->identity().stream, runtime_view->identity().event,
        std::move(*shared_operations), *pinned);
    if (!shared_experts.ok()) return shared_experts.status();
    // CreateComplete takes ownership of device_resources. Preserve the pager's
    // destination identities before that move; reading slot_bases() from the
    // moved-from Result produced an empty Host-Spill transfer lane.
    std::vector<std::uintptr_t> expert_slot_bases;
    if (host_spill) expert_slot_bases = device_resources->slot_bases();
    auto infrastructure = DeepSeekRankComputeInfrastructure::CreateComplete(
        std::move(*device_resources), std::move(*attention_state),
        std::move(*attention), std::move(*learned_router_device),
        std::move(*hash_router_staging),
        std::move(*learned_router_staging), std::move(*endpoint),
        std::move(*endpoint_device),
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
        std::move(dspark), std::move(dspark_device),
#endif
        std::move(*dense_mhc),
        std::move(*attention_projection_device),
        std::move(*mhc_device),
        std::move(*rope_tables),
        std::move(*request_input_staging),
        runtime_view->identity().stream,
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
        std::move(dspark_resident_experts),
#endif
        std::move(resident_experts));
    if (!infrastructure.ok()) return infrastructure.status();
    auto shared_attached = (*infrastructure)->attach_shared_experts(std::move(*shared_experts));
    if (!shared_attached.ok()) return shared_attached;
    Result<DeepSeekRankComputeLaneSet> lanes = host_spill
        ? [&]() -> Result<DeepSeekRankComputeLaneSet> {
            auto transfer_lane = runtime_view->CreateExpertTransferLane(
                *built->expert_source(), std::move(expert_slot_bases),
                config.transfer_reservation_window());
            if (!transfer_lane.ok()) return transfer_lane.status();
            return DeepSeekRankComputeLaneSet::CreateOwnedInfrastructure(
                pipeline.rank(rank), capacity.max_sequences,
                capacity.expert_tokens, *built->expert_pager(),
                std::move(*infrastructure),
                std::move(*transfer_lane),
                std::move(*kernel_lane));
          }()
        : DeepSeekRankComputeLaneSet::CreateOwnedResidentInfrastructure(
              pipeline.rank(rank), capacity.max_sequences,
              capacity.expert_tokens, std::move(*infrastructure),
              std::move(*kernel_lane));
    if (!lanes.ok()) return lanes.status();
    auto attach_status = built->attach_compute_lanes(
        std::make_unique<DeepSeekRankComputeLaneSet>(std::move(*lanes)));
    if (!attach_status.ok()) return attach_status;
    allocators.push_back(std::move(*allocator));
    runtimes.push_back(std::move(*runtime));
    ranks.push_back(std::move(*built));
  }
  Result<DeepSeekEngineResources> resources = [&]()
      -> Result<DeepSeekEngineResources> {
    return DeepSeekEngineResources::CreateOwnedWithRuntimes(
        config.epoch(), config.artifact_poll_interval_ms(),
        std::move(allocators), std::move(pinned), std::move(runtimes),
        std::move(ranks));
  }();
  if (!resources.ok()) return resources.status();
  auto engine = DeepSeekEngine::Create(
      std::make_unique<DeepSeekEngineResources>(std::move(*resources)),
      std::make_unique<CatalogArtifactPoller>(
          std::move(catalog), std::move(admission_anchor)),
      std::move(capacity));
  if (!engine.ok()) return engine.status();
  return std::make_unique<DeepSeekEngine>(std::move(*engine));
}


Result<std::unique_ptr<DeepSeekEngine>>
NvidiaDeepSeekEngineBootstrap::
BuildDevelopmentSm89Pp1FromArtifactCapabilityWithRuntimeFactory(
    const pih_verified_artifact_api_v1& artifact_api,
    const std::filesystem::path& trusted_generation_root,
    const Sha256Digest& expected_artifact_root,
    std::uint64_t maximum_shard_bytes, DeepSeekPipelineCapacity capacity,
    const NvidiaDeepSeekBootstrapConfig& config,
    NvidiaDeepSeekRankRuntimeFactory& runtime_factory) {
  if (trusted_generation_root.empty() || maximum_shard_bytes == 0) {
    return Status::InvalidArgument(
        "DeepSeek artifact capability bootstrap input is invalid");
  }
  if (capacity.dspark_enabled || config.world_size() != 1 ||
      config.execution_topology() !=
          DeepSeekExecutionTopology::kInProcessRankSetDevelopment) {
    return Status::FailedPrecondition(
        "DeepSeek artifact capability bootstrap requires development PP1");
  }
  auto pipeline = DeepSeekPipelinePlan::Create(1, false);
  if (!pipeline.ok()) return pipeline.status();
  auto catalog =
      DeepSeekCapabilityArtifactCatalog::OpenFlash0731TargetGeneration(
          artifact_api, trusted_generation_root, expected_artifact_root,
          *pipeline, maximum_shard_bytes);
  if (!catalog.ok()) return catalog.status();
  return BuildDevelopmentSm89Pp1FromArtifactCatalogWithRuntimeFactory(
      std::move(*catalog), std::move(capacity), config, runtime_factory);
}

Result<std::unique_ptr<DeepSeekEngine>>
NvidiaDeepSeekEngineBootstrap::
BuildDevelopmentSm89Pp1FromArtifactCatalogWithRuntimeFactory(
    DeepSeekCapabilityArtifactCatalog catalog,
    DeepSeekPipelineCapacity capacity,
    const NvidiaDeepSeekBootstrapConfig& config,
    NvidiaDeepSeekRankRuntimeFactory& runtime_factory) {
  if (capacity.dspark_enabled || config.world_size() != 1 ||
      config.execution_topology() !=
          DeepSeekExecutionTopology::kInProcessRankSetDevelopment) {
    return Status::FailedPrecondition(
        "DeepSeek artifact catalog bootstrap requires development PP1");
  }
  auto pipeline = DeepSeekPipelinePlan::Create(1, false);
  if (!pipeline.ok()) return pipeline.status();
  return BuildCore(
      std::move(catalog), *pipeline, std::move(capacity), config,
      false, nullptr, runtime_factory);
}


}  // namespace pih
