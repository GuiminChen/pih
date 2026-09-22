#include "pih/model/deepseek_engine_resources.h"

#include "pih/model/deepseek_rank_attention_resources.h"
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
#include "pih/model/deepseek_dspark_decode_mtp_plan_input_assembler.h"
#include "pih/model/deepseek_dspark_target_hidden_capture_assembler.h"
#endif

#include <algorithm>
#include <new>
#include <utility>

namespace pih {

DeepSeekEngineResources& DeepSeekEngineResources::operator=(
    DeepSeekEngineResources&& other) noexcept {
  if (this != &other) {
    // Ranks must release their CUDA allocations before their runtime and
    // allocator owners. A default member-wise assignment reverses that rule.
    this->~DeepSeekEngineResources();
    ::new (static_cast<void*>(this)) DeepSeekEngineResources(std::move(other));
  }
  return *this;
}

bool DeepSeekEngineResources::all_compute_ready() const noexcept {
  return std::ranges::all_of(
      ranks_, [](const auto& rank) { return rank.compute_ready(); });
}

Result<DeepSeekEngineResources> DeepSeekEngineResources::Create(
    std::uint64_t epoch, std::uint32_t artifact_poll_interval_ms,
    std::vector<DeepSeekRankEngineResources> ranks) {
  return CreateImpl(epoch, artifact_poll_interval_ms, {}, nullptr, false,
                    {}, false,
                    std::move(ranks));
}

Result<DeepSeekEngineResources> DeepSeekEngineResources::CreateOwned(
    std::uint64_t epoch, std::uint32_t artifact_poll_interval_ms,
    std::vector<std::unique_ptr<Allocator>> device_allocators,
    std::unique_ptr<RegisteredPinnedAllocator> pinned_allocator,
    std::vector<DeepSeekRankEngineResources> ranks) {
  return CreateImpl(epoch, artifact_poll_interval_ms,
                    std::move(device_allocators),
                    std::move(pinned_allocator), true, {}, false,
                    std::move(ranks));
}

Result<DeepSeekEngineResources>
DeepSeekEngineResources::CreateOwnedWithRuntimes(
    std::uint64_t epoch, std::uint32_t artifact_poll_interval_ms,
    std::vector<std::unique_ptr<Allocator>> device_allocators,
    std::unique_ptr<RegisteredPinnedAllocator> pinned_allocator,
    std::vector<std::unique_ptr<DeepSeekRankRuntimeOwner>> runtime_owners,
    std::vector<DeepSeekRankEngineResources> ranks) {
  return CreateImpl(epoch, artifact_poll_interval_ms,
                    std::move(device_allocators),
                    std::move(pinned_allocator), true,
                    std::move(runtime_owners), true,
                    std::move(ranks));
}


Result<DeepSeekEngineResources> DeepSeekEngineResources::CreateImpl(
    std::uint64_t epoch, std::uint32_t artifact_poll_interval_ms,
    std::vector<std::unique_ptr<Allocator>> device_allocators,
    std::unique_ptr<RegisteredPinnedAllocator> pinned_allocator,
    bool require_owned_allocators,
    std::vector<std::unique_ptr<DeepSeekRankRuntimeOwner>> runtime_owners,
    bool require_runtime_owners,
    std::vector<DeepSeekRankEngineResources> ranks) {
  if (epoch == 0 || ranks.size() != 1) {
    return Status::InvalidArgument("DeepSeek engine generation topology is invalid");
  }
  std::sort(ranks.begin(), ranks.end(), [](const auto& left, const auto& right) {
    return left.rank() < right.rank();
  });
  const auto world_size = static_cast<std::uint32_t>(ranks.size());
  if (require_owned_allocators &&
      (device_allocators.size() != world_size || pinned_allocator == nullptr ||
       std::ranges::any_of(device_allocators,
                           [](const auto& value) { return value == nullptr; }))) {
    return Status::InvalidArgument(
        "DeepSeek owned allocator set differs from rank topology");
  }
  if (require_runtime_owners &&
      (runtime_owners.size() != world_size ||
       std::ranges::any_of(runtime_owners,
                           [](const auto& value) { return value == nullptr; }))) {
    return Status::InvalidArgument(
        "DeepSeek runtime owner set differs from rank topology");
  }
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    if (ranks[rank].rank() != rank || ranks[rank].receipt().epoch() != epoch ||
        ranks[rank].receipt().world_size() != world_size) {
      return Status::FailedPrecondition(
          "DeepSeek engine rank resource set is incomplete or inconsistent");
    }
  }
  auto guard_value = DeepSeekArtifactEpochGuard::Create(
      epoch, artifact_poll_interval_ms, world_size);
  if (!guard_value.ok()) return guard_value.status();
  auto guard = std::make_unique<DeepSeekArtifactEpochGuard>(
      std::move(*guard_value));
  auto barrier_value = DeepSeekEngineBootstrapBarrier::Create(*guard);
  if (!barrier_value.ok()) return barrier_value.status();
  auto barrier = std::make_unique<DeepSeekEngineBootstrapBarrier>(
      std::move(*barrier_value));
  for (const auto& rank : ranks) {
    auto status = barrier->accept(rank.receipt());
    if (!status.ok()) return status;
  }
  if (!barrier->ready() || !guard->admission_allowed()) {
    return Status::Internal("DeepSeek engine bootstrap barrier did not publish ready");
  }
  return DeepSeekEngineResources(
      std::move(guard), std::move(barrier), std::move(device_allocators),
      std::move(pinned_allocator), std::move(runtime_owners),
      std::move(ranks));
}

Status DeepSeekEngineResources::verify_weight_seals() const {
  if (!admission_allowed()) {
    return Status::FailedPrecondition("DeepSeek engine generation is not ready");
  }
  for (const auto& rank : ranks_) {
    auto status = rank.verify_weight_seal();
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Result<std::vector<DeepSeekRankPlanReservation>>
DeepSeekEngineResources::prepare_rank_plan_resources(
    DeepSeekPipelinePlanDescriptor descriptor) {
  if (!admission_allowed()) {
    return Status::FailedPrecondition(
        "DeepSeek engine generation cannot reserve rank plan resources");
  }
  std::vector<DeepSeekRankPlanReservation> reservations;
  reservations.reserve(ranks_.size());
  for (auto& rank : ranks_) {
    auto reservation = rank.prepare_plan_resources(descriptor);
    if (!reservation.ok()) return reservation.status();
    reservations.push_back(std::move(*reservation));
  }
  return reservations;
}

Status DeepSeekEngineResources::bind_rank_compute_plans(
    DeepSeekPipelinePlanDescriptor descriptor,
    std::vector<DeepSeekRankComputePlanWork> work) {
  if (!admission_allowed() || work.size() != ranks_.size()) {
    return Status::InvalidArgument(
        "DeepSeek rank compute work differs from engine topology");
  }
  std::size_t bound = 0;
  for (; bound < ranks_.size(); ++bound) {
    auto status = ranks_[bound].bind_compute_plan(descriptor, work[bound]);
    if (!status.ok()) {
      while (bound > 0) {
        --bound;
        (void)ranks_[bound].abort_compute_plan(descriptor);
      }
      return status;
    }
  }
  return Status::Ok();
}

Status DeepSeekEngineResources::abort_rank_compute_plans(
    const DeepSeekPipelinePlanDescriptor& descriptor) noexcept {
  if (!std::ranges::all_of(ranks_, [&descriptor](const auto& rank) {
        return rank.can_abort_compute_plan(descriptor);
      })) {
    return Status::FailedPrecondition(
        "DeepSeek rank compute plans cannot be atomically aborted");
  }
  for (auto& rank : ranks_) {
    const auto status = rank.abort_compute_plan(descriptor);
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

Status DeepSeekEngineResources::activate_rank_runtime(std::uint32_t rank) {
  if (rank >= ranks_.size()) {
    return Status::InvalidArgument(
        "DeepSeek rank runtime activation rank is out of range");
  }
  if (runtime_owners_.empty()) return Status::Ok();
  if (runtime_owners_.size() != ranks_.size() ||
      runtime_owners_[rank] == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek rank runtime activation owner is incomplete");
  }
  return runtime_owners_[rank]->activate();
}

Result<DeepSeekLearnedRouterOperations*>
DeepSeekEngineResources::learned_router_operations(
    std::uint32_t rank) noexcept {
  if (rank >= ranks_.size() || runtime_owners_.size() != ranks_.size() ||
      runtime_owners_[rank] == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek learned router runtime owner is unavailable");
  }
  auto* operations = runtime_owners_[rank]->learned_router_operations();
  if (operations == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek learned router operations are unavailable");
  }
  return operations;
}

Result<DeepSeekRankRouterFactorySet>
DeepSeekEngineResources::assemble_rank_router_factories(
    std::uint32_t rank, std::uint32_t maximum_tokens,
    std::uint32_t vocabulary_size) {
  if (rank >= ranks_.size() || maximum_tokens == 0 ||
      vocabulary_size == 0) {
    return Status::InvalidArgument(
        "DeepSeek rank router assembly identity is invalid");
  }
  auto* bundle = ranks_[rank].compute_bundle();
  if (bundle == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek rank router assembly requires compute lanes");
  }
  auto operations = learned_router_operations(rank);
  if (!operations.ok()) return operations.status();
  return DeepSeekRankRouterFactorySet::Create(
      bundle->stage().layers, maximum_tokens, vocabulary_size,
      ranks_[rank].weight_source(), bundle->expert_store(), **operations);
}

Result<DeepSeekNativePlanCompiler>
DeepSeekEngineResources::assemble_native_plan_compiler(
    std::uint32_t maximum_sequences, std::uint32_t maximum_tokens,
    std::uint32_t vocabulary_size) {
  if (maximum_sequences == 0 || maximum_tokens == 0 ||
      vocabulary_size == 0 || ranks_.empty()) {
    return Status::InvalidArgument(
        "DeepSeek native compiler capacity is invalid");
  }
  auto* final_bundle = ranks_.back().compute_bundle();
  if (final_bundle == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek native compiler requires complete compute lanes");
  }
  auto topology = DeepSeekPipelinePlan::Create(
      world_size(), final_bundle->stage().owns_dspark);
  if (!topology.ok()) return topology.status();
  std::vector<DeepSeekRankPlanCompiler> compilers;
  compilers.reserve(world_size());
  for (std::uint32_t rank = 0; rank < world_size(); ++rank) {
    auto* bundle = ranks_[rank].compute_bundle();
    auto* attention = ranks_[rank].attention_resources();
    auto* dense_runtime = ranks_[rank].dense_mhc_runtime_resources();
    if (bundle == nullptr || attention == nullptr || dense_runtime == nullptr) {
      return Status::FailedPrecondition(
          "DeepSeek native compiler runtime resources are incomplete");
    }
    auto routers = assemble_rank_router_factories(
        rank, maximum_tokens, vocabulary_size);
    if (!routers.ok()) return routers.status();
    const auto stage = bundle->stage();
    auto dense = DeepSeekDenseMhcWorkFactory::Create(
        stage.layers, maximum_sequences, maximum_tokens);
    if (!dense.ok()) return dense.status();
    auto endpoint = DeepSeekEndpointWorkFactory::Create(
        stage, maximum_sequences);
    if (!endpoint.ok()) return endpoint.status();
    auto compiler = DeepSeekRankPlanCompiler::Create(
        stage, std::move(routers->hash_router()),
        std::move(routers->learned_router()), attention->work_factory(),
        std::move(*dense), std::move(*endpoint), dense_runtime);
    if (!compiler.ok()) return compiler.status();
    compilers.push_back(std::move(*compiler));
  }
  return DeepSeekNativePlanCompiler::Create(
      std::move(*topology), std::move(compilers));
}

Result<DeepSeekDeferredNativePlanCompiler>
DeepSeekEngineResources::assemble_production_deferred_plan_compiler(
    DeepSeekPipelinePlanDescriptor descriptor,
    std::vector<DeepSeekProductionRankPlanSeed> rank_seeds,
    std::uint32_t maximum_sequences, std::uint32_t maximum_tokens,
    std::uint32_t vocabulary_size) {
  if (rank_seeds.size() != world_size() ||
      descriptor.engine_epoch != epoch() || descriptor.plan_sequence == 0 ||
      descriptor.phase == DeepSeekPlanPhase::kDrain ||
      descriptor.token_count == 0 || descriptor.sequence_count != 1) {
    return Status::InvalidArgument(
        "DeepSeek production deferred compiler request is invalid");
  }
  auto native = assemble_native_plan_compiler(
      maximum_sequences, maximum_tokens, vocabulary_size);
  if (!native.ok()) return native.status();
  auto topology = native->topology();
  auto rank_compilers = std::move(*native).release_rank_compilers();
  std::vector<std::unique_ptr<DeepSeekDeferredRankPlanInputAssembler>>
      assemblers;
  assemblers.reserve(world_size());
  for (std::uint32_t rank = 0; rank < world_size(); ++rank) {
    auto assembler = DeepSeekProductionRankPlanInputAssembler::Create(
        *this, rank, descriptor, std::move(rank_seeds[rank]));
    if (!assembler.ok()) return assembler.status();
    assemblers.push_back(std::move(*assembler));
  }
  return DeepSeekDeferredNativePlanCompiler::Create(
      std::move(topology), descriptor, std::move(rank_compilers),
      std::move(assemblers));
}

Result<std::vector<DeepSeekLearnedRouterLayerPlanWork>>
DeepSeekEngineResources::assemble_rank_learned_router_plan_input(
    std::uint32_t rank, std::uint32_t token_count,
    std::span<const DeepSeekLearnedRouterLayerInput> layer_inputs) {
  if (rank >= ranks_.size() || runtime_owners_.size() != ranks_.size() ||
      runtime_owners_[rank] == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek learned router plan rank is unavailable");
  }
  auto* bundle = ranks_[rank].compute_bundle();
  auto* device = ranks_[rank].learned_router_device_resources();
  auto* staging = ranks_[rank].learned_router_staging_pool();
  auto& runtime = *runtime_owners_[rank];
  if (bundle == nullptr || device == nullptr || staging == nullptr ||
      runtime.plan_compute_stream() == 0 ||
      runtime.plan_completion_event() == 0) {
    return Status::FailedPrecondition(
        "DeepSeek learned router plan resources are incomplete");
  }
  auto weights = DeepSeekLearnedRouterWeightBindings::Resolve(
      bundle->stage().layers, ranks_[rank].resident_weights());
  if (!weights.ok()) return weights.status();
  return DeepSeekLearnedRouterPlanInputAssembler::Assemble(
      token_count, layer_inputs, *weights, device->view(), *staging,
      runtime.plan_compute_stream(), runtime.plan_completion_event(),
      device->maximum_tokens());
}

Result<std::vector<DeepSeekHashRouterLayerPlanWork>>
DeepSeekEngineResources::assemble_rank_hash_router_plan_input(
    std::uint32_t rank, std::uint32_t token_count,
    std::span<const DeepSeekLearnedRouterLayerInput> layer_inputs) {
  if (rank >= ranks_.size() || runtime_owners_.size() != ranks_.size() ||
      runtime_owners_[rank] == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek hash router plan rank is unavailable");
  }
  auto* bundle = ranks_[rank].compute_bundle();
  auto* device = ranks_[rank].learned_router_device_resources();
  auto* staging = ranks_[rank].hash_router_staging_pool();
  auto& runtime = *runtime_owners_[rank];
  if (bundle == nullptr || device == nullptr || staging == nullptr ||
      runtime.plan_compute_stream() == 0 ||
      runtime.plan_completion_event() == 0) {
    return Status::FailedPrecondition(
        "DeepSeek hash router plan resources are incomplete");
  }
  auto weights = DeepSeekHashRouterWeightBindings::Resolve(
      bundle->stage().layers, ranks_[rank].resident_weights());
  if (!weights.ok()) return weights.status();
  return DeepSeekHashRouterPlanInputAssembler::Assemble(
      token_count, layer_inputs, *weights, device->view(), *staging,
      runtime.plan_compute_stream(), runtime.plan_completion_event(),
      device->maximum_tokens());
}

Result<DeepSeekDenseMhcStageSubmissions>
DeepSeekEngineResources::assemble_rank_dense_mhc_stage_submissions(
    std::uint32_t rank, std::uint32_t token_count,
    std::uintptr_t initial_residual_bf16,
    std::uint32_t table_position_count) {
  if (rank >= ranks_.size() || token_count == 0 ||
      initial_residual_bf16 == 0 ||
      table_position_count == 0 ||
      runtime_owners_.size() != ranks_.size() ||
      runtime_owners_[rank] == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek dense mHC rank submission identity is invalid");
  }
  auto* bundle = ranks_[rank].compute_bundle();
  auto* attention =
      ranks_[rank].attention_projection_device_resources();
  auto* mhc = ranks_[rank].mhc_device_resources();
  auto* rope = ranks_[rank].rope_table_device_resources();
  const auto stream = runtime_owners_[rank]->plan_compute_stream();
  if (bundle == nullptr || attention == nullptr || mhc == nullptr ||
      rope == nullptr ||
      stream == 0) {
    return Status::FailedPrecondition(
        "DeepSeek dense mHC rank submission resources are incomplete");
  }
  auto weights = DeepSeekDenseMhcStageWeightBindings::Resolve(
      bundle->stage().layers, ranks_[rank].resident_weights());
  if (!weights.ok()) return weights.status();
  auto submissions = DeepSeekDenseMhcStageSubmissionAssembler::Assemble(
      bundle->stage().layers, weights->layers(), *attention, *mhc, *rope,
      token_count, initial_residual_bf16,
      table_position_count, stream);
  if (!submissions.ok()) return submissions.status();
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  if (bundle->stage().owns_dspark) {
    auto* dspark = ranks_[rank].dspark_device_resources();
    if (dspark == nullptr) {
      return Status::FailedPrecondition(
          "DeepSeek DSpark target hidden capture storage is unavailable");
    }
    auto status = DeepSeekDsparkTargetHiddenCaptureAssembler::Bind(
        bundle->stage(), *dspark, token_count, submissions->layers);
    if (!status.ok()) return status;
  }
#else
  if (bundle->stage().owns_dspark) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark is not present in this model bundle");
  }
#endif
  return submissions;
}

Result<DeepSeekRequestInputStagingLease>
DeepSeekEngineResources::stage_rank_request_inputs(
    std::uint32_t rank, std::span<const std::uint32_t> token_ids,
    std::span<const std::uint32_t> positions) {
  if (rank >= ranks_.size() || runtime_owners_.size() != ranks_.size() ||
      runtime_owners_[rank] == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek request input rank is unavailable");
  }
  auto* projection =
      ranks_[rank].attention_projection_device_resources();
  auto* staging = ranks_[rank].request_input_staging_resources();
  auto& runtime = *runtime_owners_[rank];
  auto* operations = runtime.request_input_copy_operations();
  const auto stream = runtime.plan_compute_stream();
  if (projection == nullptr || staging == nullptr || operations == nullptr ||
      stream == 0) {
    return Status::FailedPrecondition(
        "DeepSeek request input staging resources are incomplete");
  }
  return staging->stage(token_ids, positions, projection->view(), stream,
                        *operations);
}

Result<std::uintptr_t> DeepSeekEngineResources::rank_request_token_ids_u32(
    std::uint32_t rank) {
  if (rank >= ranks_.size()) {
    return Status::InvalidArgument(
        "DeepSeek request token rank is unavailable");
  }
  auto* projection = ranks_[rank].attention_projection_device_resources();
  if (projection == nullptr || projection->view().token_ids_u32 == 0) {
    return Status::FailedPrecondition(
        "DeepSeek request token device storage is unavailable");
  }
  return projection->view().token_ids_u32;
}

Result<std::uintptr_t> DeepSeekEngineResources::rank_initial_residual_bf16(
    std::uint32_t rank) {
  if (rank >= ranks_.size()) {
    return Status::InvalidArgument(
        "DeepSeek initial residual rank is unavailable");
  }
  auto* bundle = ranks_[rank].compute_bundle();
  auto* endpoint = ranks_[rank].endpoint_device_resources();
  if (bundle == nullptr || endpoint == nullptr ||
      !bundle->stage().owns_embedding ||
      endpoint->view().embedding_output_hc_bf16 == 0) {
    return Status::FailedPrecondition(
        "DeepSeek initial residual requires the embedding rank");
  }
  return endpoint->view().embedding_output_hc_bf16;
}


Result<DeepSeekEndpointStageSequenceWork>
DeepSeekEngineResources::assemble_rank_endpoint_plan_input(
    std::uint32_t rank, std::uint32_t token_count,
    std::uintptr_t token_ids_u32, std::uintptr_t final_hc_bf16,
    std::optional<DeepSeekPreparedSamplingInput> sampling) {
  if (rank >= ranks_.size() || runtime_owners_.size() != ranks_.size() ||
      runtime_owners_[rank] == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek endpoint plan rank is unavailable");
  }
  auto* bundle = ranks_[rank].compute_bundle();
  auto* runtime = ranks_[rank].endpoint_runtime_resources();
  auto* device = ranks_[rank].endpoint_device_resources();
  const auto stream = runtime_owners_[rank]->plan_compute_stream();
  if (bundle == nullptr || runtime == nullptr || device == nullptr ||
      runtime->executor() == nullptr || stream == 0) {
    return Status::FailedPrecondition(
        "DeepSeek endpoint plan resources are incomplete");
  }
  auto weights = DeepSeekEndpointWeightBindings::Resolve(
      bundle->stage(), ranks_[rank].resident_weights());
  if (!weights.ok()) return weights.status();
  return DeepSeekEndpointPlanInputAssembler::Assemble(
      bundle->stage(), token_count, token_ids_u32, final_hc_bf16,
      *weights, device->view(), runtime->executor(), stream,
      device->maximum_tokens(), std::move(sampling));
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Result<DeepSeekDsparkStageWork>
DeepSeekEngineResources::assemble_rank_dspark_plan_input(
    std::uint32_t rank, DeepSeekPlanPhase phase,
    std::uint32_t token_count, std::uintptr_t input_token_ids_u32) {
  if (rank >= ranks_.size() || runtime_owners_.size() != ranks_.size() ||
      runtime_owners_[rank] == nullptr || token_count == 0 ||
      (phase != DeepSeekPlanPhase::kPrefill &&
       phase != DeepSeekPlanPhase::kDecode) ||
      (phase == DeepSeekPlanPhase::kDecode && token_count != 1)) {
    return Status::InvalidArgument(
        "DeepSeek DSpark plan rank is unavailable");
  }
  auto* bundle = ranks_[rank].compute_bundle();
  auto* runtime = ranks_[rank].dspark_runtime_resources();
  auto* device = ranks_[rank].dspark_device_resources();
  const auto stream = runtime_owners_[rank]->plan_compute_stream();
  if (bundle == nullptr || runtime == nullptr || device == nullptr ||
      runtime->embed_coordinator() == nullptr ||
      runtime->head_executor() == nullptr || stream == 0) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark plan resources are incomplete");
  }
  auto dspark_weights = ranks_[rank].dspark_weight_bindings();
  if (!dspark_weights.ok()) return dspark_weights.status();
  if (phase == DeepSeekPlanPhase::kPrefill) {
    return DeepSeekDsparkPlanInputAssembler::AssemblePrefill(
        bundle->stage(), token_count, **dspark_weights, device->view(),
        runtime->embed_coordinator(), nullptr, stream,
        device->maximum_tokens());
  }
  auto endpoint_weights = DeepSeekEndpointWeightBindings::Resolve(
      bundle->stage(), ranks_[rank].resident_weights());
  if (!endpoint_weights.ok()) return endpoint_weights.status();
  return DeepSeekDsparkPlanInputAssembler::Assemble(
      bundle->stage(), input_token_ids_u32,
      *endpoint_weights, **dspark_weights, device->view(),
      runtime->embed_coordinator(), runtime->head_executor(),
      nullptr, stream, device->maximum_tokens());
}

Result<std::vector<DeepSeekBoundDsparkMtpStageWork>>
DeepSeekEngineResources::assemble_rank_dspark_prefill_mtp_plan_input(
    std::uint32_t rank, std::uint32_t token_count,
    std::uint32_t position_table_count) {
  if (rank >= ranks_.size() || token_count == 0 ||
      position_table_count < token_count ||
      runtime_owners_.size() != ranks_.size() ||
      runtime_owners_[rank] == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek DSpark prefill MTP plan identity is invalid");
  }
  auto* bundle = ranks_[rank].compute_bundle();
  auto* runtime = ranks_[rank].dspark_runtime_resources();
  auto* device = ranks_[rank].dspark_device_resources();
  auto* state_pool = ranks_[rank].attention_state_pool();
  auto* projection =
      ranks_[rank].attention_projection_device_resources();
  auto* rope = ranks_[rank].rope_table_device_resources();
  const auto stream = runtime_owners_[rank]->plan_compute_stream();
  const auto completion_event =
      runtime_owners_[rank]->plan_completion_event();
  if (bundle == nullptr || !bundle->stage().owns_dspark || runtime == nullptr ||
      device == nullptr || state_pool == nullptr || projection == nullptr ||
      rope == nullptr || stream == 0 || completion_event == 0 ||
      token_count > device->maximum_tokens() ||
      token_count > projection->maximum_tokens() ||
      position_table_count > rope->maximum_positions()) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark prefill MTP resources are incomplete");
  }
  auto weights = ranks_[rank].dspark_weight_bindings();
  if (!weights.ok()) return weights.status();
  const auto device_view = device->view();
  const auto projection_view = projection->view();
  const auto rope_view = rope->view();
  std::vector<DeepSeekBoundDsparkMtpStageWork> result;
  result.reserve(kDeepSeekDsparkStageCount);
  for (std::size_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    const auto stage = static_cast<DeepSeekDsparkStageId>(index);
    auto* operation = runtime->prefill_operation(stage);
    const auto& common = (*weights)->common(stage);
    if (operation == nullptr || common.stage != stage ||
        common.generation != (*weights)->generation()) {
      return Status::FailedPrecondition(
          "DeepSeek DSpark prefill stage runtime is incomplete");
    }
    DeepSeekBoundDsparkMtpStageWork work;
    work.attention = operation;
    work.prefill_resources = {
        stage,
        {common.attention.wkv_fp8,
         common.attention.wkv_scale_ue8m0,
         common.attention.kv_norm_bf16,
         common.attention.generation},
        &state_pool->fixed_layout(),
        nullptr,
        0,
        {},
        device_view.main_normalized_bf16,
        device_view.main_quant_fp8,
        device_view.main_quant_scale_ue8m0,
        projection_view.positions_u32,
        device_view.prefill_kv_bf16,
        rope_view.base_f32,
        device_view.error_flag_u32,
        stream,
        completion_event,
        token_count,
        position_table_count,
        (*weights)->generation()};
    result.push_back(work);
  }
  return result;
}

Result<std::vector<DeepSeekBoundDsparkMtpStageWork>>
DeepSeekEngineResources::assemble_rank_dspark_decode_mtp_plan_input(
    std::uint32_t rank, std::uint32_t current_position,
    std::uint32_t position_table_count) {
  if (rank >= ranks_.size() || position_table_count == 0 ||
      current_position > UINT32_MAX - 5U ||
      current_position + 5U >= position_table_count ||
      runtime_owners_.size() != ranks_.size() ||
      runtime_owners_[rank] == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek DSpark decode MTP plan identity is invalid");
  }
  auto* bundle = ranks_[rank].compute_bundle();
  auto* runtime = ranks_[rank].dspark_runtime_resources();
  auto* device = ranks_[rank].dspark_device_resources();
  auto* state_pool = ranks_[rank].attention_state_pool();
  auto* projection =
      ranks_[rank].attention_projection_device_resources();
  auto* mhc = ranks_[rank].mhc_device_resources();
  auto* rope = ranks_[rank].rope_table_device_resources();
  auto* learned_router =
      ranks_[rank].learned_router_device_resources();
  auto* expert_kernel = ranks_[rank].dspark_expert_kernel();
  const auto expert_arena = ranks_[rank].expert_compute_arena();
  const auto expert_accumulator =
      ranks_[rank].expert_accumulator_f32();
  const auto stream = runtime_owners_[rank]->plan_compute_stream();
  const auto completion_event =
      runtime_owners_[rank]->plan_completion_event();
  if (bundle == nullptr || !bundle->stage().owns_dspark || runtime == nullptr ||
      device == nullptr || state_pool == nullptr || projection == nullptr ||
      mhc == nullptr || rope == nullptr || learned_router == nullptr ||
      expert_kernel == nullptr || expert_accumulator == 0 || stream == 0 ||
      completion_event == 0 || position_table_count > rope->maximum_positions()) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark decode MTP resources are incomplete");
  }
  auto weights = ranks_[rank].dspark_weight_bindings();
  if (!weights.ok()) return weights.status();
  const auto* dspark_experts =
      ranks_[rank].dspark_resident_expert_bindings();
  if (dspark_experts == nullptr ||
      dspark_experts->generation() != (*weights)->generation()) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark resident experts are unavailable");
  }

  const auto device_view = device->view();
  const auto projection_view = projection->view();
  const auto mhc_view = mhc->view();
  const auto rope_view = rope->view();
  const auto router_view = learned_router->view();
  const auto scores = runtime->router_host_scores();
  const auto bias = runtime->router_host_bias();
  DeepSeekDsparkDecodeMtpPlanInput input;
  input.weights = *weights;
  input.resident_experts = dspark_experts;
  input.state_layout = &state_pool->fixed_layout();
  input.device = device_view;
  input.attention_workspace = projection_view;
  input.mhc_workspace = mhc_view;
  input.rope_frequencies_f32 = rope_view.base_f32;
  input.router_scores_f32 = router_view.scores_f32;
  input.router_host_scores = scores;
  input.router_host_bias = bias;
  input.expert_arena = expert_arena;
  input.expert_accumulator_f32 = expert_accumulator;
  input.expert_kernel = expert_kernel;
  input.stream = stream;
  input.completion_event = completion_event;
  input.current_position = current_position;
  input.position_table_count = position_table_count;
  for (std::size_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    const auto stage = static_cast<DeepSeekDsparkStageId>(index);
    input.attention_operations[index] =
        runtime->decode_attention_operation(stage);
    input.moe_operations[index] = runtime->moe_operation(stage);
  }
  return DeepSeekDsparkDecodeMtpPlanInputAssembler::Assemble(input);
}
#endif


}  // namespace pih
