#include "pih/model/deepseek_engine.h"

#include <algorithm>
#include <new>
#include <utility>


namespace pih {

DeepSeekEngine& DeepSeekEngine::operator=(DeepSeekEngine&& other) noexcept {
  if (this != &other) {
    // Active rank runtimes and reservations borrow from resources_. Preserve
    // reverse destruction order when replacing a live engine instance.
    this->~DeepSeekEngine();
    ::new (static_cast<void*>(this)) DeepSeekEngine(std::move(other));
  }
  return *this;
}

namespace {


}  // namespace

Result<DeepSeekEngine> DeepSeekEngine::Create(
    std::unique_ptr<DeepSeekEngineResources> resources,
    std::unique_ptr<DeepSeekEngineArtifactPoller> artifact_poller,
    DeepSeekPipelineCapacity pipeline_capacity) {
  if (resources == nullptr || artifact_poller == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek engine resources and artifact poller are required");
  }
  if (!resources->admission_allowed()) {
    return Status::FailedPrecondition(
        "DeepSeek engine generation is not ready for admission");
  }
  const auto seal_status = resources->verify_weight_seals();
  if (!seal_status.ok()) return seal_status;
  if (pipeline_capacity.world_size != resources->world_size()) {
    return Status::InvalidArgument(
        "DeepSeek pipeline capacity world size differs from engine");
  }
  const auto rank_runtime_capacity = pipeline_capacity;
  auto control_plane = DeepSeekControlPlane::Create(
      resources->epoch(), std::move(pipeline_capacity));
  if (!control_plane.ok()) return control_plane.status();
  std::vector<DeepSeekPipelineResourceSet> rank_runtime_resources;
  rank_runtime_resources.reserve(resources->world_size());
  for (std::uint32_t rank = 0; rank < resources->world_size(); ++rank) {
    auto rank_resources = DeepSeekPipelineResourceSet::Create(
        rank_runtime_capacity);
    if (!rank_resources.ok()) return rank_resources.status();
    rank_runtime_resources.push_back(std::move(*rank_resources));
  }
  DeepSeekEngine engine(
      std::move(resources), std::move(artifact_poller),
      std::make_unique<DeepSeekControlPlane>(std::move(*control_plane)),
      std::move(rank_runtime_resources));
  auto* attention_pool = engine.resources_->rank(0).attention_state_pool();
  const auto attention_capacity = attention_pool == nullptr
      ? rank_runtime_capacity.max_sequences
      : attention_pool->sequence_capacity();
  if (attention_capacity == 0) {
    return Status::FailedPrecondition(
        "DeepSeek engine has no request attention state capacity");
  }
  for (std::uint32_t rank = 1; rank < engine.world_size(); ++rank) {
    const auto* rank_pool =
        engine.resources_->rank(rank).attention_state_pool();
    if ((rank_pool == nullptr) != (attention_pool == nullptr) ||
        (rank_pool != nullptr &&
         rank_pool->sequence_capacity() != attention_capacity)) {
      return Status::FailedPrecondition(
          "DeepSeek rank attention state capacities are inconsistent");
    }
  }
  engine.free_attention_state_slots_.reserve(
      attention_capacity);
  for (std::uint32_t slot = attention_capacity; slot > 0; --slot) {
    engine.free_attention_state_slots_.push_back(slot - 1);
  }
  const auto poll_status = engine.poll_artifacts();
  if (!poll_status.ok()) return poll_status;
  return engine;
}

Status DeepSeekEngine::admit_request() {
  if (state_ != DeepSeekEngineState::kReady ||
      !resources_->admission_allowed()) {
    return Status::FailedPrecondition(
        "DeepSeek engine generation is not accepting requests");
  }
  return poll_artifacts_if_due();
}

Status DeepSeekEngine::submit_request(std::uint64_t request_id,
                                      std::uint64_t request_generation) {
  const auto admission = admit_request();
  if (!admission.ok()) return admission;
  if (free_attention_state_slots_.empty()) {
    return Status::ResourceExhausted(
        "DeepSeek request attention state capacity is exhausted");
  }
  if (next_attention_sequence_ == 0) {
    return Status::ResourceExhausted(
        "DeepSeek logical attention sequence identity is exhausted");
  }
  const auto slot = free_attention_state_slots_.back();
  const auto [binding, inserted] = request_attention_bindings_.emplace(
      request_id,
      RequestAttentionBinding{request_generation,
                              {next_attention_sequence_, slot}});
  if (!inserted) {
    return Status::FailedPrecondition(
        "DeepSeek request attention binding already exists");
  }
  const auto status = control_plane_->submit(request_id, request_generation);
  if (!status.ok()) {
    request_attention_bindings_.erase(binding);
    return status;
  }
  free_attention_state_slots_.pop_back();
  ++next_attention_sequence_;
  return Status::Ok();
}

Status DeepSeekEngine::cancel_request(std::uint64_t request_id,
                                      std::uint64_t request_generation) {
  // Input PREPARE may have failed before a coordinator/runtime was bound.
  // Do not retire the attention slot while its input DMA is unproven.
  for (std::uint32_t rank = 0; rank < world_size(); ++rank) {
    auto* rope = resources_->rank(rank).rope_table_device_resources();
    if (rope != nullptr) {
      const auto retired = rope->retire_pending();
      if (!retired.ok())
        return fail(DeepSeekEngineFailure::kWorkerLost, retired);
    }
    auto* staging = resources_->rank(rank).request_input_staging_resources();
    if (staging != nullptr) {
      const auto retired = staging->retire_pending();
      if (!retired.ok())
        return fail(DeepSeekEngineFailure::kWorkerLost, retired);
    }
  }
  const auto status = control_plane_->cancel(request_id, request_generation);
  if (status.ok() && !control_plane_->execution_live()) {
    if (bound_execution_) {
      if (current_rank_runtimes_.empty()) {
        return fail(DeepSeekEngineFailure::kWorkerLost,
                    Status::Internal(
                        "DeepSeek cancelled bound execution lost runtimes"));
      }
      const auto descriptor = current_rank_runtimes_.front().descriptor();
      const auto abort = resources_->abort_rank_compute_plans(descriptor);
      if (!abort.ok()) {
        return fail(DeepSeekEngineFailure::kWorkerLost, abort);
      }
      current_rank_runtimes_.clear();
      current_rank_work_owners_.clear();
      rank_completion_reported_.clear();
      bound_execution_ = false;
      publish_bound_sampled_token_ = false;
      bound_sampled_token_staged_ = false;
      bound_sampling_config_.reset();
      bound_sampling_.reset();
    }
    current_rank_plan_resources_.clear();
  }
  return status;
}

Status DeepSeekEngine::finish_request(std::uint64_t request_id,
                                      std::uint64_t request_generation) {
  return control_plane_->finish_request(request_id, request_generation);
}

Status DeepSeekEngine::retire_request(std::uint64_t request_id,
                                      std::uint64_t request_generation) {
  const auto request =
      control_plane_->validate_retire(request_id, request_generation);
  if (!request.ok()) return request;
  const auto binding =
      validate_release_attention_binding(request_id, request_generation);
  if (!binding.ok()) return binding;
  const auto status = control_plane_->retire(request_id, request_generation);
  if (!status.ok()) return status;
  return release_attention_binding(request_id, request_generation);
}

Status DeepSeekEngine::configure_ledger(
    std::uint64_t request_id, std::uint64_t request_generation,
    std::uint64_t prompt_token_count,
    std::uint32_t maximum_completion_tokens,
    std::uint32_t minimum_completion_tokens) {
  return control_plane_->configure_ledger(
      request_id, request_generation, prompt_token_count,
      maximum_completion_tokens, minimum_completion_tokens);
}

Status DeepSeekEngine::configure_sampling(
    std::uint64_t request_id, std::uint64_t request_generation,
    DeepSeekRequestSamplingConfig config) {
  return control_plane_->configure_sampling(
      request_id, request_generation, config);
}

Result<DeepSeekRequestState> DeepSeekEngine::request_state(
    std::uint64_t request_id, std::uint64_t request_generation) const {
  return control_plane_->request_state(request_id, request_generation);
}

Result<DeepSeekAcceptedTokenSnapshot> DeepSeekEngine::accepted_token_snapshot(
    std::uint64_t request_id, std::uint64_t request_generation) const {
  return control_plane_->accepted_token_snapshot(request_id, request_generation);
}

Result<DeepSeekAttentionSequenceBinding>
DeepSeekEngine::request_attention_binding(
    std::uint64_t request_id, std::uint64_t request_generation) const {
  const auto iterator = request_attention_bindings_.find(request_id);
  if (iterator == request_attention_bindings_.end() ||
      iterator->second.request_generation != request_generation) {
    return Status::InvalidArgument(
        "DeepSeek request attention binding is not live");
  }
  return iterator->second.binding;
}

Status DeepSeekEngine::prepare_pipeline(
    DeepSeekPipelinePlanDescriptor descriptor,
    std::vector<DeepSeekRequestIdentity> request_identities) {
  const auto admission = admit_request();
  if (!admission.ok()) return admission;
  if (descriptor.engine_epoch != epoch()) {
    return Status::FailedPrecondition(
        "DeepSeek pipeline descriptor epoch differs from engine");
  }
  if (!current_rank_plan_resources_.empty()) {
    return Status::ResourceExhausted(
        "DeepSeek engine already owns rank plan resources");
  }
  auto rank_resources = resources_->prepare_rank_plan_resources(descriptor);
  if (!rank_resources.ok()) return rank_resources.status();
  const auto status = control_plane_->prepare_plan(
      descriptor, std::move(request_identities));
  if (!status.ok()) return status;
  current_rank_plan_resources_ = std::move(*rank_resources);
  return Status::Ok();
}

Status DeepSeekEngine::prepare_bound_pipeline(
    DeepSeekPipelinePlanDescriptor descriptor,
    std::vector<DeepSeekRequestIdentity> request_identities,
    std::vector<DeepSeekRankComputePlanWork> rank_work) {
  if (world_size() != 1 || rank_work.size() != 1) {
    return Status::FailedPrecondition(
        "DeepSeek PP1 engine requires one local compute plan");
  }
  if (descriptor.phase != DeepSeekPlanPhase::kDrain &&
      std::ranges::any_of(rank_work, [](const auto& work) {
        return work.lifetime_owner == nullptr;
      })) {
    return Status::InvalidArgument(
        "DeepSeek bound compute work has no lifetime owner");
  }
  if (!resources_->execution_topology_ready()) {
    return Status::FailedPrecondition(
        "DeepSeek execution topology resources are incomplete");
  }
  std::vector<std::shared_ptr<const DeepSeekRankComputePlanWorkOwner>>
      work_owners;
  work_owners.reserve(rank_work.size());
  for (const auto& work : rank_work) {
    work_owners.push_back(work.lifetime_owner);
  }
  std::vector<DeepSeekRankPlanRuntime> prepared_runtimes;
  prepared_runtimes.reserve(world_size());
  std::vector<bool> completion_reported(world_size(), false);
  for (std::uint32_t rank = 0; rank < world_size(); ++rank) {
    auto activated = resources_->activate_rank_runtime(rank);
    if (!activated.ok()) return activated;
  }
  const auto prepare_status = prepare_pipeline(
      descriptor, std::move(request_identities));
  if (!prepare_status.ok()) return prepare_status;
  const auto rollback = [this, &descriptor, &prepared_runtimes](
                            std::uint32_t rank, Status cause) {
    // A successful rejection deliberately returns the original non-OK cause;
    // ownership state, not that status value, proves whether rollback closed.
    (void)control_plane_->stage_reject(rank, cause);
    const auto abort = resources_->abort_rank_compute_plans(descriptor);
    if (control_plane_->execution_live() || !abort.ok()) {
      return fail(DeepSeekEngineFailure::kWorkerLost,
                  Status::Internal(
                      "DeepSeek bound runtime prepare rollback failed"));
    }
    prepared_runtimes.clear();
    current_rank_plan_resources_.clear();
    return cause;
  };
  const auto bind_status = resources_->bind_rank_compute_plans(
      descriptor, std::move(rank_work));
  if (!bind_status.ok()) return rollback(0, bind_status);
  if (current_rank_plan_resources_.size() != world_size()) {
    return rollback(0, Status::FailedPrecondition(
        "DeepSeek bound runtime lost rank plan reservations"));
  }
  for (std::uint32_t rank = 0; rank < world_size(); ++rank) {
    auto activated = resources_->activate_rank_runtime(rank);
    if (!activated.ok()) return rollback(rank, std::move(activated));
    auto transaction = rank_runtime_resources_[rank].prepare(descriptor);
    if (!transaction.ok()) return rollback(rank, transaction.status());
    auto* bundle = resources_->rank(rank).compute_bundle();
    if (bundle == nullptr) {
      return rollback(rank, Status::FailedPrecondition(
          "DeepSeek bound runtime lost rank compute resources"));
    }
    auto runtime = DeepSeekRankPlanRuntime::CreateBorrowedCompute(
        std::move(*transaction), bundle->stage(),
        std::move(current_rank_plan_resources_[rank]), bundle->driver());
    if (!runtime.ok()) return rollback(rank, runtime.status());
    auto ready = runtime->mark_ready();
    if (!ready.ok()) return rollback(rank, ready);
    ready = control_plane_->stage_ready(rank);
    if (!ready.ok()) return rollback(rank, ready);
    prepared_runtimes.push_back(std::move(*runtime));
  }
  current_rank_plan_resources_.clear();
  current_rank_runtimes_ = std::move(prepared_runtimes);
  current_rank_work_owners_ = std::move(work_owners);
  rank_completion_reported_ = std::move(completion_reported);
  bound_execution_ = true;
  return Status::Ok();
}

Status DeepSeekEngine::prepare_native_pipeline(
    DeepSeekNativeRankComputePlan plan,
    std::vector<DeepSeekRequestIdentity> request_identities) {
  const auto descriptor = plan.descriptor();
  auto rank_work = std::move(plan).release_rank_work();
  return prepare_bound_pipeline(
      descriptor, std::move(request_identities), std::move(rank_work));
}

Status DeepSeekEngine::prepare_deferred_native_pipeline(
    DeepSeekPipelinePlanDescriptor descriptor,
    std::vector<DeepSeekRequestIdentity> request_identities,
    DeepSeekDeferredRankComputePlanCompiler& compiler) {
  if (world_size() != 1) {
    return Status::FailedPrecondition(
        "DeepSeek PP1 engine cannot prepare distributed execution");
  }
  if (compiler.world_size() != world_size() ||
      descriptor.phase == DeepSeekPlanPhase::kDrain) {
    return Status::InvalidArgument(
        "DeepSeek deferred native compiler topology is invalid");
  }
  std::vector<DeepSeekRankComputePlanWork> rank_work;
  rank_work.reserve(world_size());
  std::vector<DeepSeekRankPlanRuntime> runtimes;
  runtimes.reserve(world_size());
  std::vector<std::shared_ptr<const DeepSeekRankComputePlanWorkOwner>> owners;
  owners.reserve(world_size());
  std::vector<bool> completion_reported(world_size(), false);

  for (std::uint32_t rank = 0; rank < world_size(); ++rank) {
    auto status = resources_->activate_rank_runtime(rank);
    if (!status.ok()) return status;
  }


  for (std::uint32_t rank = 0; rank < world_size(); ++rank) {
    constexpr std::uintptr_t incoming = 0;
    auto work = compiler.compile(rank, incoming);
    if (!work.ok()) return work.status();
    if (work->lifetime_owner == nullptr
        ) {
      return Status::InvalidArgument(
          "DeepSeek deferred rank work ownership is invalid");
    }
    rank_work.push_back(std::move(*work));
  }
  for (const auto& work : rank_work) owners.push_back(work.lifetime_owner);
  auto status = prepare_pipeline(descriptor, std::move(request_identities));
  if (!status.ok()) return status;
  const auto rollback = [this, &descriptor, &runtimes
                         ](
                            std::uint32_t rank, Status cause) {
    // stage_reject returns the original cause after a successful rollback.
    // The cleared execution owner is the authoritative success signal.
    (void)control_plane_->stage_reject(rank, cause);
    const auto abort = resources_->abort_rank_compute_plans(descriptor);
    if (control_plane_->execution_live() || !abort.ok()) {
      return fail(DeepSeekEngineFailure::kWorkerLost,
                  Status::Internal(
                      "DeepSeek deferred runtime prepare rollback failed"));
    }
    runtimes.clear();
    current_rank_plan_resources_.clear();
    return cause;
  };
  status = resources_->bind_rank_compute_plans(descriptor,
                                               std::move(rank_work));
  if (!status.ok()) return rollback(0, status);

  for (std::uint32_t rank = 0; rank < world_size(); ++rank) {
    auto* bundle = resources_->rank(rank).compute_bundle();
    if (bundle == nullptr) {
      return rollback(rank, Status::FailedPrecondition(
          "DeepSeek deferred runtime lost compute bundle"));
    }
    auto transaction = rank_runtime_resources_[rank].prepare(descriptor);
    if (!transaction.ok()) return rollback(rank, transaction.status());
    auto runtime = DeepSeekRankPlanRuntime::CreateBorrowedCompute(
        std::move(*transaction), bundle->stage(),
        std::move(current_rank_plan_resources_[rank]), bundle->driver());
    if (!runtime.ok()) return rollback(rank, runtime.status());
    status = runtime->mark_ready();
    if (!status.ok()) return rollback(rank, status);
    status = control_plane_->stage_ready(rank);
    if (!status.ok()) return rollback(rank, status);
    runtimes.push_back(std::move(*runtime));
  }
  current_rank_plan_resources_.clear();
  current_rank_runtimes_ = std::move(runtimes);
  current_rank_work_owners_ = std::move(owners);
  rank_completion_reported_ = std::move(completion_reported);
  bound_execution_ = true;
  return Status::Ok();
}

Status DeepSeekEngine::prepare_production_pipeline(
    DeepSeekPipelinePlanDescriptor descriptor,
    std::vector<DeepSeekRequestIdentity> request_identities,
    std::vector<DeepSeekProductionRankPlanSeed> rank_seeds,
    std::uint32_t maximum_sequences, std::uint32_t maximum_tokens,
    std::uint32_t vocabulary_size, bool publish_sampled_token) {
  if (descriptor.engine_epoch != epoch() ||
      request_identities.size() != descriptor.sequence_count ||
      descriptor.sequence_count != 1 || rank_seeds.size() != world_size()) {
    return Status::InvalidArgument(
        "DeepSeek production pipeline identity is invalid");
  }
  const auto admission = admit_request();
  if (!admission.ok()) return admission;
  for (std::uint32_t rank = 0; rank < world_size(); ++rank) {
    const auto activated = resources_->activate_rank_runtime(rank);
    if (!activated.ok())
      return fail(DeepSeekEngineFailure::kWorkerLost, activated);
    auto* rope = resources_->rank(rank).rope_table_device_resources();
    if (rope == nullptr)
      return Status::FailedPrecondition("DeepSeek production RoPE tables are unavailable");
    const auto initialized = rope->ensure_initialized();
    if (!initialized.ok())
      return fail(DeepSeekEngineFailure::kWorkerLost, initialized);
  }
  auto binding = request_attention_binding(
      request_identities.front().request_id,
      request_identities.front().request_generation);
  if (!binding.ok()) return binding.status();
  std::optional<DeepSeekPreparedSamplingInput> prepared_sampling;
  auto sampling = control_plane_->sampling_config(
      request_identities.front().request_id,
      request_identities.front().request_generation);
  if (sampling.ok() && publish_sampled_token) {
    auto snapshot = control_plane_->accepted_token_snapshot(
        request_identities.front().request_id,
        request_identities.front().request_generation);
    if (!snapshot.ok()) return snapshot.status();
    DeepSeekSamplingDescriptor sampling_descriptor{
        sampling->temperature, sampling->top_p, sampling->top_k,
        sampling->effective_seed, snapshot->sample_ordinal,
        sampling->logprobs_enabled, sampling->top_logprobs_count};
    if (snapshot->accepted_completion_count <
        snapshot->minimum_completion_tokens) {
      for (std::uint32_t index = 0; index < sampling->stop_token_count;
           ++index) {
        sampling_descriptor.suppressed_tokens.token_ids[
            sampling_descriptor.suppressed_tokens.token_count++] =
            sampling->stop_token_ids[index];
      }
      const auto* begin = sampling_descriptor.suppressed_tokens.token_ids;
      const auto* end = begin +
          sampling_descriptor.suppressed_tokens.token_count;
      if (std::find(begin, end, 1U) == end) {
        sampling_descriptor.suppressed_tokens.token_ids[
            sampling_descriptor.suppressed_tokens.token_count++] = 1U;
      }
    }
    prepared_sampling = DeepSeekPreparedSamplingInput{
        sampling->config_id, sampling->mode, sampling_descriptor};
  }
  const auto& canonical_tokens = rank_seeds.front().token_ids;
  const auto& canonical_positions = rank_seeds.front().positions;
  const auto canonical_table_positions =
      rank_seeds.front().table_position_count;
  for (std::uint32_t rank = 0; rank < world_size(); ++rank) {
    if (rank_seeds[rank].token_ids != canonical_tokens ||
        rank_seeds[rank].positions != canonical_positions ||
        rank_seeds[rank].table_position_count != canonical_table_positions ||
        !rank_seeds[rank].decode_attention.empty() ||
        !rank_seeds[rank].chunk_attention.empty() ||
        rank_seeds[rank].sampling.has_value()) {
      return Status::InvalidArgument(
          "DeepSeek production logical token input differs across ranks");
    }
    auto* state_pool = resources_->rank(rank).attention_state_pool();
    if (state_pool == nullptr) {
      return Status::FailedPrecondition(
          "DeepSeek production rank attention state is unavailable");
    }
    rank_seeds[rank].attention_shape_seeds.clear();
    for (const auto& layer : state_pool->fixed_layout().descriptors()) {
      DeepSeekAttentionLayerPlanShapeSeed shape;
      shape.layer = layer.layer_id;
      shape.ratio = layer.kind == DeepSeekFixedLayerKind::kRatio4 ? 4U
          : layer.kind == DeepSeekFixedLayerKind::kRatio128 ? 128U : 0U;
      shape.compressed_physical_offset = 128;
      shape.positions = canonical_positions;
      rank_seeds[rank].attention_shape_seeds.push_back(std::move(shape));
    }
    rank_seeds[rank].attention_binding = *binding;
    rank_seeds[rank].sampling = prepared_sampling;
  }
  auto compiler = resources_->assemble_production_deferred_plan_compiler(
      descriptor, std::move(rank_seeds), maximum_sequences, maximum_tokens,
      vocabulary_size);
  if (!compiler.ok()) return compiler.status();
  const auto status = prepare_deferred_native_pipeline(
      descriptor, std::move(request_identities), *compiler);
  if (status.ok()) {
    publish_bound_sampled_token_ = publish_sampled_token;
    bound_sampled_token_staged_ = false;
    bound_sampling_config_ =
        sampling.ok() && publish_sampled_token
            ? std::optional<DeepSeekRequestSamplingConfig>(*sampling)
            : std::nullopt;
    bound_sampling_ = prepared_sampling;
  }
  return status;
}

Status DeepSeekEngine::validate_release_attention_binding(
    std::uint64_t request_id, std::uint64_t request_generation) const {
  const auto iterator = request_attention_bindings_.find(request_id);
  if (iterator == request_attention_bindings_.end() ||
      iterator->second.request_generation != request_generation) {
    return Status::FailedPrecondition(
        "DeepSeek request attention binding was lost before retirement");
  }
  return Status::Ok();
}

Status DeepSeekEngine::release_attention_binding(
    std::uint64_t request_id, std::uint64_t request_generation) {
  const auto validation =
      validate_release_attention_binding(request_id, request_generation);
  if (!validation.ok()) return validation;
  const auto iterator = request_attention_bindings_.find(request_id);
  free_attention_state_slots_.push_back(iterator->second.binding.state_slot);
  request_attention_bindings_.erase(iterator);
  return Status::Ok();
}

Status DeepSeekEngine::prepare_bound_drain_pipeline(
    std::uint64_t plan_sequence,
    std::vector<DeepSeekRequestIdentity> request_identities) {
  if (plan_sequence == 0 || request_identities.empty()) {
    return Status::InvalidArgument(
        "DeepSeek bound drain identity is invalid");
  }
  const auto sequence_count =
      static_cast<std::uint32_t>(request_identities.size());
  return prepare_bound_pipeline(
      {epoch(), plan_sequence, DeepSeekPlanPhase::kDrain, 0,
       sequence_count},
      std::move(request_identities),
      std::vector<DeepSeekRankComputePlanWork>(world_size()));
}

Status DeepSeekEngine::pipeline_stage_ready(std::uint32_t rank) {
  if (bound_execution_) {
    return Status::FailedPrecondition(
        "DeepSeek bound execution owns READY transitions");
  }
  return control_plane_->stage_ready(rank);
}

Status DeepSeekEngine::commit_pipeline() {
  if (bound_execution_) {
    if (current_rank_runtimes_.size() != world_size()) {
      return fail(DeepSeekEngineFailure::kWorkerLost,
                  Status::FailedPrecondition(
                      "DeepSeek bound execution lost rank runtimes"));
    }
    for (std::uint32_t rank = 0; rank < world_size(); ++rank) {
      auto status = resources_->activate_rank_runtime(rank);
      if (!status.ok()) {
        return fail(DeepSeekEngineFailure::kWorkerLost, std::move(status));
      }
      status = current_rank_runtimes_[rank].prepare_commit();
      if (!status.ok()) return fail(DeepSeekEngineFailure::kWorkerLost, status);
    }
    auto status = control_plane_->validate_commit_plan();
    if (!status.ok()) return fail(DeepSeekEngineFailure::kWorkerLost, status);
    for (auto& runtime : current_rank_runtimes_) {
      status = runtime.commit();
      if (!status.ok()) return fail(DeepSeekEngineFailure::kWorkerLost, status);
    }
    status = control_plane_->commit_plan();
    if (!status.ok()) return fail(DeepSeekEngineFailure::kWorkerLost, status);
    return Status::Ok();
  }
  if (current_rank_plan_resources_.size() != world_size()) {
    return Status::FailedPrecondition(
        "DeepSeek engine lost rank plan reservations before COMMIT");
  }
  for (const auto& reservation : current_rank_plan_resources_) {
    const auto status = reservation.validate_commit();
    if (!status.ok()) return status;
  }
  const auto validation = control_plane_->validate_commit_plan();
  if (!validation.ok()) return validation;
  for (auto& reservation : current_rank_plan_resources_) {
    const auto status = reservation.commit();
    if (!status.ok()) return status;
  }
  const auto status = control_plane_->commit_plan();
  if (!status.ok()) current_rank_plan_resources_.clear();
  return status;
}

Status DeepSeekEngine::stage_accepted_tokens(
    std::uint64_t plan_sequence,
    std::vector<DeepSeekAcceptedTokenBatch> batches) {
  return control_plane_->stage_accepted_tokens(
      plan_sequence, std::move(batches));
}

Status DeepSeekEngine::pipeline_stage_complete(std::uint32_t rank) {
  if (bound_execution_) {
    return Status::FailedPrecondition(
        "DeepSeek bound execution owns COMPLETE transitions");
  }
  const auto status = control_plane_->stage_complete(rank);
  if (status.ok() && !control_plane_->execution_live()) {
    current_rank_plan_resources_.clear();
  }
  return status;
}

Status DeepSeekEngine::advance_bound_pipeline() {
  if (!bound_execution_ || current_rank_runtimes_.size() != world_size() ||
      rank_completion_reported_.size() != world_size()) {
    return Status::FailedPrecondition(
        "DeepSeek engine has no bound execution to advance");
  }
  for (std::uint32_t rank = 0; rank < world_size(); ++rank) {
    if (rank_completion_reported_[rank]) continue;
    auto status = resources_->activate_rank_runtime(rank);
    if (!status.ok()) {
      return fail(DeepSeekEngineFailure::kWorkerLost, std::move(status));
    }
    status = current_rank_runtimes_[rank].advance_staged();
    if (!status.ok() && status.code() != StatusCode::kUnavailable) {
      return fail(DeepSeekEngineFailure::kWorkerLost, status);
    }
    if (current_rank_runtimes_[rank].state() ==
        DeepSeekRankPlanRuntimeState::kCompletionReady) {
      if (publish_bound_sampled_token_ && !bound_sampled_token_staged_ &&
          rank + 1 == world_size()) {
        auto* endpoint =
            resources_->rank(rank).endpoint_runtime_resources();
        if (endpoint == nullptr) {
          return fail(DeepSeekEngineFailure::kWorkerLost,
                      Status::FailedPrecondition(
                          "DeepSeek sampled token owner is unavailable"));
        }
        DeepSeekSamplingResult sampled;
        if (bound_sampling_config_.has_value()) {
          auto result = endpoint->sampling_result(
              bound_sampling_config_->top_logprobs_count);
          if (!result.ok()) {
            return fail(DeepSeekEngineFailure::kWorkerLost, result.status());
          }
          sampled = *result;
        } else {
          auto token = endpoint->sampled_token();
          if (!token.ok()) {
            return fail(DeepSeekEngineFailure::kWorkerLost, token.status());
          }
          sampled.token_id = *token;
        }
        if (sampled.token_id >= 129280) {
          return fail(DeepSeekEngineFailure::kWorkerLost,
                      Status::Internal(
                          "DeepSeek sampled token is outside vocabulary"));
        }
        const auto descriptor = current_rank_runtimes_[rank].descriptor();
        const bool request_stop = bound_sampling_config_.has_value() &&
            std::find(bound_sampling_config_->stop_token_ids.begin(),
                      bound_sampling_config_->stop_token_ids.begin() +
                          bound_sampling_config_->stop_token_count,
                      sampled.token_id) !=
                bound_sampling_config_->stop_token_ids.begin() +
                    bound_sampling_config_->stop_token_count;
        const auto finish_reason = sampled.token_id == 1U || request_stop
            ? DeepSeekFinishReason::kStop
            : DeepSeekFinishReason::kNone;
        DeepSeekAcceptedTokenBatch batch{{sampled.token_id}, finish_reason};
        if (bound_sampling_config_.has_value()) {
          DeepSeekAcceptedTokenBatch::SelectedLogprobReceipt receipt{
              bound_sampling_config_->config_id, sampled.selected_logprob};
          receipt.top_logprobs.reserve(sampled.top_token_ids.size());
          for (std::size_t index = 0; index < sampled.top_token_ids.size();
               ++index) {
            receipt.top_logprobs.push_back(
                {sampled.top_token_ids[index], sampled.top_logprobs[index],
                 static_cast<std::uint32_t>(index + 1U)});
          }
          batch.selected_logprobs.push_back(std::move(receipt));
        }
        if (bound_sampling_.has_value() &&
            bound_sampling_->mode == DeepSeekSamplingMode::kStochastic) {
          batch.sampling_receipts.push_back({
              bound_sampling_->config_id,
              bound_sampling_->descriptor.sample_ordinal,
              sampled.selected_logprob, sampled.rng_word});
        }
        status = control_plane_->stage_accepted_tokens(
            descriptor.plan_sequence, {std::move(batch)});
        if (!status.ok()) {
          return fail(DeepSeekEngineFailure::kWorkerLost, status);
        }
        bound_sampled_token_staged_ = true;
      }
      status = control_plane_->prepare_stage_complete(rank);
      if (!status.ok()) return fail(DeepSeekEngineFailure::kWorkerLost, status);
      status = current_rank_runtimes_[rank].complete();
      if (!status.ok()) return fail(DeepSeekEngineFailure::kWorkerLost, status);
      status = control_plane_->stage_complete(rank);
      if (!status.ok()) return fail(DeepSeekEngineFailure::kWorkerLost, status);
      rank_completion_reported_[rank] = true;
    }
  }
  if (std::ranges::all_of(rank_completion_reported_,
                          [](bool reported) { return reported; })) {
    current_rank_runtimes_.clear();
    current_rank_work_owners_.clear();
    rank_completion_reported_.clear();
    bound_execution_ = false;
    publish_bound_sampled_token_ = false;
    bound_sampled_token_staged_ = false;
    bound_sampling_config_.reset();
    bound_sampling_.reset();
  }
  return Status::Ok();
}

Status DeepSeekEngine::acknowledge_output_plan(
    std::uint64_t plan_sequence) {
  return control_plane_->acknowledge_output_plan(plan_sequence);
}

Status DeepSeekEngine::pipeline_stage_failed(std::uint32_t rank,
                                             Status reason) {
  if (bound_execution_) {
    return Status::FailedPrecondition(
        "DeepSeek bound execution owns FAILED transitions");
  }
  const auto status = control_plane_->stage_failed(rank, std::move(reason));
  if (!status.ok()) {
    return fail(DeepSeekEngineFailure::kWorkerLost, status);
  }
  return status;
}

Status DeepSeekEngine::poll_artifacts() {
  if (state_ != DeepSeekEngineState::kReady) return admit_request();
  const auto seal_status = resources_->verify_weight_seals();
  if (!seal_status.ok()) {
    resources_->artifact_guard().report_mapping_fault(0);
    return fail(DeepSeekEngineFailure::kMappingFault, seal_status);
  }
  const auto poll_status = resources_->artifact_guard().poll(*artifact_poller_);
  if (!poll_status.ok()) {
    return fail(DeepSeekEngineFailure::kArtifactIntegrity, poll_status);
  }
  next_artifact_poll_ = std::chrono::steady_clock::now() +
      std::chrono::milliseconds(resources_->artifact_poll_interval_ms());
  return Status::Ok();
}

Status DeepSeekEngine::poll_artifacts_if_due() {
  if (std::chrono::steady_clock::now() < next_artifact_poll_) {
    return Status::Ok();
  }
  return poll_artifacts();
}

Status DeepSeekEngine::report_mapping_fault(std::uint32_t rank) {
  if (state_ != DeepSeekEngineState::kReady) return admit_request();
  const auto status = resources_->artifact_guard().report_mapping_fault(rank);
  if (!status.ok()) return status;
  return fail(DeepSeekEngineFailure::kMappingFault,
              Status::FailedPrecondition(
                  "DeepSeek engine failed after a mapping fault"));
}

Status DeepSeekEngine::report_worker_lost(std::uint32_t rank) {
  if (state_ != DeepSeekEngineState::kReady) return admit_request();
  const auto status = resources_->artifact_guard().report_worker_lost(rank);
  if (!status.ok()) return status;
  return fail(DeepSeekEngineFailure::kWorkerLost,
              Status::FailedPrecondition(
                  "DeepSeek engine failed after losing a worker"));
}

void DeepSeekEngine::fail_all() noexcept {
  // Cleanup failure is a generation failure even when no compute plan was
  // launched. Never admit another request into partially retired state.
  if (state_ != DeepSeekEngineState::kClosed) {
    state_ = DeepSeekEngineState::kFailed;
    if (failure_ == DeepSeekEngineFailure::kNone)
      failure_ = DeepSeekEngineFailure::kWorkerLost;
  }
  const auto abort_status = abort_bound_compute_plans();
  control_plane_->fail_all();
  if (!abort_status.ok()) {
    state_ = DeepSeekEngineState::kFailed;
    failure_ = DeepSeekEngineFailure::kWorkerLost;
    return;
  }
  current_rank_runtimes_.clear();
  current_rank_work_owners_.clear();
  rank_completion_reported_.clear();
  bound_execution_ = false;
  publish_bound_sampled_token_ = false;
  bound_sampled_token_staged_ = false;
  bound_sampling_config_.reset();
  bound_sampling_.reset();
}

Status DeepSeekEngine::begin_close() noexcept {
  if (state_ == DeepSeekEngineState::kClosed ||
      state_ == DeepSeekEngineState::kClosing) {
    return Status::Ok();
  }
  if (state_ == DeepSeekEngineState::kReady) control_plane_->cancel_all();
  if (!bound_execution_ && control_plane_->execution_live()) {
    state_ = DeepSeekEngineState::kFailed;
    failure_ = DeepSeekEngineFailure::kWorkerLost;
    control_plane_->fail_all();
    return Status::FailedPrecondition(
        "DeepSeek manual execution has no verified close frontier");
  }
  const auto compute_abort = abort_bound_compute_plans();
  if (!compute_abort.ok()) {
    state_ = DeepSeekEngineState::kFailed;
    failure_ = DeepSeekEngineFailure::kWorkerLost;
    control_plane_->fail_all();
    return compute_abort;
  }
  current_rank_runtimes_.clear();
  current_rank_work_owners_.clear();
  rank_completion_reported_.clear();
  bound_execution_ = false;
  publish_bound_sampled_token_ = false;
  bound_sampled_token_staged_ = false;
  bound_sampling_config_.reset();
  bound_sampling_.reset();
  state_ = DeepSeekEngineState::kClosing;
  return Status::Ok();
}

Status DeepSeekEngine::advance_close() noexcept {
  if (state_ == DeepSeekEngineState::kClosed) return Status::Ok();
  const auto begin = begin_close();
  if (!begin.ok()) return begin;
  state_ = DeepSeekEngineState::kClosed;
  return Status::Ok();
}

Status DeepSeekEngine::close() noexcept {
  return advance_close();
}

std::string_view DeepSeekEngine::public_error() const noexcept {
  switch (failure_) {
    case DeepSeekEngineFailure::kArtifactIntegrity:
      return "engine_artifact_integrity_failed";
    case DeepSeekEngineFailure::kMappingFault:
    case DeepSeekEngineFailure::kWorkerLost:
      return "engine_unhealthy";
    case DeepSeekEngineFailure::kNone:
      return {};
  }
  return "engine_unhealthy";
}

Status DeepSeekEngine::fail(DeepSeekEngineFailure failure, Status cause) {
  if (state_ == DeepSeekEngineState::kReady) {
    const auto abort_status = abort_bound_compute_plans();
    state_ = DeepSeekEngineState::kFailed;
    failure_ = failure;
    control_plane_->fail_all();
    if (!abort_status.ok()) return abort_status;
    current_rank_runtimes_.clear();
    current_rank_work_owners_.clear();
    rank_completion_reported_.clear();
    bound_execution_ = false;
    publish_bound_sampled_token_ = false;
    bound_sampled_token_staged_ = false;
    bound_sampling_config_.reset();
    bound_sampling_.reset();
  }
  return cause;
}

Status DeepSeekEngine::abort_bound_compute_plans() noexcept {
  // PREPARE submits input copies before a bound compute runtime exists.
  // Retire those copies before releasing any request or engine allocation.
  for (std::uint32_t rank = 0; rank < world_size(); ++rank) {
    auto* rope = resources_->rank(rank).rope_table_device_resources();
    if (rope != nullptr) {
      const auto status = rope->retire_pending();
      if (!status.ok()) return status;
    }
    auto* staging = resources_->rank(rank).request_input_staging_resources();
    if (staging != nullptr) {
      const auto status = staging->retire_pending();
      if (!status.ok()) return status;
    }
  }
  if (!bound_execution_) return Status::Ok();
  if (current_rank_runtimes_.empty()) {
    return Status::FailedPrecondition(
        "DeepSeek bound execution lost its rank runtime ownership");
  }
  return resources_->abort_rank_compute_plans(
      current_rank_runtimes_.front().descriptor());
}

}  // namespace pih
