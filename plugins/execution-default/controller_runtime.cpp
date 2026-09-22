#include "pih/scheduler/controller_runtime.h"

#include <algorithm>
#include <array>
#include <limits>

#include "pih/core/checked_math.h"

namespace pih {
namespace {
constexpr std::string_view kAdmittedDomain =
    "pih-controller-admitted-state-v1";

Status update_u64(Sha256& digest, std::uint64_t value) {
  std::array<std::byte, 8> wire{};
  for (std::size_t i = 0; i < wire.size(); ++i) {
    wire[i] = static_cast<std::byte>(value >> (i * 8U));
  }
  return digest.update(wire);
}
}  // namespace

ControllerRuntime::ControllerRuntime(ControllerRuntime&& other) noexcept
    : epoch_(other.epoch_),
      ingress_(std::move(other.ingress_)),
      slots_(std::move(other.slots_)),
      scheduler_(std::move(other.scheduler_)),
      metadata_(std::move(other.metadata_)),
      output_credits_(std::move(other.output_credits_)),
      pending_output_bursts_(std::move(other.pending_output_bursts_)),
      candidates_(std::move(other.candidates_)),
      candidate_states_(std::move(other.candidate_states_)),
      candidate_tokens_(std::move(other.candidate_tokens_)),
      candidate_slot_indices_(std::move(other.candidate_slot_indices_)),
      selected_slot_indices_(std::move(other.selected_slot_indices_)),
      selected_sampling_(std::move(other.selected_sampling_)),
      completion_digests_(std::move(other.completion_digests_)),
      sampled_completions_(std::move(other.sampled_completions_)),
      sampled_token_storage_(std::move(other.sampled_token_storage_)),
      output_maximum_slots_per_credit_(other.output_maximum_slots_per_credit_),
      output_maximum_bytes_per_credit_(other.output_maximum_bytes_per_credit_),
      output_plan_kind_(other.output_plan_kind_),
      current_plan_(std::move(other.current_plan_)),
      current_output_burst_(std::move(other.current_output_burst_)),
      current_plan_digest_(other.current_plan_digest_),
      current_tier_(other.current_tier_),
      current_selected_count_(other.current_selected_count_),
      maximum_prefill_chunk_tokens_(other.maximum_prefill_chunk_tokens_),
      maximum_execution_bucket_tokens_(other.maximum_execution_bucket_tokens_),
      pad_to_maximum_execution_bucket_(other.pad_to_maximum_execution_bucket_),
      profile_revision_(std::move(other.profile_revision_)),
      resource_vector_hash_(other.resource_vector_hash_),
      eos_token_id_(other.eos_token_id_),
      next_ready_enqueue_sequence_(other.next_ready_enqueue_sequence_),
      next_plan_sequence_(other.next_plan_sequence_),
      last_controller_now_ns_(other.last_controller_now_ns_),
      consecutive_decode_rounds_(other.consecutive_decode_rounds_),
      last_successful_plan_was_protected_prefill_(
          other.last_successful_plan_was_protected_prefill_),
      state_(other.state_.load(std::memory_order_acquire)),
      active_sequence_count_(
          other.active_sequence_count_.load(std::memory_order_acquire)) {}

Result<ControllerRuntime> ControllerRuntime::Create(
    ControllerRuntimeLimits limits) {
  bool resource_hash_nonzero = false;
  for (const auto value : limits.resource_vector_hash.bytes) {
    resource_hash_nonzero = resource_hash_nonzero || value != std::byte{0};
  }
  if (limits.epoch == 0 || limits.maximum_sequences == 0 ||
      limits.maximum_sequences != limits.ingress.requests.maximum_requests ||
      limits.scheduler.maximum_candidates != limits.maximum_sequences ||
      limits.metadata.maximum_sequences <
          limits.scheduler.maximum_sequences_per_plan ||
      limits.metadata.maximum_execution_bucket_tokens <
          limits.scheduler.maximum_real_tokens_per_plan ||
      limits.maximum_prefill_chunk_tokens == 0 ||
      limits.maximum_prefill_chunk_tokens >
          limits.scheduler.maximum_real_tokens_per_plan ||
      limits.profile_revision.empty() || limits.profile_revision.size() > 128 ||
      !resource_hash_nonzero) {
    return Status::InvalidArgument("controller runtime limits are inconsistent");
  }
  auto ingress = ControllerIngress::Create(limits.ingress);
  if (!ingress.ok()) return ingress.status();
  auto scheduler = PhaseTierScheduler::Create(limits.scheduler);
  if (!scheduler.ok()) return scheduler.status();
  auto metadata = PackedTokenMetadataArena::Create(limits.metadata);
  if (!metadata.ok()) return metadata.status();
  auto output_credits = OutputBurstCreditPool::Create(
      {limits.output_burst_credit_count, 1, 5,
       limits.output_maximum_slots_per_credit,
       limits.output_maximum_bytes_per_credit});
  if (!output_credits.ok()) return output_credits.status();
  ControllerRuntime runtime(limits, std::move(*ingress), std::move(*scheduler),
                            std::move(*metadata), std::move(*output_credits));
  runtime.maximum_execution_bucket_tokens_ =
      limits.metadata.maximum_execution_bucket_tokens;
  return runtime;
}

Result<ControllerCommand> ControllerRuntime::submit_admit(
    std::uint64_t request_generation,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t maximum_new_tokens,
    const ControllerRequestSampling& sampling) {
  if (state_.load(std::memory_order_acquire) != ControllerRuntimeState::kReady) {
    return Status::FailedPrecondition("controller runtime is not ready");
  }
  return ingress_.submit_admit(epoch_, request_generation, prompt_token_ids,
                               maximum_new_tokens, sampling);
}

Result<ControllerCommand> ControllerRuntime::submit_cancel(
    std::uint64_t request_generation) {
  if (state_.load(std::memory_order_acquire) != ControllerRuntimeState::kReady)
    return Status::FailedPrecondition("controller runtime is not ready");
  return ingress_.submit_cancel(epoch_, request_generation);
}

Result<Sha256Digest> ControllerRuntime::admitted_state_digest(
    const ControllerAdmissionView& admission) const {
  Sha256 digest;
  Status status = digest.update(std::as_bytes(std::span(kAdmittedDomain)));
  if (status.ok()) status = update_u64(digest, epoch_);
  if (status.ok()) {
    status = update_u64(digest, admission.request.request_generation);
  }
  if (status.ok()) status = update_u64(digest, admission.request.slot_generation);
  if (status.ok()) status = digest.update(admission.request.payload_digest.bytes);
  if (!status.ok()) return status;
  return digest.finalize();
}

Result<ControllerRuntimeStep> ControllerRuntime::step(std::int64_t now_ns) {
  return step_impl(now_ns, nullptr);
}

Result<ControllerRuntimeStep> ControllerRuntime::step(
    std::int64_t now_ns, ControllerAdmissionParticipant& admission) {
  return step_impl(now_ns, &admission);
}

Result<ControllerRuntimeStep> ControllerRuntime::step_impl(
    std::int64_t now_ns, ControllerAdmissionParticipant* participant) {
  if (state_.load(std::memory_order_acquire) != ControllerRuntimeState::kReady) {
    return Status::FailedPrecondition("controller runtime is not ready");
  }
  if (now_ns < last_controller_now_ns_ || next_ready_enqueue_sequence_ == 0) {
    return Status::InvalidArgument("controller time or enqueue sequence is invalid");
  }
  last_controller_now_ns_ = now_ns;
  if (ingress_.queued_command_count() == 0) {
    return ControllerRuntimeStep::kIdle;
  }
  if (ingress_.available_event_credit() == 0) {
    return ControllerRuntimeStep::kOutputBackpressured;
  }
  auto item = ingress_.try_take_command();
  if (!item.ok()) {
    state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
    return item.status();
  }
  if (!item->has_value()) return ControllerRuntimeStep::kIdle;
  if ((**item).command.kind == ControllerCommandKind::kCancel) {
    auto digest = controller_cancel_payload_digest(
        epoch_, (**item).command.request_generation);
    if (!digest.ok() || *digest != (**item).command.payload_digest) {
      state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
      return Status::Internal("controller cancel command digest drifted");
    }
    Slot* target = nullptr;
    for (auto& slot : slots_) {
      if (slot.sequence.has_value() &&
          slot.sequence->sequence_generation() ==
              (**item).command.request_generation) {
        target = &slot;
        break;
      }
    }
    if (target == nullptr)
      return Status::FailedPrecondition("controller cancel target is not active");
    auto cancelled = target->sequence->request_cancel();
    if (!cancelled.ok()) return cancelled;
    if (!target->sequence->has_active_plan()) {
      auto draining = target->sequence->begin_draining();
      if (!draining.ok()) return draining;
      auto event = ingress_.publish_event(
          {ControllerOutputEventKind::kDraining, epoch_,
           target->sequence->sequence_generation(), 0, 0, *digest});
      if (!event.ok()) {
        state_.store(ControllerRuntimeState::kFailed,
                     std::memory_order_release);
        return event.status();
      }
    }
    return ControllerRuntimeStep::kCancelRequested;
  }
  if ((**item).command.kind != ControllerCommandKind::kAdmit ||
      !(**item).request.has_value()) {
    state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
    return Status::Internal("unsupported controller command reached runtime");
  }
  const ControllerAdmissionView admission{(**item).command,
                                           *(**item).request};
  Slot* free_slot = nullptr;
  for (auto& slot : slots_) {
    if (!slot.sequence.has_value()) {
      free_slot = &slot;
      break;
    }
  }
  if (free_slot == nullptr) {
    state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
    return Status::Internal("controller persistent sequence ledger overflowed");
  }
  if (participant != nullptr) {
    const Status reserved = participant->reserve(admission);
    if (!reserved.ok()) {
      const Status released = ingress_.release_request(admission.request);
      if (!released.ok() ||
          reserved.code() != StatusCode::kResourceExhausted) {
        state_.store(ControllerRuntimeState::kFailed,
                     std::memory_order_release);
      }
      if (!released.ok()) {
        return Status::Internal("controller admission reserve cleanup failed");
      }
      if (reserved.code() == StatusCode::kResourceExhausted) {
        auto rejected = ingress_.publish_event(
            {ControllerOutputEventKind::kFailed, epoch_,
             admission.request.request_generation, 0, 0,
             admission.command.payload_digest});
        if (!rejected.ok()) {
          state_.store(ControllerRuntimeState::kFailed,
                       std::memory_order_release);
          return rejected.status();
        }
        return ControllerRuntimeStep::kAdmissionRejected;
      }
      return reserved;
    }
  }
  auto sequence = ControllerSequence::Admit(
      admission.request.request_generation,
      admission.request.prompt_token_ids.size(), 1);
  auto state_digest = admitted_state_digest(admission);
  if (!sequence.ok() || !state_digest.ok()) {
    Status rollback = Status::Ok();
    if (participant != nullptr) {
      rollback = participant->rollback(admission.request.request_generation);
    }
    const auto release = ingress_.release_request(admission.request);
    state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
    if (!rollback.ok() || !release.ok()) {
      return Status::Internal("controller admission cleanup failed");
    }
    return sequence.ok() ? state_digest.status() : sequence.status();
  }
  free_slot->request = admission.request;
  free_slot->sequence.emplace(std::move(*sequence));
  free_slot->ready_enqueue_sequence = next_ready_enqueue_sequence_++;
  free_slot->prefill_eligible_since_ns = now_ns;
  if (participant != nullptr) {
    const Status published = participant->publish(
        admission.request.request_generation);
    if (!published.ok()) {
      const auto release = ingress_.release_request(free_slot->request);
      free_slot->sequence.reset();
      state_.store(ControllerRuntimeState::kFailed,
                   std::memory_order_release);
      if (!release.ok()) {
        return Status::Internal("controller admission publish cleanup failed");
      }
      return published;
    }
  }
  auto event = ingress_.publish_event(
      {ControllerOutputEventKind::kAdmitted, epoch_,
       admission.request.request_generation, 0, 0, *state_digest});
  if (!event.ok()) {
    const auto release = ingress_.release_request(free_slot->request);
    free_slot->sequence.reset();
    state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
    if (!release.ok()) {
      return Status::Internal("controller admitted-event rollback failed");
    }
    return event.status();
  }
  active_sequence_count_.fetch_add(1, std::memory_order_release);
  return ControllerRuntimeStep::kAdmitted;
}

Result<std::optional<ControllerPreparedPlanView>>
ControllerRuntime::prepare_next_plan(std::int64_t now_ns) {
  if (state_.load(std::memory_order_acquire) != ControllerRuntimeState::kReady ||
      current_plan_.has_value()) {
    return Status::FailedPrecondition("controller cannot prepare another plan");
  }
  if (now_ns < last_controller_now_ns_ || next_plan_sequence_ == 0) {
    return Status::InvalidArgument("controller plan identity is invalid");
  }
  last_controller_now_ns_ = now_ns;
  std::uint32_t candidate_count = 0;
  for (std::uint32_t slot_index = 0; slot_index < slots_.size(); ++slot_index) {
    auto& slot = slots_[slot_index];
    if (!slot.sequence.has_value()) continue;
    auto phase = slot.sequence->ready_phase();
    if (!phase.ok()) continue;
    // Draft-token ownership is added with the speculative ledger. Until then,
    // a VERIFY-ready sequence cannot be represented by this payload arena.
    if (*phase == PackedTokenPhase::kVerify) {
      return Status::FailedPrecondition(
          "verify-ready sequence has no bound draft payload");
    }
    auto count = slot.sequence->next_real_token_count(
        maximum_prefill_chunk_tokens_, 0);
    auto produces_logits = slot.sequence->next_plan_produces_logits(
        maximum_prefill_chunk_tokens_, 0);
    if (!count.ok() || !produces_logits.ok()) {
      return count.ok() ? produces_logits.status() : count.status();
    }
    const auto start = slot.sequence->model_processed_length();
    std::span<const std::uint32_t> token_span;
    if (*phase == PackedTokenPhase::kPrefill) {
      if (start > slot.request.prompt_token_ids.size() ||
          *count > slot.request.prompt_token_ids.size() - start) {
        return Status::Internal("controller prompt chunk mapping drifted");
      }
      token_span = slot.request.prompt_token_ids.subspan(
          static_cast<std::size_t>(start), *count);
    } else {
      if (*phase != PackedTokenPhase::kDecode || !slot.pending_token.has_value() ||
          *count != 1) {
        return Status::FailedPrecondition(
            "decode-ready sequence has no valid pending token");
      }
      token_span = std::span<const std::uint32_t>(&*slot.pending_token, 1);
    }
    auto input_digest = packed_token_input_digest(token_span);
    if (!input_digest.ok()) return input_digest.status();
    candidates_[candidate_count] = SchedulerCandidate{
        slot.sequence->sequence_generation(), *phase,
        slot.last_service_plan_sequence, slot.ready_enqueue_sequence,
        slot.prefill_eligible_since_ns, *count,
        SchedulerInfeasibility::kNone};
    candidate_states_[candidate_count] = ScheduledSequenceState{
        slot.sequence->sequence_generation(), start,
        slot.sequence->state_generation(), *input_digest};
    candidate_tokens_[candidate_count] = ScheduledSequenceTokens{
        slot.sequence->sequence_generation(), token_span, *produces_logits};
    candidate_slot_indices_[candidate_count] = slot_index;
    ++candidate_count;
  }
  if (candidate_count == 0) {
    return std::optional<ControllerPreparedPlanView>{};
  }
  auto decision = scheduler_.select(
      {now_ns, consecutive_decode_rounds_,
       last_successful_plan_was_protected_prefill_,
       std::span(candidates_).first(candidate_count)});
  if (!decision.ok()) return decision.status();
  if (decision->tier == SchedulerTier::kIdle) {
    return std::optional<ControllerPreparedPlanView>{};
  }
  const auto execution_bucket = pad_to_maximum_execution_bucket_
                                    ? maximum_execution_bucket_tokens_
                                    : decision->total_real_tokens;
  auto plan = scheduler_.assemble_current_plan(
      decision->generation,
      {epoch_, next_plan_sequence_, profile_revision_, resource_vector_hash_,
       execution_bucket, maximum_execution_bucket_tokens_},
      std::span(candidate_states_).first(candidate_count));
  if (!plan.ok()) return plan.status();
  auto plan_digest = plan->semantic_digest();
  if (!plan_digest.ok()) return plan_digest.status();
  auto metadata = scheduler_.materialize_current_metadata(
      decision->generation, *plan,
      std::span(candidate_tokens_).first(candidate_count), metadata_);
  if (!metadata.ok()) return metadata.status();
  std::uint32_t maximum_records = 0;
  for (const auto candidate_index : decision->included_candidate_indices) {
    maximum_records = std::max(
        maximum_records,
        candidate_tokens_[candidate_index].produces_logits ? 1U : 0U);
  }
  auto output_burst = output_credits_.acquire(
      next_plan_sequence_, output_plan_kind_, maximum_records,
      output_maximum_slots_per_credit_, output_maximum_bytes_per_credit_);
  if (!output_burst.ok()) return output_burst.status();
  std::uint32_t prepared_count = 0;
  for (const auto candidate_index : decision->included_candidate_indices) {
    const auto slot_index = candidate_slot_indices_[candidate_index];
    auto& slot = slots_[slot_index];
    const auto status = slot.sequence->prepare(
        next_plan_sequence_, plan->phase(),
        candidates_[candidate_index].real_token_count);
    if (!status.ok()) {
      for (std::uint32_t i = 0; i < prepared_count; ++i) {
        const auto rollback_slot = selected_slot_indices_[i];
        const auto rollback = slots_[rollback_slot].sequence->abort_prepared(
            next_plan_sequence_);
        if (!rollback.ok()) {
          state_.store(ControllerRuntimeState::kFailed,
                       std::memory_order_release);
          return Status::Internal("controller plan prepare rollback failed");
        }
      }
      const auto credit_rollback = output_credits_.abort(*output_burst);
      if (!credit_rollback.ok()) {
        state_.store(ControllerRuntimeState::kFailed,
                     std::memory_order_release);
        return Status::Internal("controller output credit rollback failed");
      }
      return status;
    }
    selected_slot_indices_[prepared_count] = slot_index;
    selected_sampling_[prepared_count] = slot.request.sampling;
    selected_sampling_[prepared_count].sample_ordinal =
        slot.sequence->accepted_completion_count();
    ++prepared_count;
  }
  current_selected_count_ = prepared_count;
  current_output_burst_.emplace(*output_burst);
  current_plan_.emplace(std::move(*plan));
  current_plan_digest_ = *plan_digest;
  current_tier_ = decision->tier;
  ++next_plan_sequence_;
  return std::optional<ControllerPreparedPlanView>{ControllerPreparedPlanView{
      &*current_plan_, *metadata,
      std::span(selected_slot_indices_).first(current_selected_count_),
      std::span(selected_sampling_).first(current_selected_count_),
      *plan_digest}};
}

Status ControllerRuntime::commit_current_plan(std::uint64_t plan_sequence,
                                              Sha256Digest plan_digest) {
  if (!current_plan_.has_value() || !current_output_burst_.has_value() ||
      current_plan_->plan_sequence() != plan_sequence ||
      current_plan_digest_ != plan_digest) {
    return Status::FailedPrecondition("controller plan commit identity drifted");
  }
  auto credit = output_credits_.commit(*current_output_burst_);
  if (!credit.ok()) return credit;
  for (std::uint32_t i = 0; i < current_selected_count_; ++i) {
    const auto status = slots_[selected_slot_indices_[i]].sequence->commit(
        plan_sequence);
    if (!status.ok()) {
      state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
      return Status::Internal("controller plan commit became partial");
    }
  }
  return Status::Ok();
}

Status ControllerRuntime::mark_current_plan_in_flight(
    std::uint64_t plan_sequence, Sha256Digest plan_digest) {
  if (!current_plan_.has_value() || current_plan_->plan_sequence() != plan_sequence ||
      current_plan_digest_ != plan_digest) {
    return Status::FailedPrecondition("controller in-flight identity drifted");
  }
  for (std::uint32_t i = 0; i < current_selected_count_; ++i) {
    const auto status =
        slots_[selected_slot_indices_[i]].sequence->mark_in_flight(plan_sequence);
    if (!status.ok()) {
      state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
      return Status::Internal("controller in-flight publication became partial");
    }
  }
  return Status::Ok();
}

void ControllerRuntime::clear_current_plan() noexcept {
  current_plan_.reset();
  current_plan_digest_ = {};
  current_tier_ = SchedulerTier::kIdle;
  current_selected_count_ = 0;
  current_output_burst_.reset();
}

Status ControllerRuntime::abort_current_plan(std::uint64_t plan_sequence,
                                             Sha256Digest plan_digest) {
  if (!current_plan_.has_value() || !current_output_burst_.has_value() ||
      current_plan_->plan_sequence() != plan_sequence ||
      current_plan_digest_ != plan_digest) {
    return Status::FailedPrecondition("controller plan abort identity drifted");
  }
  for (std::uint32_t i = 0; i < current_selected_count_; ++i) {
    const auto status = slots_[selected_slot_indices_[i]].sequence->abort_prepared(
        plan_sequence);
    if (!status.ok()) {
      state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
      return Status::Internal("controller plan abort became partial");
    }
  }
  auto credit = output_credits_.abort(*current_output_burst_);
  if (!credit.ok()) {
    state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
    return Status::Internal("controller output credit abort failed");
  }
  clear_current_plan();
  return Status::Ok();
}

Result<Sha256Digest> ControllerRuntime::completion_state_digest(
    const ControllerPlanCompletion& completion) const {
  constexpr std::string_view domain = "pih-controller-completion-state-v1";
  Sha256 digest;
  Status status = digest.update(std::as_bytes(std::span(domain)));
  if (status.ok()) status = digest.update(current_plan_digest_.bytes);
  for (const auto value : {completion.plan_sequence,
                           completion.sequence_generation,
                           completion.committed_model_processed_length,
                           completion.accepted_completion_count,
                           completion.state_generation,
                           static_cast<std::uint64_t>(completion.next_decode_phase),
                           static_cast<std::uint64_t>(completion.terminal)}) {
    if (status.ok()) status = update_u64(digest, value);
  }
  if (!status.ok()) return status;
  return digest.finalize();
}

Status ControllerRuntime::complete_current_plan(
    std::span<const ControllerBackendSequenceResult> results,
    std::int64_t now_ns) {
  if (!current_plan_.has_value() || !current_output_burst_.has_value() ||
      results.size() != current_selected_count_ ||
      now_ns < last_controller_now_ns_ || next_ready_enqueue_sequence_ == 0) {
    return Status::FailedPrecondition("controller completion batch is invalid");
  }
  std::uint64_t required_events = 0;
  std::uint64_t required_requeues = 0;
  for (std::uint32_t i = 0; i < current_selected_count_; ++i) {
    const auto& result = results[i];
    const auto& sequence = *slots_[selected_slot_indices_[i]].sequence;
    const auto validation = sequence.validate_completion(result.completion);
    if (!validation.ok()) return validation;
    const auto previous = sequence.accepted_completion_count();
    if (result.completion.accepted_completion_count - previous !=
        result.accepted_token_ids.size()) {
      return Status::InvalidArgument("controller accepted-token result drifted");
    }
    for (const auto token : result.accepted_token_ids) {
      if (token >= 151936) {
        return Status::InvalidArgument("controller backend token is invalid");
      }
    }
    const bool will_drain =
        result.completion.terminal ||
        sequence.state() == ControllerSequenceState::kCancelRequested;
    if (result.completion.accepted_completion_count >
            slots_[selected_slot_indices_[i]].request.maximum_new_tokens ||
        (result.completion.accepted_completion_count ==
             slots_[selected_slot_indices_[i]].request.maximum_new_tokens &&
         !result.completion.terminal)) {
      return Status::InvalidArgument(
          "controller generation limit requires terminal completion");
    }
    const bool will_need_pending_token =
        !will_drain && result.completion.committed_model_processed_length >=
                           sequence.prompt_token_count();
    if (will_need_pending_token && result.accepted_token_ids.empty()) {
      return Status::InvalidArgument(
          "controller decode boundary requires a pending token");
    }
    if (result.accepted_token_ids.size() >
        std::numeric_limits<std::uint64_t>::max() - required_events) {
      return Status::ResourceExhausted("controller output event count overflowed");
    }
    required_events += result.accepted_token_ids.size();
    if (will_drain && required_events ==
                          std::numeric_limits<std::uint64_t>::max()) {
      return Status::ResourceExhausted("controller output event count overflowed");
    }
    required_events += will_drain ? 1U : 0U;
    required_requeues += will_drain ? 0U : 1U;
    auto digest = completion_state_digest(result.completion);
    if (!digest.ok()) return digest.status();
    completion_digests_[i] = *digest;
  }
  if (required_events > ingress_.available_event_credit()) {
    return Status::ResourceExhausted("controller output event credit is unavailable");
  }
  if (required_requeues != 0 &&
      next_ready_enqueue_sequence_ >
          std::numeric_limits<std::uint64_t>::max() - required_requeues + 1) {
    return Status::FailedPrecondition("controller ready enqueue sequence wrapped");
  }
  auto transferred = output_credits_.transfer(*current_output_burst_);
  if (!transferred.ok()) return transferred;
  last_controller_now_ns_ = now_ns;
  const auto completed_plan_sequence = current_plan_->plan_sequence();
  const auto completed_phase = current_plan_->phase();
  const auto completed_tier = current_tier_;
  std::uint32_t plan_event_index = 0;
  if (completed_phase != PackedTokenPhase::kPrefill &&
      consecutive_decode_rounds_ == std::numeric_limits<std::uint32_t>::max()) {
    state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
    return Status::Internal("controller decode round counter wrapped");
  }
  for (std::uint32_t i = 0; i < current_selected_count_; ++i) {
    auto& slot = slots_[selected_slot_indices_[i]];
    const auto previous = slot.sequence->accepted_completion_count();
    const auto completion = slot.sequence->complete(results[i].completion);
    if (!completion.ok()) {
      state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
      return Status::Internal("validated controller completion failed to apply");
    }
    for (std::size_t token_index = 0;
         token_index < results[i].accepted_token_ids.size(); ++token_index) {
      auto event = ingress_.publish_event(
          {ControllerOutputEventKind::kTokenCommitted, epoch_,
           slot.sequence->sequence_generation(), previous + token_index + 1,
           results[i].accepted_token_ids[token_index], completion_digests_[i],
           completed_plan_sequence, plan_event_index++,
           static_cast<std::uint32_t>(required_events)});
      if (!event.ok()) {
        state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
        return Status::Internal("reserved controller token event publish failed");
      }
    }
    if (slot.sequence->state() == ControllerSequenceState::kDraining) {
      slot.pending_token.reset();
      const auto finish_reason = results[i].completion.terminal
          ? (results[i].completion.accepted_completion_count ==
                     slot.request.maximum_new_tokens
                 ? ControllerFinishReason::kLength
                 : ControllerFinishReason::kStop)
          : ControllerFinishReason::kNone;
      auto event = ingress_.publish_event(
          {ControllerOutputEventKind::kDraining, epoch_,
           slot.sequence->sequence_generation(), 0, 0, completion_digests_[i],
           completed_plan_sequence, plan_event_index++,
           static_cast<std::uint32_t>(required_events), finish_reason});
      if (!event.ok()) {
        state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
        return Status::Internal("reserved controller drain event publish failed");
      }
    } else {
      if (slot.sequence->state() == ControllerSequenceState::kReadyDecode) {
        slot.pending_token = results[i].accepted_token_ids.back();
      }
      slot.ready_enqueue_sequence = next_ready_enqueue_sequence_++;
      slot.prefill_eligible_since_ns = now_ns;
    }
    slot.last_service_plan_sequence = completed_plan_sequence;
  }
  if (completed_phase == PackedTokenPhase::kPrefill) {
    consecutive_decode_rounds_ = 0;
    last_successful_plan_was_protected_prefill_ =
        completed_tier == SchedulerTier::kProtectedPrefill;
  } else {
    ++consecutive_decode_rounds_;
    last_successful_plan_was_protected_prefill_ = false;
  }
  // Nonterminal prefill chunks publish no events. There is no consumer-visible
  // plan ID to acknowledge, so retaining their transferred credit would fill
  // the bounded output pool before a long prompt reaches its sampling frontier.
  if (required_events == 0) {
    const auto released = output_credits_.release(*current_output_burst_);
    if (!released.ok()) {
      state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
      return released;
    }
    clear_current_plan();
    return Status::Ok();
  }
  auto pending = std::find_if(
      pending_output_bursts_.begin(), pending_output_bursts_.end(),
      [](const auto& lease) { return !lease.has_value(); });
  if (pending == pending_output_bursts_.end()) {
    state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
    return Status::Internal("controller transferred output credit has no owner slot");
  }
  pending->emplace(*current_output_burst_);
  clear_current_plan();
  return Status::Ok();
}

Status ControllerRuntime::acknowledge_output_plan(std::uint64_t plan_sequence) {
  if (plan_sequence == 0) {
    return Status::InvalidArgument("controller output ACK plan is zero");
  }
  auto pending = std::find_if(
      pending_output_bursts_.begin(), pending_output_bursts_.end(),
      [plan_sequence](const auto& lease) {
        return lease.has_value() && lease->plan_sequence == plan_sequence;
      });
  if (pending == pending_output_bursts_.end()) {
    return Status::FailedPrecondition("controller output ACK is stale or forged");
  }
  auto released = output_credits_.release(**pending);
  if (!released.ok()) {
    state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
    return Status::Internal("controller output ACK release failed");
  }
  pending->reset();
  return Status::Ok();
}

Status ControllerRuntime::complete_current_sampled_tokens(
    std::span<const std::uint32_t> sampled_token_ids,
    std::int64_t now_ns) {
  if (!current_plan_.has_value() ||
      current_plan_->sequence_count() != current_selected_count_) {
    return Status::FailedPrecondition(
        "controller has no canonical sampled completion plan");
  }
  std::size_t next_sample = 0;
  for (std::uint32_t i = 0; i < current_selected_count_; ++i) {
    auto& slot = slots_[selected_slot_indices_[i]];
    const auto& sequence = *slot.sequence;
    auto committed = checked_add_u64(
        sequence.model_processed_length(), current_plan_->real_token_counts()[i]);
    if (!committed.ok() || sequence.state_generation() == UINT64_MAX)
      return Status::ResourceExhausted(
          "controller sampled completion identity exhausted");
    const bool produces_token =
        current_plan_->phase() != PackedTokenPhase::kPrefill ||
        *committed == sequence.prompt_token_count();
    if (produces_token && next_sample >= sampled_token_ids.size())
      return Status::InvalidArgument("controller sampled token count is short");
    const auto accepted = sequence.accepted_completion_count() +
                          static_cast<std::uint64_t>(produces_token);
    if (accepted < sequence.accepted_completion_count())
      return Status::ResourceExhausted(
          "controller accepted token identity exhausted");
    sampled_token_storage_[i] =
        produces_token ? sampled_token_ids[next_sample++] : 0;
    bool sampled_stop =
        produces_token && accepted >= slot.request.sampling.minimum_new_tokens &&
        eos_token_id_ != UINT32_MAX &&
        sampled_token_storage_[i] == eos_token_id_;
    if (produces_token &&
        accepted >= slot.request.sampling.minimum_new_tokens) {
      for (std::uint32_t stop = 0;
           stop < slot.request.sampling.stop_token_count; ++stop) {
        if (sampled_token_storage_[i] ==
            slot.request.sampling.stop_token_ids[stop]) {
          sampled_stop = true;
          break;
        }
      }
    }
    sampled_completions_[i] = ControllerBackendSequenceResult{
        {current_plan_->plan_sequence(), sequence.sequence_generation(),
         *committed, accepted, sequence.state_generation() + 1,
         PackedTokenPhase::kDecode,
         accepted == slot.request.maximum_new_tokens || sampled_stop},
        produces_token
            ? std::span<const std::uint32_t>(&sampled_token_storage_[i], 1)
            : std::span<const std::uint32_t>{}};
  }
  if (next_sample != sampled_token_ids.size())
    return Status::InvalidArgument("controller sampled token count is long");
  return complete_current_plan(
      std::span(sampled_completions_).first(current_selected_count_), now_ns);
}

Status ControllerRuntime::finalize_draining(
    std::uint64_t request_generation) {
  if (state_.load(std::memory_order_acquire) != ControllerRuntimeState::kReady ||
      ingress_.available_event_credit() == 0)
    return Status::ResourceExhausted(
        "controller terminal event credit is unavailable");
  Slot* target = nullptr;
  for (auto& slot : slots_) {
    if (slot.sequence.has_value() &&
        slot.sequence->sequence_generation() == request_generation) {
      target = &slot;
      break;
    }
  }
  if (target == nullptr)
    return Status::FailedPrecondition("controller drain target is not active");
  auto finished = target->sequence->finish_cancel();
  auto kind = ControllerOutputEventKind::kCancelled;
  if (!finished.ok()) {
    finished = target->sequence->finish_drain();
    kind = ControllerOutputEventKind::kCompleted;
  }
  if (!finished.ok()) return finished;
  const auto digest = target->request.payload_digest;
  auto released = ingress_.release_request(target->request);
  if (!released.ok()) {
    state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
    return Status::Internal("controller request release failed after drain");
  }
  auto event = ingress_.publish_event(
      {kind, epoch_, request_generation, 0, 0, digest});
  if (!event.ok()) {
    state_.store(ControllerRuntimeState::kFailed, std::memory_order_release);
    return Status::Internal("reserved controller terminal event publish failed");
  }
  target->pending_token.reset();
  target->sequence.reset();
  target->request = {};
  active_sequence_count_.fetch_sub(1, std::memory_order_release);
  return Status::Ok();
}

Result<std::optional<ControllerOutputEvent>>
ControllerRuntime::try_take_event() {
  return ingress_.try_take_event();
}

std::uint32_t ControllerRuntime::active_sequence_count() const noexcept {
  return active_sequence_count_.load(std::memory_order_acquire);
}

std::uint32_t ControllerRuntime::queued_command_count() const {
  return ingress_.queued_command_count();
}

bool ControllerRuntime::retirable() const {
  if (state() != ControllerRuntimeState::kReady ||
      active_sequence_count() != 0 || queued_command_count() != 0 ||
      ingress_.queued_event_count() != 0 || current_plan_.has_value() ||
      current_output_burst_.has_value() ||
      output_credits_.available() != pending_output_bursts_.size())
    return false;
  for (const auto& burst : pending_output_bursts_)
    if (burst.has_value()) return false;
  return true;
}

}  // namespace pih
