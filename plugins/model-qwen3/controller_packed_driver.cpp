#include "pih/scheduler/controller_packed_driver.h"
#include "controller_client.h"

#include <algorithm>

namespace pih {

Result<ControllerPackedDriver> ControllerPackedDriver::Create(
    qwen_plugin::ControllerClient& runtime, QwenBf16PackedBatchBackend& backend,
    std::uint32_t maximum_sequences,
    QwenBf16PackedKvMetadataLimits kv_metadata_limits,
    std::uint32_t receipt_capacity) {
  if (maximum_sequences == 0 ||
      kv_metadata_limits.maximum_sequences < maximum_sequences ||
      receipt_capacity < maximum_sequences)
    return Status::InvalidArgument("controller packed driver limits drifted");
  auto transaction = QwenBf16PackedBatchTransaction::Create(maximum_sequences);
  if (!transaction.ok()) return transaction.status();
  auto metadata = QwenBf16PackedKvMetadataArena::Create(kv_metadata_limits);
  if (!metadata.ok()) return metadata.status();
  return ControllerPackedDriver(runtime, backend, std::move(*transaction),
                                std::move(*metadata), maximum_sequences,
                                receipt_capacity);
}

Status ControllerPackedDriver::bind_sequence(
    std::uint64_t sequence_generation, QwenKvBlockTable& block_table,
    QwenKvCompletionEvent initial_last_use_event) {
  if (state_ != ControllerPackedDriverState::kReady ||
      sequence_generation == 0 || block_table.descriptor().active != 1 ||
      initial_last_use_event.handle == 0 ||
      initial_last_use_event.generation == 0)
    return Status::FailedPrecondition("controller KV binding is not ready");
  for (std::uint32_t i = 0; i < registry_count_; ++i) {
    if (registry_[i].sequence_generation == sequence_generation ||
        registry_[i].block_table == &block_table)
      return Status::FailedPrecondition(
          "controller KV binding already exists");
  }
  if (registry_count_ == registry_.size())
    return Status::ResourceExhausted("controller KV registry is full");
  registry_[registry_count_] = {sequence_generation, &block_table};
  last_use_events_[registry_count_] = initial_last_use_event;
  reclaimed_[registry_count_] = false;
  ++registry_count_;
  return Status::Ok();
}

Result<ControllerPackedDriverStep> ControllerPackedDriver::publish_pending(
    std::int64_t now_ns) {
  if (pending_token_count_ > receipts_.size() - receipt_size_)
    return ControllerPackedDriverStep::kOutputBackpressured;
  auto status = runtime_->complete_current_sampled_tokens(
      std::span(pending_tokens_).first(pending_token_count_), now_ns);
  if (!status.ok()) {
    if (status.code() == StatusCode::kResourceExhausted)
      return ControllerPackedDriverStep::kOutputBackpressured;
    state_ = ControllerPackedDriverState::kPoisoned;
    return status;
  }
  for (std::uint32_t i = 0; i < pending_token_count_; ++i) {
    const auto tail = (receipt_head_ + receipt_size_) % receipts_.size();
    receipts_[tail] = pending_receipts_[i];
    ++receipt_size_;
  }
  pending_token_count_ = 0;
  state_ = ControllerPackedDriverState::kReady;
  return ControllerPackedDriverStep::kCompleted;
}

Result<std::optional<ControllerPackedSamplingReceipt>>
ControllerPackedDriver::try_take_sampling_receipt() {
  if (receipt_size_ == 0)
    return std::optional<ControllerPackedSamplingReceipt>{};
  auto receipt = receipts_[receipt_head_];
  receipt_head_ = (receipt_head_ + 1) % receipts_.size();
  --receipt_size_;
  return std::optional<ControllerPackedSamplingReceipt>{receipt};
}

Result<ControllerPackedDriverStep> ControllerPackedDriver::execute_next(
    std::int64_t now_ns) {
  if (state_ == ControllerPackedDriverState::kPoisoned)
    return Status::FailedPrecondition("controller packed driver is poisoned");
  if (state_ == ControllerPackedDriverState::kCompletionPending)
    return publish_pending(now_ns);
  auto prepared = runtime_->prepare_next_plan(now_ns);
  if (!prepared.ok()) return prepared.status();
  if (!prepared->has_value()) return ControllerPackedDriverStep::kIdle;
  const auto& view = **prepared;
  const auto sequence_count = view.plan->sequence_count();
  if (view.selected_sampling.size() != sequence_count) {
    const auto aborted = runtime_->abort_current_plan(
        view.plan->plan_sequence(), view.plan_digest);
    if (!aborted.ok()) state_ = ControllerPackedDriverState::kPoisoned;
    return Status::Internal("controller sampling projection drifted");
  }
  for (std::size_t i = 0; i < sequence_count; ++i) {
    const auto generation = view.plan->ordered_sequence_generations()[i];
    bool found = false;
    for (std::uint32_t candidate = 0; candidate < registry_count_; ++candidate) {
      if (registry_[candidate].sequence_generation == generation) {
        active_bindings_[i] = registry_[candidate];
        found = true;
        break;
      }
    }
    if (!found) {
      auto aborted = runtime_->abort_current_plan(
          view.plan->plan_sequence(), view.plan_digest);
      if (!aborted.ok()) state_ = ControllerPackedDriverState::kPoisoned;
      return Status::FailedPrecondition("controller sequence has no KV binding");
    }
    const auto& source = view.selected_sampling[i];
    auto& projected = active_sampling_[i];
    projected = {};
    projected.mode = source.mode == ControllerSamplingMode::kGreedy
                         ? Qwen3SamplingMode::kGreedy
                         : Qwen3SamplingMode::kStochastic;
    projected.temperature = source.temperature;
    projected.top_p = source.top_p;
    projected.top_k = source.top_k;
    projected.seed = source.effective_seed;
    projected.sample_ordinal = source.sample_ordinal;
    projected.top_logprobs_count =
        source.logprobs_enabled ? source.top_logprobs_count : 0U;
    if (source.sample_ordinal < source.minimum_new_tokens) {
      for (std::uint32_t stop = 0; stop < source.stop_token_count; ++stop) {
        projected.suppressed_token_ids[projected.suppressed_token_count++] =
            source.stop_token_ids[stop];
      }
      if (runtime_->eos_token_id() != UINT32_MAX &&
          std::find(projected.suppressed_token_ids.begin(),
                    projected.suppressed_token_ids.begin() +
                        projected.suppressed_token_count,
                    runtime_->eos_token_id()) ==
              projected.suppressed_token_ids.begin() +
                  projected.suppressed_token_count) {
        projected.suppressed_token_ids[projected.suppressed_token_count++] =
            runtime_->eos_token_id();
      }
    }
  }
  auto status = runtime_->commit_current_plan(
      view.plan->plan_sequence(), view.plan_digest);
  if (status.ok()) status = runtime_->mark_current_plan_in_flight(
      view.plan->plan_sequence(), view.plan_digest);
  if (!status.ok()) {
    state_ = ControllerPackedDriverState::kPoisoned;
    return status;
  }
  auto execution = transaction_.execute(
      *view.plan, view.metadata,
      std::span(active_bindings_).first(sequence_count),
      std::span(active_sampling_).first(sequence_count), metadata_, *backend_);
  if (!execution.ok()) {
    state_ = ControllerPackedDriverState::kPoisoned;
    return execution.status();
  }
  for (std::size_t active = 0; active < sequence_count; ++active) {
    for (std::uint32_t registered = 0; registered < registry_count_;
         ++registered) {
      if (registry_[registered].sequence_generation ==
          active_bindings_[active].sequence_generation) {
        last_use_events_[registered] = execution->completion_event;
        break;
      }
    }
  }
  if (execution->sampled_token_ids.size() > pending_tokens_.size() ||
      execution->sampling_receipts.size() !=
          execution->sampled_token_ids.size()) {
    state_ = ControllerPackedDriverState::kPoisoned;
    return Status::Internal("controller backend exceeded token capacity");
  }
  pending_token_count_ =
      static_cast<std::uint32_t>(execution->sampled_token_ids.size());
  std::copy(execution->sampled_token_ids.begin(),
            execution->sampled_token_ids.end(), pending_tokens_.begin());
  std::size_t sample_index = 0;
  for (std::size_t active = 0; active < sequence_count; ++active) {
    const bool produces_token =
        sample_index < view.metadata.sample_row_index.size() &&
        view.metadata.sample_row_index[sample_index] ==
            view.plan->packed_offsets()[active + 1] - 1;
    if (!produces_token) continue;
    if (execution->sampling_receipts[sample_index].token_id !=
        pending_tokens_[sample_index]) {
      state_ = ControllerPackedDriverState::kPoisoned;
      return Status::Internal("controller sampling receipt token drifted");
    }
    pending_receipts_[sample_index] = ControllerPackedSamplingReceipt{
        active_bindings_[active].sequence_generation,
        active_sampling_[active].sample_ordinal + 1,
        execution->sampling_receipts[sample_index]};
    ++sample_index;
  }
  if (sample_index != pending_token_count_) {
    state_ = ControllerPackedDriverState::kPoisoned;
    return Status::Internal("controller sampling receipt identity drifted");
  }
  state_ = ControllerPackedDriverState::kCompletionPending;
  return publish_pending(now_ns);
}

Status ControllerPackedDriver::drain_sequence(
    std::uint64_t sequence_generation, QwenKvSlotPool& pool,
    QwenBf16KvRecycler& recycler) {
  if (state_ != ControllerPackedDriverState::kReady)
    return Status::FailedPrecondition(
        "controller driver cannot drain while work is pending");
  std::uint32_t index = registry_count_;
  for (std::uint32_t i = 0; i < registry_count_; ++i) {
    if (registry_[i].sequence_generation == sequence_generation) {
      index = i;
      break;
    }
  }
  if (index == registry_count_)
    return Status::FailedPrecondition("controller drain has no KV binding");
  if (!reclaimed_[index]) {
    auto status = recycler.recycle(
        pool, registry_[index].block_table->reserved_handles(),
        last_use_events_[index]);
    if (!status.ok()) {
      state_ = ControllerPackedDriverState::kPoisoned;
      return status;
    }
    reclaimed_[index] = true;
  }
  auto finalized = runtime_->finalize_draining(sequence_generation);
  if (!finalized.ok()) return finalized;
  const auto last = registry_count_ - 1;
  if (index != last) {
    registry_[index] = registry_[last];
    last_use_events_[index] = last_use_events_[last];
    reclaimed_[index] = reclaimed_[last];
  }
  registry_[last] = {};
  last_use_events_[last] = {};
  reclaimed_[last] = false;
  --registry_count_;
  return Status::Ok();
}

}  // namespace pih
