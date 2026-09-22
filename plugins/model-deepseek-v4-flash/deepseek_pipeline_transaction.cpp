#include "pih/model/deepseek_pipeline_transaction.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pih {
namespace {

Result<std::uint64_t> aligned_256(std::uint64_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - 255U) {
    return Status::ResourceExhausted("DeepSeek pipeline byte overflow");
  }
  return (value + 255U) & ~std::uint64_t{255U};
}

}  // namespace

Result<DeepSeekPipelineCapacity> DeepSeekPipelineCapacity::Create(
    std::uint32_t world_size, std::uint32_t max_prefill_chunk_tokens,
    std::uint32_t max_decode_sequences,
    std::uint32_t max_verify_sequences, bool dspark_enabled) {
  if (world_size < 1 || world_size > 4 || max_prefill_chunk_tokens == 0 ||
      max_decode_sequences == 0 || max_verify_sequences == 0) {
    return Status::InvalidArgument("DeepSeek pipeline capacity bounds are invalid");
  }
  if (dspark_enabled && max_verify_sequences >
                            std::numeric_limits<std::uint32_t>::max() / 5U) {
    return Status::ResourceExhausted("DeepSeek verify token count overflow");
  }
  const auto verify_tokens = dspark_enabled ? max_verify_sequences * 5U : 0U;
  const auto max_pipeline_tokens =
      std::max({max_prefill_chunk_tokens, max_decode_sequences, verify_tokens});
  const auto max_sequences = std::max(max_decode_sequences, max_verify_sequences);
  if (max_sequences > max_pipeline_tokens) {
    return Status::InvalidArgument(
        "DeepSeek sequence control ceiling exceeds pipeline token ceiling");
  }
  const std::uint64_t boundary_raw =
      std::uint64_t{max_pipeline_tokens} * 32768U;
  auto boundary_slot = aligned_256(boundary_raw);
  if (!boundary_slot.ok()) return boundary_slot.status();
  const std::uint64_t control_raw = 80U + std::uint64_t{64U} * max_sequences +
                                    std::uint64_t{4U} * max_pipeline_tokens +
                                        16U;
  auto control = aligned_256(control_raw);
  if (!control.ok()) return control.status();
  const std::uint64_t expert_workspace =
      std::uint64_t{16480U} * max_pipeline_tokens + 1280U;

  DeepSeekPipelineCapacity result;
  result.world_size = world_size;
  result.max_pipeline_tokens = max_pipeline_tokens;
  result.expert_tokens = max_pipeline_tokens;
  result.max_sequences = max_sequences;
  result.max_prefill_chunk_tokens = max_prefill_chunk_tokens;
  result.max_decode_sequences = max_decode_sequences;
  result.max_verify_sequences = max_verify_sequences;
  result.dspark_enabled = dspark_enabled;
  result.boundary_slot_bytes = *boundary_slot;
  result.control_payload_bytes = *control;
  result.expert_workspace_bytes = expert_workspace;
  result.ranks.resize(world_size);
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    if (rank > 0) {
      result.ranks[rank].recv_bytes =
          kBoundaryCredits * result.boundary_slot_bytes;
    }
    if (rank + 1 < world_size) {
      result.ranks[rank].output_hold_bytes =
          kBoundaryCredits * result.boundary_slot_bytes;
    }
  }
  return result;
}

const DeepSeekPipelineRankCapacity& DeepSeekPipelineCapacity::rank(
    std::uint32_t rank) const {
  if (rank >= ranks.size()) throw std::out_of_range("DeepSeek capacity rank");
  return ranks[rank];
}

Status DeepSeekPipelineStagePool::reserve() {
  if (inject_failure_) {
    return Status::ResourceExhausted("injected DeepSeek stage prepare failure");
  }
  if (control_available_ == 0 || moe_lane_available_ == 0 ||
      (has_incoming_ && incoming_available_ == 0) ||
      (has_outgoing_ && outgoing_available_ == 0)) {
    return Status::ResourceExhausted("DeepSeek stage transient resources exhausted");
  }
  --control_available_;
  --moe_lane_available_;
  if (has_incoming_) --incoming_available_;
  if (has_outgoing_) --outgoing_available_;
  return Status::Ok();
}

void DeepSeekPipelineStagePool::restore() noexcept {
  ++control_available_;
  ++moe_lane_available_;
  if (has_incoming_) ++incoming_available_;
  if (has_outgoing_) ++outgoing_available_;
}

Result<DeepSeekPipelineResourceSet> DeepSeekPipelineResourceSet::Create(
    DeepSeekPipelineCapacity capacity) {
  if (capacity.world_size == 0 ||
      capacity.ranks.size() != capacity.world_size ||
      capacity.expert_tokens != capacity.max_pipeline_tokens) {
    return Status::InvalidArgument("DeepSeek pipeline capacity is inconsistent");
  }
  DeepSeekPipelineResourceSet resources;
  resources.capacity_ = std::move(capacity);
  resources.stages_.resize(resources.capacity_.world_size);
  for (std::uint32_t rank = 0; rank < resources.capacity_.world_size; ++rank) {
    auto& stage = resources.stages_[rank];
    stage.has_incoming_ = rank > 0;
    stage.has_outgoing_ = rank + 1 < resources.capacity_.world_size;
    stage.incoming_available_ = stage.has_incoming_ ? 2U : 0U;
    stage.outgoing_available_ = stage.has_outgoing_ ? 2U : 0U;
  }
  return resources;
}

Result<DeepSeekPipelineTransaction> DeepSeekPipelineResourceSet::prepare(
    DeepSeekPipelinePlanDescriptor descriptor) {
  if (transaction_live_) {
    return Status::ResourceExhausted("DeepSeek pipeline transaction already live");
  }
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.sequence_count == 0 ||
      descriptor.token_count > capacity_.max_pipeline_tokens ||
      descriptor.sequence_count > capacity_.max_sequences) {
    return Status::InvalidArgument("DeepSeek pipeline descriptor bounds are invalid");
  }
  switch (descriptor.phase) {
    case DeepSeekPlanPhase::kPrefill:
      if (descriptor.token_count == 0 ||
          descriptor.token_count > capacity_.max_prefill_chunk_tokens ||
          descriptor.sequence_count > descriptor.token_count) {
        return Status::InvalidArgument("DeepSeek prefill plan bounds are invalid");
      }
      break;
    case DeepSeekPlanPhase::kDecode:
      if (descriptor.token_count != descriptor.sequence_count ||
          descriptor.sequence_count > capacity_.max_decode_sequences) {
        return Status::InvalidArgument("DeepSeek decode plan bounds are invalid");
      }
      break;
    case DeepSeekPlanPhase::kVerify:
      if (!capacity_.dspark_enabled || descriptor.token_count == 0 ||
          descriptor.sequence_count > capacity_.max_verify_sequences ||
          descriptor.token_count > descriptor.sequence_count * 5U) {
        return Status::InvalidArgument("DeepSeek verify plan bounds are invalid");
      }
      break;
    case DeepSeekPlanPhase::kDrain:
      if (descriptor.token_count != 0) {
        return Status::InvalidArgument("DeepSeek drain plan must have zero tokens");
      }
      break;
  }
  if (engine_epoch_ != 0 && descriptor.engine_epoch != engine_epoch_) {
    return Status::FailedPrecondition("DeepSeek pipeline engine epoch changed");
  }
  if (descriptor.plan_sequence != last_plan_sequence_ + 1) {
    return Status::FailedPrecondition("DeepSeek pipeline plan sequence is not strict");
  }
  std::uint32_t reserved = 0;
  for (auto& stage : stages_) {
    const auto status = stage.reserve();
    if (!status.ok()) {
      for (std::uint32_t i = 0; i < reserved; ++i) stages_[i].restore();
      return status;
    }
    ++reserved;
  }
  engine_epoch_ = descriptor.engine_epoch;
  last_plan_sequence_ = descriptor.plan_sequence;
  transaction_live_ = true;
  return DeepSeekPipelineTransaction(*this, descriptor);
}

DeepSeekPipelineStagePool& DeepSeekPipelineResourceSet::rank(std::uint32_t rank) {
  if (rank >= stages_.size()) throw std::out_of_range("DeepSeek resource rank");
  return stages_[rank];
}

void DeepSeekPipelineResourceSet::restore_all() noexcept {
  if (!transaction_live_) return;
  for (auto& stage : stages_) stage.restore();
  transaction_live_ = false;
}

DeepSeekPipelineTransaction::DeepSeekPipelineTransaction(
    DeepSeekPipelineResourceSet& owner,
    DeepSeekPipelinePlanDescriptor descriptor) noexcept
    : owner_(&owner), descriptor_(descriptor) {}

DeepSeekPipelineTransaction::DeepSeekPipelineTransaction(
    DeepSeekPipelineTransaction&& other) noexcept {
  *this = std::move(other);
}

DeepSeekPipelineTransaction& DeepSeekPipelineTransaction::operator=(
    DeepSeekPipelineTransaction&& other) noexcept {
  if (this != &other) {
    rollback_if_prepared();
    owner_ = std::exchange(other.owner_, nullptr);
    descriptor_ = other.descriptor_;
    state_ = other.state_;
  }
  return *this;
}

DeepSeekPipelineTransaction::~DeepSeekPipelineTransaction() {
  rollback_if_prepared();
}

void DeepSeekPipelineTransaction::rollback_if_prepared() noexcept {
  if (owner_ != nullptr && state_ == DeepSeekPipelineTransactionState::kPrepared) {
    owner_->restore_all();
    state_ = DeepSeekPipelineTransactionState::kAborted;
  }
}

Status DeepSeekPipelineTransaction::validate_commit() const {
  if (owner_ == nullptr || state_ != DeepSeekPipelineTransactionState::kPrepared) {
    return Status::FailedPrecondition("DeepSeek plan is not prepared");
  }
  return Status::Ok();
}

Status DeepSeekPipelineTransaction::commit() {
  const auto status = validate_commit();
  if (!status.ok()) return status;
  state_ = DeepSeekPipelineTransactionState::kCommitted;
  return Status::Ok();
}

Status DeepSeekPipelineTransaction::validate_abort_prepare() const {
  if (owner_ == nullptr || state_ != DeepSeekPipelineTransactionState::kPrepared) {
    return Status::FailedPrecondition("DeepSeek prepared plan cannot be aborted");
  }
  return Status::Ok();
}

Status DeepSeekPipelineTransaction::abort_prepare() {
  const auto status = validate_abort_prepare();
  if (!status.ok()) return status;
  owner_->restore_all();
  state_ = DeepSeekPipelineTransactionState::kAborted;
  return Status::Ok();
}

Status DeepSeekPipelineTransaction::validate_complete() const {
  if (owner_ == nullptr || state_ != DeepSeekPipelineTransactionState::kCommitted) {
    return Status::FailedPrecondition("DeepSeek plan is not committed");
  }
  return Status::Ok();
}

Status DeepSeekPipelineTransaction::complete() {
  const auto status = validate_complete();
  if (!status.ok()) return status;
  owner_->restore_all();
  state_ = DeepSeekPipelineTransactionState::kCompleted;
  return Status::Ok();
}

}  // namespace pih
