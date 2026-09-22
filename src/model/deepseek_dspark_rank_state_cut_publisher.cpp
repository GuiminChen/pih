#include "pih/model/deepseek_dspark_rank_state_cut_publisher.h"

namespace pih {

Result<DeepSeekDsparkRankCutReceipt> DeepSeekDsparkRankCutReceipt::Create(
    bool terminal_drain,
    const DeepSeekDsparkGpuStateDigestPoll& gpu_state_digest,
    DeepSeekDsparkPrefixProof prefix_proof) {
  if (gpu_state_digest.status() != DeepSeekExpertAsyncStatus::kSuccess ||
      gpu_state_digest.plan_sequence() == 0 ||
      gpu_state_digest.old_generation() == 0 ||
      gpu_state_digest.new_generation() !=
          gpu_state_digest.old_generation() + 1 ||
      prefix_proof.rank != gpu_state_digest.rank() ||
      prefix_proof.local_state_hash != gpu_state_digest.local_state_hash()) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark rank cut receipt authority is invalid");
  }
  DeepSeekDsparkRankCutReceipt value;
  value.rank_ = gpu_state_digest.rank();
  value.plan_sequence_ = gpu_state_digest.plan_sequence();
  value.old_generation_ = gpu_state_digest.old_generation();
  value.new_generation_ = gpu_state_digest.new_generation();
  value.retained_record_count_ = gpu_state_digest.retained_record_count();
  value.terminal_drain_ = terminal_drain;
  value.prefix_proof_ = std::move(prefix_proof);
  return value;
}

Result<DeepSeekDsparkRankStateCutPublisher>
DeepSeekDsparkRankStateCutPublisher::Create(
    std::uint32_t rank,
    DeepSeekAttentionSequenceTransaction& transaction,
    DeepSeekPipelineStageExecutor& stage_executor,
    const DeepSeekDsparkRankCutReceipt& receipt,
    DeepSeekDsparkTerminalStateDrainer* terminal_drainer) {
  if (rank >= 4 ||
      transaction.state() !=
          DeepSeekAttentionSequenceTransactionState::kReadyToResolve ||
      stage_executor.state() !=
          DeepSeekPipelineStageExecutorState::kComplete ||
      stage_executor.descriptor().plan_sequence != receipt.plan_sequence() ||
      stage_executor.descriptor().phase != DeepSeekPlanPhase::kVerify ||
      stage_executor.stage().rank != rank ||
      !stage_executor.stage().owns_dspark ||
      receipt.plan_sequence() == 0 || receipt.old_generation() == 0 ||
      receipt.new_generation() != receipt.old_generation() + 1 ||
      receipt.rank() != rank ||
      (receipt.terminal_drain() && terminal_drainer == nullptr)) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark rank cut is not publishable");
  }
  DeepSeekDsparkRankStateCutPublisher value;
  value.rank_ = rank;
  value.transaction_ = &transaction;
  value.stage_executor_ = &stage_executor;
  value.receipt_ = receipt;
  value.terminal_drainer_ = terminal_drainer;
  return value;
}

Result<DeepSeekDsparkStateCutAck>
DeepSeekDsparkRankStateCutPublisher::publish(
    const DeepSeekDsparkStateCutDecision& decision) {
  if (published_ || poisoned_ ||
      transaction_->state() !=
          DeepSeekAttentionSequenceTransactionState::kReadyToResolve ||
      stage_executor_->state() !=
          DeepSeekPipelineStageExecutorState::kComplete ||
      stage_executor_->descriptor().plan_sequence != decision.plan_sequence ||
      decision.plan_sequence != receipt_.plan_sequence() ||
      decision.old_generation != receipt_.old_generation() ||
      decision.new_generation != receipt_.new_generation() ||
      decision.retained_record_count != receipt_.retained_record_count() ||
      decision.terminal_drain != receipt_.terminal_drain()) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark rank cut decision mismatched");
  }
  auto proof = verify_deepseek_dspark_prefix_proof(
      decision.plan_sequence, decision.old_generation,
      decision.new_generation, receipt_.prefix_proof(),
      decision.per_rank_prefix_hash_merkle_root);
  if (!proof.ok()) return proof;
  Status status = Status::Ok();
  if (decision.terminal_drain) {
    status = transaction_->abort();
    if (status.ok()) {
      status = terminal_drainer_->drain_committed_sequence_state(decision);
    }
  } else {
    status = transaction_->validate_commit();
    if (status.ok()) status = transaction_->commit();
  }
  if (!status.ok()) {
    poisoned_ = true;
    return status;
  }
  published_ = true;
  return DeepSeekDsparkStateCutAck{
      rank_, decision.plan_sequence, decision.old_generation,
      decision.new_generation, decision.retained_record_count,
      decision.processed_delta, decision.terminal_drain, true,
      true, true, decision.per_rank_prefix_hash_merkle_root};
}

}  // namespace pih
