#pragma once

#include "pih/model/deepseek_attention_sequence_transaction.h"
#include "pih/model/deepseek_dspark_gpu_state_digest.h"
#include "pih/model/deepseek_dspark_prefix_commitment.h"
#include "pih/model/deepseek_dspark_state_cut_ack.h"
#include "pih/model/deepseek_pipeline_stage_executor.h"

namespace pih {

class DeepSeekDsparkRankCutReceipt final {
 public:
  static Result<DeepSeekDsparkRankCutReceipt> Create(
      bool terminal_drain,
      const DeepSeekDsparkGpuStateDigestPoll& gpu_state_digest,
      DeepSeekDsparkPrefixProof prefix_proof);
  [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::uint64_t plan_sequence() const noexcept {
    return plan_sequence_;
  }
  [[nodiscard]] std::uint64_t old_generation() const noexcept {
    return old_generation_;
  }
  [[nodiscard]] std::uint64_t new_generation() const noexcept {
    return new_generation_;
  }
  [[nodiscard]] std::uint32_t retained_record_count() const noexcept {
    return retained_record_count_;
  }
  [[nodiscard]] bool terminal_drain() const noexcept {
    return terminal_drain_;
  }
  [[nodiscard]] const DeepSeekDsparkPrefixProof& prefix_proof() const noexcept {
    return prefix_proof_;
  }

 private:
  std::uint32_t rank_ = 0;
  std::uint64_t plan_sequence_ = 0;
  std::uint64_t old_generation_ = 0;
  std::uint64_t new_generation_ = 0;
  std::uint32_t retained_record_count_ = 0;
  bool terminal_drain_ = false;
  DeepSeekDsparkPrefixProof prefix_proof_;
};

class DeepSeekDsparkTerminalStateDrainer {
 public:
  virtual ~DeepSeekDsparkTerminalStateDrainer() = default;
  virtual Status drain_committed_sequence_state(
      const DeepSeekDsparkStateCutDecision& decision) = 0;
};

class DeepSeekDsparkRankStateCutPublisher final {
 public:
  static Result<DeepSeekDsparkRankStateCutPublisher> Create(
      std::uint32_t rank,
      DeepSeekAttentionSequenceTransaction& transaction,
      DeepSeekPipelineStageExecutor& stage_executor,
      const DeepSeekDsparkRankCutReceipt& receipt,
      DeepSeekDsparkTerminalStateDrainer* terminal_drainer = nullptr);
  Result<DeepSeekDsparkStateCutAck> publish(
      const DeepSeekDsparkStateCutDecision& decision);
  [[nodiscard]] bool published() const noexcept { return published_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }

 private:
  std::uint32_t rank_ = 0;
  DeepSeekAttentionSequenceTransaction* transaction_ = nullptr;
  DeepSeekPipelineStageExecutor* stage_executor_ = nullptr;
  DeepSeekDsparkRankCutReceipt receipt_;
  DeepSeekDsparkTerminalStateDrainer* terminal_drainer_ = nullptr;
  bool published_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
