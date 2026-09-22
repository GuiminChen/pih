#pragma once

#include "pih/model/deepseek_dspark_state_cut_publisher.h"
#include "pih/scheduler/controller_sequence.h"

namespace pih {

class DeepSeekDsparkControllerPublishOperations final
    : public DeepSeekDsparkStateCutPublishOperations {
 public:
  static Result<DeepSeekDsparkControllerPublishOperations> Create(
      ControllerSequence& sequence, std::uint64_t sequence_generation,
      PackedTokenPhase next_decode_phase);
  Status validate_engine_healthy() override;
  Status publish_ledger_and_output(
      const DeepSeekDsparkStateCutDecision& decision) override;

 private:
  Status validate_decision(
      const DeepSeekDsparkStateCutDecision& decision) const;

  ControllerSequence* sequence_ = nullptr;
  std::uint64_t sequence_generation_ = 0;
  PackedTokenPhase next_decode_phase_ = PackedTokenPhase::kDecode;
  bool ledger_published_ = false;
};

}  // namespace pih
