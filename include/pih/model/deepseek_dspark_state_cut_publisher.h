#pragma once

#include "pih/model/deepseek_dspark_state_cut_ack.h"

namespace pih {

class DeepSeekDsparkStateCutPublishOperations {
 public:
  virtual ~DeepSeekDsparkStateCutPublishOperations() = default;
  virtual Status validate_engine_healthy() = 0;
  virtual Status publish_ledger_and_output(
      const DeepSeekDsparkStateCutDecision& decision) = 0;
};

class DeepSeekDsparkStateCutPublisher final {
 public:
  static Result<DeepSeekDsparkStateCutPublisher> Create(
      DeepSeekDsparkStateCutPublishOperations& operations);
  Status publish(
      const DeepSeekDsparkStateCutDecision& decision,
      std::uint32_t world_size,
      std::span<const DeepSeekDsparkStateCutAck> acknowledgements,
      bool engine_poisoned);
  [[nodiscard]] bool published() const noexcept { return published_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }

 private:
  Status run(Status status);
  DeepSeekDsparkStateCutPublishOperations* operations_ = nullptr;
  bool published_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
