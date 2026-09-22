#pragma once

#include "pih/model/deepseek_dspark_stage_backend.h"

namespace pih {

enum class DeepSeekDsparkMtpOperatorKind : std::uint8_t {
  kAttention,
  kMoe,
};

struct DeepSeekDsparkMtpOperatorCommand final {
  DeepSeekDsparkStageId stage = DeepSeekDsparkStageId::kMtp0;
  DeepSeekDsparkMtpOperatorKind kind =
      DeepSeekDsparkMtpOperatorKind::kAttention;
  bool operator==(const DeepSeekDsparkMtpOperatorCommand&) const = default;
};

class DeepSeekDsparkMtpOperatorBackend {
 public:
  virtual ~DeepSeekDsparkMtpOperatorBackend() = default;
  virtual Status launch(const DeepSeekDsparkMtpOperatorCommand& command,
                        const DeepSeekPipelinePlanDescriptor& plan) = 0;
  virtual Result<DeepSeekStageComputeStatus> poll() = 0;
  virtual Status cancel() = 0;
};

class DeepSeekDsparkMtpBlockExecutor final
    : public DeepSeekDsparkBlockExecutor {
 public:
  static Result<DeepSeekDsparkMtpBlockExecutor> Create(
      DeepSeekDsparkMtpOperatorBackend& backend);
  Status launch(DeepSeekDsparkStageId stage_id,
                const DeepSeekPipelinePlanDescriptor& plan) override;
  Result<DeepSeekStageComputeStatus> poll() override;
  Status cancel() override;

 private:
  enum class State : std::uint8_t {
    kIdle,
    kAttention,
    kMoe,
    kPoisoned,
  };
  Status poison(Status status);
  DeepSeekDsparkMtpOperatorBackend* backend_ = nullptr;
  DeepSeekPipelinePlanDescriptor plan_;
  DeepSeekDsparkStageId stage_id_ = DeepSeekDsparkStageId::kMtp0;
  State state_ = State::kIdle;
};

}  // namespace pih
