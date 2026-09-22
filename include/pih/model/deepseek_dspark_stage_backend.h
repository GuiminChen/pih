#pragma once

#include <array>
#include <span>

#include "pih/model/deepseek_dspark_embed_coordinator.h"
#include "pih/model/deepseek_dspark_head_executor.h"
#include "pih/model/deepseek_dspark_stage_identity.h"
#include "pih/model/deepseek_stage_compute_driver.h"

namespace pih {
class DeepSeekDsparkBlockExecutor {
 public:
  virtual ~DeepSeekDsparkBlockExecutor() = default;
  virtual Status launch(DeepSeekDsparkStageId stage_id,
                        const DeepSeekPipelinePlanDescriptor& plan) = 0;
  virtual Result<DeepSeekStageComputeStatus> poll() = 0;
  virtual Status cancel() = 0;
};

enum class DeepSeekDsparkStageEvidenceState : std::uint8_t {
  kNotStarted,
  kInFlight,
  kSucceeded,
  kFailed,
  kCancelled,
};

struct DeepSeekDsparkStageExecutionEvidence final {
  DeepSeekDsparkStageId stage = DeepSeekDsparkStageId::kMtp0;
  DeepSeekDsparkStageEvidenceState state =
      DeepSeekDsparkStageEvidenceState::kNotStarted;
  std::uint64_t engine_epoch = 0;
  std::uint64_t plan_sequence = 0;
};
enum class DeepSeekDsparkStageWorkKind : std::uint8_t {
  kPrefillStateInitialization,
  kDecodeProposal,
};
struct DeepSeekDsparkStageWork final {
  DeepSeekDsparkStageWorkKind kind =
      DeepSeekDsparkStageWorkKind::kDecodeProposal;
  DeepSeekDsparkEmbedCoordinator* embed_coordinator = nullptr;
  DeepSeekDsparkHeadExecutor* head_executor = nullptr;
  DeepSeekAttentionSequenceTransaction* transaction = nullptr;
  DeepSeekDsparkEmbedSubmission embed;
  DeepSeekDsparkHeadSubmission head;
};
class DeepSeekDsparkStageWorkProvider {
 public:
  virtual ~DeepSeekDsparkStageWorkProvider() = default;
  virtual Result<const DeepSeekDsparkStageWork*> resolve(
      const DeepSeekPipelinePlanDescriptor& plan) = 0;
};
class DeepSeekDsparkStageOperatorBackend final
    : public DeepSeekStageOperatorBackend {
 public:
  static Result<DeepSeekDsparkStageOperatorBackend> Create(
      DeepSeekStageOperatorBackend& fallback,
      DeepSeekDsparkStageWorkProvider& provider,
      DeepSeekDsparkBlockExecutor& block_executor);
  Status launch(const DeepSeekStageOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor& plan) override;
  Result<DeepSeekStageComputeStatus> poll() override;
  Status cancel();
  [[nodiscard]] std::span<const DeepSeekDsparkStageExecutionEvidence>
  evidence() const noexcept {
    return evidence_;
  }
  // Submission is ordered after stage 2. Device completion remains owned by
  // the surrounding attention transaction and its external error channel.
  [[nodiscard]] bool head_submitted() const noexcept {
    return head_submitted_;
  }
  [[nodiscard]] bool prefill_state_initialized() const noexcept {
    return prefill_state_initialized_;
  }
 private:
  enum class Mode : std::uint8_t { kIdle, kFallback, kBlocks, kPoisoned };
  Status poison(Status status);
  Status rollback_work() noexcept;
  Status launch_stage(std::uint32_t index);
  DeepSeekStageOperatorBackend* fallback_ = nullptr;
  DeepSeekDsparkStageWorkProvider* provider_ = nullptr;
  DeepSeekDsparkBlockExecutor* block_executor_ = nullptr;
  const DeepSeekDsparkStageWork* work_ = nullptr;
  DeepSeekPipelinePlanDescriptor plan_;
  std::uint32_t next_stage_ = 0;
  std::array<DeepSeekDsparkStageExecutionEvidence,
             kDeepSeekDsparkStageCount> evidence_{};
  bool head_submitted_ = false;
  bool prefill_state_initialized_ = false;
  Mode mode_ = Mode::kIdle;
};
}  // namespace pih
