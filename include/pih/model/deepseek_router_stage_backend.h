#pragma once

#include <optional>
#include <span>

#include "pih/model/deepseek_hash_router.h"
#include "pih/model/deepseek_hash_router_coordinator.h"
#include "pih/model/deepseek_learned_router_coordinator.h"

namespace pih {

struct DeepSeekHashRouterStageWork final {
  std::span<const std::uint32_t> token_ids;
  std::span<const float> raw_scores;
  std::uint32_t vocabulary_size = 0;
  std::span<const std::uint16_t> token_to_experts;
  DeepSeekRouteScratchArena* scratch = nullptr;
  DeepSeekExpertPlanStore* store = nullptr;
  DeepSeekHashRouterCoordinator* coordinator = nullptr;
  DeepSeekHashRouterSubmission submission;
};

struct DeepSeekLearnedRouterStageWork final {
  DeepSeekLearnedRouterCoordinator* coordinator = nullptr;
  DeepSeekLearnedRouterSubmission submission;
};

class DeepSeekRouterStageWorkProvider {
 public:
  virtual ~DeepSeekRouterStageWorkProvider() = default;
  // Router publication and routed execution must refer to this exact object.
  virtual DeepSeekExpertPlanProvider* plan_provider() noexcept = 0;
  virtual Result<DeepSeekHashRouterStageWork> resolve_hash(
      const DeepSeekStageOperatorCommand& command,
      const DeepSeekPipelinePlanDescriptor& plan) = 0;
  virtual Result<DeepSeekLearnedRouterStageWork> resolve_learned(
      const DeepSeekStageOperatorCommand& command,
      const DeepSeekPipelinePlanDescriptor& plan) = 0;
};

// Publishes the canonical expert plan before allowing the routed backend to
// begin paging or expert execution. An error is fail-stop for this instance.
class DeepSeekRouterStageOperatorBackend final
    : public DeepSeekStageOperatorBackend {
 public:
  static Result<DeepSeekRouterStageOperatorBackend> Create(
      DeepSeekStageOperatorBackend& routed_backend,
      DeepSeekRouterStageWorkProvider& work_provider);

  Status launch(const DeepSeekStageOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor& plan) override;
  Result<DeepSeekStageComputeStatus> poll() override;

 private:
  enum class Mode : std::uint8_t {
    kIdle,
    kInner,
    kHashRouter,
    kLearnedRouter,
    kPoisoned
  };
  Status poison(Status status);

  DeepSeekStageOperatorBackend* routed_backend_ = nullptr;
  DeepSeekRouterStageWorkProvider* work_provider_ = nullptr;
  DeepSeekLearnedRouterCoordinator* learned_coordinator_ = nullptr;
  DeepSeekHashRouterCoordinator* hash_coordinator_ = nullptr;
  std::optional<DeepSeekStageOperatorCommand> pending_command_;
  std::optional<DeepSeekPipelinePlanDescriptor> pending_plan_;
  Mode mode_ = Mode::kIdle;
};

}  // namespace pih
