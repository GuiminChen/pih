#pragma once

#include <cstdint>
#include <vector>

#include "pih/model/deepseek_pipeline_stage_executor.h"

namespace pih {

enum class DeepSeekStageOperatorKind : std::uint8_t {
  kEmbedding,
  kAttention,
  kMoe,
  kHead,
  kDspark,
};

struct DeepSeekStageOperatorCommand final {
  DeepSeekStageOperatorKind kind = DeepSeekStageOperatorKind::kAttention;
  std::uint32_t layer = 0;
  bool operator==(const DeepSeekStageOperatorCommand&) const = default;
};

class DeepSeekStageOperatorBackend {
 public:
  virtual ~DeepSeekStageOperatorBackend() = default;
  virtual Status launch(const DeepSeekStageOperatorCommand& command,
                        const DeepSeekPipelinePlanDescriptor& plan) = 0;
  virtual Result<DeepSeekStageComputeStatus> poll() = 0;
};

class DeepSeekModelStageComputeDriver final : public DeepSeekStageComputeDriver {
 public:
  explicit DeepSeekModelStageComputeDriver(DeepSeekStageOperatorBackend& backend)
      : backend_(&backend) {}

  Status launch(const DeepSeekPipelinePlanDescriptor& plan,
                const DeepSeekStagePlan& stage) override;
  Result<DeepSeekStageComputeStatus> poll() override;
  [[nodiscard]] const std::vector<DeepSeekStageOperatorCommand>& commands()
      const noexcept {
    return commands_;
  }
  [[nodiscard]] std::size_t next_command_index() const noexcept {
    return next_command_;
  }

 private:
  Status dispatch_next();
  DeepSeekStageOperatorBackend* backend_ = nullptr;
  DeepSeekPipelinePlanDescriptor plan_;
  std::vector<DeepSeekStageOperatorCommand> commands_;
  std::size_t next_command_ = 0;
  bool launched_ = false;
  bool command_inflight_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
