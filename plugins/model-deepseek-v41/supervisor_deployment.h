#pragma once
#include "supervisor_config.h"

namespace pih::deepseek_v41 {
// Immutable authenticated deployment plus serialized request identity allocation.
// No request API accepts paths, executable identities, devices, budgets or timeouts.
// This is admission only: the caller owns process custody and retirement.
class SupervisorDeployment final {
 public:
  static Result<std::unique_ptr<SupervisorDeployment>> Create(SupervisorConfig config);
  SupervisorDeployment(const SupervisorDeployment&) = delete;
  SupervisorDeployment& operator=(const SupervisorDeployment&) = delete;
  Result<std::unique_ptr<SupervisorRequest>> Admit(std::string_view rendered_prompt,
      SamplingParameters sampling, TokenStopConfig stopping);
  const SupervisorConfig& configuration() const noexcept { return config_; }
 private:
  SupervisorDeployment(SupervisorConfig config, std::shared_ptr<const V41Tokenizer> tokenizer)
      : config_(std::move(config)), tokenizer_(std::move(tokenizer)),
        next_generation_(config_.identity.sequence_generation),
        next_sampling_id_(config_.identity.sampling_config_id) {}
  const SupervisorConfig config_;
  std::shared_ptr<const V41Tokenizer> tokenizer_;
  std::uint64_t next_generation_, next_sampling_id_;
};
// Also used before loading tokenizer files; never changes sealed defaults.
Status ValidateSupervisorRequestPolicy(const SupervisorConfig& deployment,
    std::string_view rendered_prompt, const SamplingParameters& sampling,
    const TokenStopConfig& stopping);
}  // namespace pih::deepseek_v41
