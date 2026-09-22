#include "supervisor_deployment.h"
#include <new>

namespace pih::deepseek_v41 {
Result<std::unique_ptr<SupervisorDeployment>> SupervisorDeployment::Create(SupervisorConfig config) {
  const auto policy = ValidateSupervisorRequestPolicy(config, config.rendered_prompt, config.sampling, config.stopping);
  if (!policy.ok()) return policy;
  auto tokenizer = V41Tokenizer::Open(config.tokenizer_directory, config.budgets.vocabulary_bytes);
  if (!tokenizer.ok()) return tokenizer.status();
  try {
    return std::unique_ptr<SupervisorDeployment>(new SupervisorDeployment(
        std::move(config), std::shared_ptr<const V41Tokenizer>(std::move(*tokenizer))));
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("Supervisor deployment admission allocation failed");
  }
}
Result<std::unique_ptr<SupervisorRequest>> SupervisorDeployment::Admit(
    std::string_view prompt, SamplingParameters sampling, TokenStopConfig stopping) {
  const auto policy = ValidateSupervisorRequestPolicy(config_, prompt, sampling, stopping);
  if (!policy.ok()) return policy;
  if (next_generation_ > INT64_MAX || next_sampling_id_ > INT64_MAX)
    return Status::ResourceExhausted("Supervisor request identity space exhausted");
  auto identity = config_.identity;
  identity.sequence_generation = next_generation_;
  identity.sampling_config_id = next_sampling_id_;
  auto request = SupervisorRequest::CreateWithTokenizer(tokenizer_, prompt, identity,
      sampling, std::move(stopping), config_.maximum_positions, config_.budgets);
  if (!request.ok()) return request.status();
  ++next_generation_; ++next_sampling_id_;
  return request;
}
}  // namespace pih::deepseek_v41
