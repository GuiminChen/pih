#include "supervisor_deployment.h"
#include <algorithm>

namespace pih::deepseek_v41 {
Status ValidateSupervisorRequestPolicy(const SupervisorConfig& config,
    std::string_view prompt, const SamplingParameters& sampling,
    const TokenStopConfig& stopping) {
  if (prompt.empty() || prompt.size() > (1U << 20) || !config.identity.epoch ||
      !config.identity.sequence_generation || !config.identity.sampling_config_id ||
      config.identity.epoch > INT64_MAX || config.identity.sequence_generation > INT64_MAX ||
      config.identity.sampling_config_id > INT64_MAX || !config.first_plan || config.first_plan > INT64_MAX ||
      config.identity.plan_seq || config.maximum_positions < 2 ||
      config.maximum_positions > 1048576 || !stopping.maximum ||
      stopping.maximum > config.stopping.maximum || stopping.maximum >= config.maximum_positions ||
      stopping.maximum > config.budgets.record_bytes / sizeof(AcceptedToken) ||
      sampling.ordinal || sampling.suppressed_count)
    return Status::InvalidArgument("Supervisor request exceeds sealed admission policy");
  if (!config.budgets.vocabulary_bytes || config.budgets.vocabulary_bytes > (64U << 20) ||
      !config.budgets.output_slots || config.budgets.output_slots > 4096 ||
      config.budgets.output_slots > config.budgets.publication_bytes / sizeof(TokenPublication) ||
      config.budgets.publication_bytes > INT64_MAX || config.budgets.record_bytes > INT64_MAX)
    return Status::InvalidArgument("Supervisor output or vocabulary reservation is insufficient");
  if (std::any_of(std::begin(sampling.suppressed), std::end(sampling.suppressed),
                  [](auto token) { return token != 0; }))
    return Status::InvalidArgument("Supervisor initial suppression storage must be zero");
  // Model stop-token semantics belong to the deployment. Per-request stop
  // strings are allowed, but may not remove or replace those token identities.
  if (stopping.token_count != config.stopping.token_count ||
      !std::equal(std::begin(stopping.tokens), std::end(stopping.tokens), std::begin(config.stopping.tokens)))
    return Status::InvalidArgument("Supervisor request cannot change model stop tokens");
  const auto parameters = ValidateSamplingParameters(sampling);
  if (!parameters.ok()) return parameters;
  auto stop = TokenStopState::Create(stopping);
  return stop.ok() ? Status::Ok() : stop.status();
}
}  // namespace pih::deepseek_v41
