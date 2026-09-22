#include "supervisor_request.h"
#include <new>

namespace pih::deepseek_v41 {
Result<std::unique_ptr<SupervisorRequest>> SupervisorRequest::CreateWithTokenizer(
    std::shared_ptr<const V41Tokenizer> tokenizer, std::string_view rendered_prompt,
    SamplingIdentity identity, SamplingParameters sampling, TokenStopConfig stopping,
    std::uint32_t maximum_positions, SupervisorRequestBudgets budgets) {
  if (!tokenizer) return Status::InvalidArgument("Supervisor authenticated tokenizer is absent");
  if (!maximum_positions || maximum_positions > 1048576 || !stopping.maximum ||
      stopping.maximum >= maximum_positions)
    return Status::InvalidArgument("Supervisor request context reservation invalid");
  const auto parameters = ValidateSamplingParameters(sampling);
  if (!parameters.ok()) return parameters;
  try {
    auto prompt = tokenizer->EncodeRendered(rendered_prompt);
    if (!prompt.ok()) return prompt.status();
    if (prompt->size() > maximum_positions - stopping.maximum)
      return Status::InvalidArgument("Prompt and completion exceed reserved worker context");
    auto output = TokenOutputQueue::Create(budgets.output_slots, budgets.publication_bytes);
    if (!output.ok()) return output.status();
    auto ledger = TokenLedger::Create(identity, static_cast<std::uint32_t>(prompt->size()),
        sampling, std::move(stopping), budgets.record_bytes, **output);
    if (!ledger.ok()) return ledger.status();
    auto owner = std::unique_ptr<SupervisorRequest>(new SupervisorRequest);
    owner->tokenizer_ = std::move(tokenizer);
    owner->prompt_ = std::move(*prompt);
    owner->output_ = std::move(*output);
    owner->ledger_ = std::move(*ledger);
    owner->maximum_positions_ = maximum_positions;
    return owner;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("Supervisor request admission allocation failed");
  }
}
Result<WorkerBootstrap> SupervisorRequest::BindBootstrap(const WorkerBootstrap& placement) const {
  if (ledger_->failed() || !ledger_->records().empty() || ledger_->pending_output())
    return Status::FailedPrecondition("Cannot bind an already used supervisor request");
  try {
    auto bootstrap = placement;
    bootstrap.identity = ledger_->identity();
    bootstrap.sampling = ledger_->initial_sampling();
    bootstrap.prompt_tokens = static_cast<std::uint32_t>(prompt_.size());
    bootstrap.maximum_positions = maximum_positions_;
    const auto valid = bootstrap.Validate();
    if (!valid.ok()) return valid;
    return bootstrap;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("Supervisor bootstrap allocation failed");
  }
}
}  // namespace pih::deepseek_v41
