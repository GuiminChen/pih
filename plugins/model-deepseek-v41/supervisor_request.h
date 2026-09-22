#pragma once
#include "tokenizer.h"
#include "token_ledger.h"
#include "worker_bootstrap.h"

namespace pih::deepseek_v41 {
struct SupervisorRequestBudgets final {
  std::uint64_t vocabulary_bytes = 0, publication_bytes = 0, record_bytes = 0;
  std::uint32_t output_slots = 0;
};
// CPU-only admission owner. Must outlive the session and all publication leases.
// No launch, GPU initialization, chat rendering, or implicit process cleanup.
class SupervisorRequest final {
 public:
  SupervisorRequest(const SupervisorRequest&) = delete;
  SupervisorRequest& operator=(const SupervisorRequest&) = delete;
  // Copy placement/artifact/deadline fields; replace only sequence-owned fields.
  // Full bootstrap validation is required before returning it to the launcher.
  Result<WorkerBootstrap> BindBootstrap(const WorkerBootstrap& placement) const;
  std::span<const std::uint32_t> prompt() const noexcept { return prompt_; }
  std::span<const std::string_view> token_bytes() const noexcept { return tokenizer_->token_bytes(); }
  TokenLedger& ledger() noexcept { return *ledger_; }
  TokenOutputQueue& output() noexcept { return *output_; }
 private:
  friend class SupervisorDeployment;
  // Only the immutable deployment may allocate identities and bind its tokenizer.
  static Result<std::unique_ptr<SupervisorRequest>> CreateWithTokenizer(
      std::shared_ptr<const V41Tokenizer> tokenizer, std::string_view rendered_prompt,
      SamplingIdentity identity, SamplingParameters sampling, TokenStopConfig stopping,
      std::uint32_t maximum_positions, SupervisorRequestBudgets budgets);
  SupervisorRequest() = default;
  // Reverse destruction keeps borrowers shorter-lived than their owners.
  std::shared_ptr<const V41Tokenizer> tokenizer_;
  std::vector<std::uint32_t> prompt_;
  std::unique_ptr<TokenOutputQueue> output_;
  std::unique_ptr<TokenLedger> ledger_;
  std::uint32_t maximum_positions_ = 0;
};
}  // namespace pih::deepseek_v41
