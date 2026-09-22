#pragma once
#include "token_output.h"
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace pih::deepseek_v41 {
struct TokenSamplingRequest final {
  SamplingIdentity identity;
  SamplingParameters parameters;
  std::uint32_t processed_length = 0;
};
// Serialized local request ledger. No GPU/transport or distributed authority.
// The controller must reserve output credit and close all rank boundaries before
// CommitLocal; bytes/results become externally visible only after that commit.
class TokenLedger final {
 public:
  static Result<std::unique_ptr<TokenLedger>> Create(SamplingIdentity identity, std::uint32_t prompt_tokens,
      SamplingParameters sampling, TokenStopConfig stopping, std::uint64_t record_budget_bytes, TokenOutputQueue& output);
  TokenLedger(const TokenLedger&) = delete;
  TokenLedger& operator=(const TokenLedger&) = delete;
  Result<TokenSamplingRequest> Prepare(std::uint64_t plan_seq);
  Status MarkInFlight(std::uint64_t plan_seq);
  Status Stage(const SamplingObservation& observation, std::string_view token_bytes);
  Result<TokenOutputLease> CommitLocal(std::uint64_t plan_seq);
  Status Abort(std::uint64_t plan_seq);
  Status AbortRetired(std::uint64_t plan_seq);
  std::optional<TokenOutputLease> pending_output() const noexcept { return output_lease_; }
  void Fail() noexcept { failed_ = true; }
  bool failed() const noexcept { return failed_; }
  std::span<const AcceptedToken> records() const noexcept { return {records_.data(), count_}; }
  std::uint64_t ordinal() const noexcept { return ordinal_; }
  std::uint32_t model_processed_length() const noexcept { return processed_; }
  std::uint32_t prompt_tokens() const noexcept { return prompt_; }
  std::uint32_t maximum_completion_tokens() const noexcept { return static_cast<std::uint32_t>(records_.size()); }
  const SamplingIdentity& identity() const noexcept { return identity_; }
  const SamplingParameters& initial_sampling() const noexcept { return sampling_; }
  std::optional<std::uint32_t> pending_input() const noexcept { return pending_input_; }
  TokenFinish finish() const noexcept { return stopping_.finish(); }
 private:
  explicit TokenLedger(TokenStopState&& stopping) : stopping_(std::move(stopping)) {}
  TokenStopState stopping_;
  SamplingIdentity identity_{};
  SamplingParameters sampling_{};
  std::vector<AcceptedToken> records_;
  std::optional<TokenSamplingRequest> prepared_;
  std::optional<TokenStopTransition> staged_;
  std::optional<std::uint32_t> pending_input_;
  std::uint64_t ordinal_ = 0, last_plan_ = 0;
  std::uint32_t prompt_ = 0, processed_ = 0, count_ = 0;
  bool failed_ = false;
  TokenOutputQueue* output_ = nullptr;
  std::optional<TokenOutputLease> output_lease_;
};
}  // namespace pih::deepseek_v41
