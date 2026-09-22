#pragma once
#include "token_stop.h"
#include "pih/scheduler/output_burst_credit_pool.h"
#include <memory>

namespace pih::deepseek_v41 {
struct AcceptedToken final {
  SamplingObservation observation;
  std::uint32_t accepted_count = 0;
  TokenFinish finish = TokenFinish::kNone;
};
struct TokenPublication final {
  AcceptedToken record;
  std::array<char, 8448> visible{};
  std::uint32_t visible_size = 0;
};
struct TokenOutputLease final {
  std::uint64_t queue_instance = 0;
  OutputBurstLease credit{};
};
// Serialized, admission-allocated publication slots, not a network serializer.
// Keep this object alive while any ledger or output receipt borrows it.
class TokenOutputQueue final {
 public:
  static Result<std::unique_ptr<TokenOutputQueue>> Create(std::uint32_t slots, std::uint64_t publication_budget_bytes);
  TokenOutputQueue(const TokenOutputQueue&) = delete;
  TokenOutputQueue& operator=(const TokenOutputQueue&) = delete;
  Result<TokenOutputLease> Reserve(std::uint64_t plan);
  Status MarkInFlight(const TokenOutputLease& lease);
  Status ValidateCommitted(const TokenOutputLease& lease) const;
  Status AbortPrepared(const TokenOutputLease& lease);
  // Caller must first prove that no worker can still produce this result.
  Status DiscardRetired(const TokenOutputLease& lease);
  Result<TokenOutputLease> Publish(const TokenOutputLease& lease, const TokenPublication& publication);
  Result<TokenPublication> Read(const TokenOutputLease& lease) const;
  Status Release(const TokenOutputLease& lease);
 private:
  explicit TokenOutputQueue(OutputBurstCreditPool&& credits) : credits_(std::move(credits)) {}
  struct Slot { TokenPublication publication{}; TokenOutputLease lease{}; bool ready = false; };
  Status Identity(const TokenOutputLease& lease) const;
  OutputBurstCreditPool credits_;
  std::vector<Slot> slots_;
  std::uint64_t instance_ = 0;
};
}  // namespace pih::deepseek_v41
