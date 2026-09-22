#pragma once
#include "token_ledger.h"
#include <chrono>

namespace pih::deepseek_v41 {
// In-process completion receipt. Not a network authentication credential.
class RankStepReceipt final {
 public:
  const SamplingIdentity& identity() const noexcept { return identity_; }
  std::uint64_t ordinal() const noexcept { return ordinal_; }
  std::uint32_t processed_length() const noexcept { return processed_; }
  std::uint32_t rank() const noexcept { return rank_; }
  std::uint32_t world() const noexcept { return world_; }
  const std::optional<SamplingObservation>& candidate() const noexcept { return candidate_; }
  bool terminal() const noexcept { return terminal_; }
  // Fixed little-endian representation, independent of C++ ABI/padding.
  std::array<std::uint8_t, 256> Encode() const noexcept;
 private:
  friend class InferenceOperation;
  friend class RankCommitGate;
  friend class RankWorkerLoop;
  friend class RankReceiptChannel;
  RankStepReceipt() = default;
  SamplingIdentity identity_{};
  std::uint64_t ordinal_ = 0;
  std::uint32_t processed_ = 0, rank_ = 0, world_ = 0;
  std::optional<SamplingObservation> candidate_;
  bool terminal_ = false;
};
// Single serialized controller step. Input credit must be marked in-flight
// before dispatch. Transport authentication and global fault ordering remain
// the controller's responsibility; no network/worker authority is invented here.
class RankCommitGate final {
 public:
  using Clock = std::chrono::steady_clock;
  static Result<RankCommitGate> Create(const TokenSamplingRequest& request, std::uint32_t world, Clock::time_point deadline);
  Status Submit(std::uint32_t source_rank, const RankStepReceipt& receipt);
  // source_rank comes from the authenticated connection, never from the frame.
  Status SubmitWire(std::uint32_t source_rank, std::span<const std::uint8_t> frame);
  bool ready() const noexcept { return !failed_ && !committed_ && mask_ == ((1U << world_) - 1); }
  void Fail() noexcept { failed_ = true; }
  Result<SamplingObservation> Candidate() const;
  Result<TokenOutputLease> Commit(TokenLedger& ledger, std::string_view decoded_token_bytes);
 private:
  RankCommitGate() = default;
  TokenSamplingRequest request_{};
  std::optional<SamplingObservation> candidate_;
  Clock::time_point deadline_{};
  std::uint32_t world_ = 0, mask_ = 0;
  bool failed_ = false, committed_ = false;
};
}  // namespace pih::deepseek_v41
