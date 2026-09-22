#pragma once
#include "rank_channel.h"
#include "rank_process_watch.h"

namespace pih::deepseek_v41 {
// Serialized controller helper. The supervisor calls Fail before any commit
// when its ordered fault/epoch state changes. All borrowed channels are exclusive.
class RankReceiptCollector final {
 public:
  using Clock = std::chrono::steady_clock;
  static Result<std::unique_ptr<RankReceiptCollector>> Begin(TokenLedger& ledger, std::uint64_t plan,
      std::span<RankReceiptChannel* const> channels, Clock::time_point deadline);
  RankReceiptCollector(const RankReceiptCollector&) = delete;
  RankReceiptCollector& operator=(const RankReceiptCollector&) = delete;
  ~RankReceiptCollector();
  const TokenSamplingRequest& request() const noexcept { return request_; }
  Result<bool> Poll();
  Result<SamplingObservation> Candidate() const;
  Result<TokenOutputLease> Commit(std::string_view decoded_token_bytes, RankProcessWatch& processes);
  void Fail() noexcept;
 private:
  RankReceiptCollector() = default;
  TokenLedger* ledger_ = nullptr;
  TokenSamplingRequest request_{};
  std::optional<RankCommitGate> gate_;
  std::array<RankReceiptChannel*, 8> channels_{};
  Clock::time_point deadline_{};
  std::uint32_t world_ = 0, mask_ = 0;
  bool active_ = false, failed_ = false, committed_ = false;
};
}  // namespace pih::deepseek_v41
