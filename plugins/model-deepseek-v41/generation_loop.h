#pragma once
#include "rank_collector.h"
#include "request_channel.h"

namespace pih::deepseek_v41 {
enum class GenerationState { kPreparing, kSending, kCollecting, kOutputReady, kFinishQueue, kFinishSend, kFinishReceive, kComplete, kFailed };
// Serialized controller sequence. Supervisor fault events must call Fail before
// Poll can commit; channels/ledger/token byte table are borrowed exclusively.
class GenerationLoop final {
 public:
  using Clock = std::chrono::steady_clock;
  static Result<std::unique_ptr<GenerationLoop>> Create(TokenLedger& ledger, std::span<const std::uint32_t> prompt,
      std::span<const std::string_view> token_bytes, std::span<RankRequestChannel* const> requests,
      std::span<RankReceiptChannel* const> receipts, RankProcessWatch& processes, std::uint64_t first_plan, Clock::time_point deadline);
  GenerationLoop(const GenerationLoop&) = delete;
  GenerationLoop& operator=(const GenerationLoop&) = delete;
  ~GenerationLoop();
  Result<GenerationState> Poll();
  // Transfers the publication lease, not bytes or credit release. Caller reads
  // and releases it through the originating TokenOutputQueue after consumption.
  Result<TokenOutputLease> TakeOutput();
  void Fail() noexcept;
 private:
  friend class GenerationSession;
  explicit GenerationLoop(TokenLedger& ledger) : ledger_(ledger) {}
  Result<GenerationState> PollImpl();
  TokenLedger& ledger_;
  RankProcessWatch* processes_ = nullptr;
  InferenceRequest request_{};
  std::span<const std::string_view> token_bytes_;
  std::array<RankRequestChannel*, 8> requests_{};
  std::array<RankReceiptChannel*, 8> receipts_{};
  std::unique_ptr<RankReceiptCollector> collector_;
  std::optional<TokenOutputLease> output_;
  Clock::time_point deadline_{};
  std::uint64_t plan_ = 0;
  std::uint32_t world_ = 0, sent_ = 0;
  GenerationState state_ = GenerationState::kPreparing;
};
}  // namespace pih::deepseek_v41
