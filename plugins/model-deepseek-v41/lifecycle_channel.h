#pragma once
#include "rank_channel.h"

namespace pih::deepseek_v41 {
enum class RankLifecycleKind : std::uint32_t { kReady = 1, kRetire = 2, kReleased = 3 };
// Dedicated one-way authenticated endpoint. Never share its socket with token
// requests/receipts. Caller enforces the global ready/terminal/release barriers.
class RankLifecycleChannel final {
 public:
  using Clock = RankReceiptChannel::Clock;
  static Result<RankLifecycleChannel> Attach(int fd, std::int32_t pid, std::uint32_t uid,
      std::uint32_t rank, std::uint32_t world, RankChannelDirection direction);
  RankLifecycleChannel(const RankLifecycleChannel&) = delete;
  RankLifecycleChannel& operator=(const RankLifecycleChannel&) = delete;
  RankLifecycleChannel(RankLifecycleChannel&&) noexcept = default;
  RankLifecycleChannel& operator=(RankLifecycleChannel&&) = delete;
  Status Queue(RankLifecycleKind kind, const SamplingIdentity& identity);
  Result<bool> PollSend(Clock::time_point deadline);
  Result<bool> PollReceive(RankLifecycleKind kind, const SamplingIdentity& identity, Clock::time_point deadline);
  std::uint32_t rank() const noexcept { return endpoint_.rank(); }
  std::uint32_t world() const noexcept { return endpoint_.world(); }
  std::int32_t peer_pid() const noexcept { return endpoint_.peer_pid(); }
  bool sender_ready() const noexcept { return endpoint_.sender_ready() && !queued_; }
  bool receiver_ready() const noexcept { return endpoint_.receiver_ready() && !offset_; }
 private:
  explicit RankLifecycleChannel(RankReceiptChannel&& endpoint) : endpoint_(std::move(endpoint)) {}
  Result<std::array<std::uint8_t, 64>> Frame(RankLifecycleKind kind, const SamplingIdentity& identity) const;
  RankReceiptChannel endpoint_;
  std::array<std::uint8_t, 64> frame_{}, expected_{};
  std::size_t offset_ = 0;
  bool queued_ = false;
};
}  // namespace pih::deepseek_v41
