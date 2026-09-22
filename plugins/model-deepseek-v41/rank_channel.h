#pragma once
#include "rank_commit.h"

namespace pih::deepseek_v41 {
enum class RankChannelDirection { kSend, kReceive };
// Exclusive Linux AF_UNIX stream endpoint. Owns a CLOEXEC duplicate, not the
// original descriptor. The supervisor supplies expected peer identity/rank.
class RankReceiptChannel final {
 public:
  using Clock = std::chrono::steady_clock;
  static Result<RankReceiptChannel> Attach(int fd, std::int32_t peer_pid, std::uint32_t peer_uid,
      std::uint32_t rank, std::uint32_t world, RankChannelDirection direction);
  RankReceiptChannel(const RankReceiptChannel&) = delete;
  RankReceiptChannel& operator=(const RankReceiptChannel&) = delete;
  RankReceiptChannel(RankReceiptChannel&& other) noexcept;
  RankReceiptChannel& operator=(RankReceiptChannel&&) = delete;
  ~RankReceiptChannel();
  Status Queue(const RankStepReceipt& receipt);
  Result<bool> PollSend(Clock::time_point deadline);
  Result<bool> PollReceive(RankCommitGate& gate, Clock::time_point deadline);
  Result<bool> PollTerminal(const SamplingIdentity& identity, std::uint64_t ordinal,
      std::uint32_t processed_length, Clock::time_point deadline);
  // Controller checks every rank boundary immediately before local commit.
  // EOF or unexpected queued data closes the gate; EAGAIN means quiet now.
  Status CheckQuiet(RankCommitGate& gate);
  std::uint32_t rank() const noexcept { return rank_; }
  std::uint32_t world() const noexcept { return world_; }
  std::int32_t peer_pid() const noexcept { return peer_pid_; }
  bool sender_ready() const noexcept { return fd_ >= 0 && !failed_ && direction_ == RankChannelDirection::kSend && !queued_; }
  bool receiver_ready() const noexcept {
    return fd_ >= 0 && !failed_ && direction_ == RankChannelDirection::kReceive && !offset_;
  }
 private:
  friend class RankRequestChannel;
  friend class RankLifecycleChannel;
  RankReceiptChannel() = default;
  int fd_ = -1;
  std::int32_t peer_pid_ = 0;
  std::uint32_t rank_ = 0, world_ = 0;
  RankChannelDirection direction_ = RankChannelDirection::kReceive;
  std::array<std::uint8_t, 256> frame_{};
  std::size_t offset_ = 0;
  bool queued_ = false, failed_ = false;
};
}  // namespace pih::deepseek_v41
