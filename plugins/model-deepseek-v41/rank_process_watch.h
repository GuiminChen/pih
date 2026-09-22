#pragma once
#include "rank_commit.h"

namespace pih::deepseek_v41 {
struct RankProcessBinding final { int pidfd = -1; std::int32_t pid = 0; };
// Supervisor supplies pidfds retained from spawn. Normal reaping is exclusive
// and only begins after all lifecycle release notices have been admitted.
class RankProcessWatch final {
 public:
  static Result<RankProcessWatch> Attach(std::span<const RankProcessBinding> ranks);
  RankProcessWatch(const RankProcessWatch&) = delete;
  RankProcessWatch& operator=(const RankProcessWatch&) = delete;
  RankProcessWatch(RankProcessWatch&& other) noexcept;
  ~RankProcessWatch();
  Status CheckLive();
  Result<bool> PollNormalExit(std::chrono::steady_clock::time_point deadline);
  std::uint32_t reaped_mask() const noexcept { return reaped_; }
  std::uint32_t world() const noexcept { return world_; }
  std::int32_t pid(std::uint32_t rank) const noexcept { return rank < world_ ? ranks_[rank].pid : 0; }
 private:
  friend class RankProcessRetirement;
  RankProcessWatch() = default;
  std::array<RankProcessBinding, 8> ranks_{};
  std::uint32_t world_ = 0;
  std::uint32_t reaped_ = 0;
  bool reaping_ = false, normal_exit_failed_ = false;
  bool failed_ = false;
};
enum class RankRetirementState { kTerminating, kKilling, kComplete, kFailed };
// Parent/subreaper-only fault teardown. No numeric-PID signal fallback.
class RankProcessRetirement final {
 public:
  using Clock = std::chrono::steady_clock;
  static Result<std::unique_ptr<RankProcessRetirement>> Create(RankProcessWatch& processes, TokenLedger& ledger,
      Clock::time_point kill_after, Clock::time_point deadline);
  RankProcessRetirement(const RankProcessRetirement&) = delete;
  RankProcessRetirement& operator=(const RankProcessRetirement&) = delete;
  ~RankProcessRetirement();
  Result<RankRetirementState> Poll();
  // Reconciles only unaccepted output credit; no CUDA memory/free guarantee.
  Status DiscardOutput();
  std::uint32_t reaped_mask() const noexcept { return reaped_; }
 private:
  RankProcessRetirement() = default;
  std::array<RankProcessBinding, 8> ranks_{};
  std::uint32_t world_ = 0, terminated_ = 0, killed_ = 0, reaped_ = 0;
  Clock::time_point kill_after_{}, deadline_{};
  RankRetirementState state_ = RankRetirementState::kTerminating;
  TokenLedger* ledger_ = nullptr;
  std::uint64_t pending_plan_ = 0;
};
}  // namespace pih::deepseek_v41
