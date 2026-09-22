#pragma once
#include "pih/core/result.h"
#include <chrono>
#include <cstdint>
#include <optional>
#include <string_view>

namespace pih::deepseek_v41 {
// Linux abstract Unix sockets: no filesystem entry, unlink or inherited peer
// credential assumption. Bind one unpredictable name per rank/protocol stream.
// The supervisor listens before spawn; the exec'd child initiates connection.
class RankSocket final {
 public:
  using Clock = std::chrono::steady_clock;
  static Result<RankSocket> Listen(std::string_view name);
  static Result<RankSocket> Connect(std::string_view name, std::int32_t server_pid,
      std::uint32_t server_uid, Clock::time_point deadline);
  RankSocket(const RankSocket&) = delete;
  RankSocket& operator=(const RankSocket&) = delete;
  RankSocket(RankSocket&& other) noexcept;
  RankSocket& operator=(RankSocket&&) = delete;
  ~RankSocket();
  // Listener only. Expected PID comes from the supervisor's spawn/pidfd ledger.
  // One accept per poll; foreign peers fail this admission, not become ranks.
  Result<std::optional<RankSocket>> PollAccept(std::int32_t pid, std::uint32_t uid, Clock::time_point deadline);
  Result<bool> PollConnected();
  // Borrow for channel Attach, which retains its own CLOEXEC duplicate. Keep
  // only one protocol reader/writer per connection and close this owner after
  // attaching. Never mutate its O_NONBLOCK flag through another duplicate.
  Result<int> ConnectedDescriptor() const;
 private:
  enum class State { kListening, kConnecting, kConnected, kFailed };
  RankSocket() = default;
  Status Authenticate(std::int32_t pid, std::uint32_t uid) const;
  int fd_ = -1;
  State state_ = State::kFailed;
  std::int32_t pid_ = 0;
  std::uint32_t uid_ = 0;
  Clock::time_point deadline_{};
};
}  // namespace pih::deepseek_v41
