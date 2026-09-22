#pragma once
#include "rank_socket.h"
#include "request_channel.h"
#include "lifecycle_channel.h"
#include <memory>

namespace pih::deepseek_v41 {
struct RankChannelViews final {
  RankRequestChannel& requests;
  RankReceiptChannel& receipts;
  RankLifecycleChannel& commands;
  RankLifecycleChannel& notices;
};
// Stable-address four-stream bundle. Order: request, receipt, command, notice.
// Listen before spawn, then BeginAccept with the pidfd-backed child identity.
class RankChannels final {
 public:
  using Clock = RankSocket::Clock;
  static Result<std::unique_ptr<RankChannels>> Listen(std::span<const std::string_view> names,
      std::uint32_t rank, std::uint32_t world);
  static Result<std::unique_ptr<RankChannels>> Connect(std::span<const std::string_view> names,
      std::uint32_t rank, std::uint32_t world, std::int32_t server_pid,
      std::uint32_t uid, Clock::time_point deadline);
  RankChannels(const RankChannels&) = delete;
  RankChannels& operator=(const RankChannels&) = delete;
  Status BeginAccept(std::int32_t child_pid, std::uint32_t uid, Clock::time_point deadline);
  Result<bool> Poll();
  Result<RankChannelViews> Views();
 private:
  RankChannels() = default;
  static Status Validate(std::span<const std::string_view> names, std::uint32_t rank, std::uint32_t world);
  Status Attach(unsigned slot, const RankSocket& socket);
  enum class State { kListening, kAccepting, kConnecting, kReady, kFailed };
  std::array<std::optional<RankSocket>, 4> sockets_;
  std::optional<RankRequestChannel> requests_;
  std::optional<RankReceiptChannel> receipts_;
  std::optional<RankLifecycleChannel> commands_, notices_;
  std::uint32_t rank_ = 0, world_ = 0, uid_ = 0, attached_ = 0;
  std::int32_t pid_ = 0;
  bool server_ = false;
  Clock::time_point deadline_{};
  State state_ = State::kFailed;
};
}  // namespace pih::deepseek_v41
