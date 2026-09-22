#pragma once
#include "rank_channel.h"
#include "inference_request.h"

namespace pih::deepseek_v41 {
// Dedicated request endpoint: never attach receipt and request receivers to
// the same socket. Uses the same authenticated Unix endpoint admission.
class RankRequestChannel final {
 public:
  using Clock = std::chrono::steady_clock;
  static Result<RankRequestChannel> Attach(int fd, std::int32_t peer_pid, std::uint32_t peer_uid,
      std::uint32_t rank, std::uint32_t world, RankChannelDirection direction);
  RankRequestChannel(const RankRequestChannel&) = delete;
  RankRequestChannel& operator=(const RankRequestChannel&) = delete;
  RankRequestChannel(RankRequestChannel&&) noexcept = default;
  RankRequestChannel& operator=(RankRequestChannel&&) = delete;
  Status Queue(const InferenceRequest& request);
  Result<bool> PollSend(Clock::time_point deadline);
  Result<std::optional<InferenceRequest>> PollReceive(Clock::time_point deadline);
  std::uint32_t rank() const noexcept { return endpoint_.rank(); }
  std::uint32_t world() const noexcept { return endpoint_.world(); }
  std::int32_t peer_pid() const noexcept { return endpoint_.peer_pid(); }
  bool receiver_ready() const noexcept { return endpoint_.receiver_ready() && !offset_; }
  bool sender_ready() const noexcept { return endpoint_.sender_ready() && !queued_; }
 private:
  explicit RankRequestChannel(RankReceiptChannel&& endpoint) : endpoint_(std::move(endpoint)) {}
  RankReceiptChannel endpoint_;
  std::array<std::uint8_t, InferenceRequest::kWireBytes> frame_{};
  std::size_t offset_ = 0;
  bool queued_ = false;
};
}  // namespace pih::deepseek_v41
