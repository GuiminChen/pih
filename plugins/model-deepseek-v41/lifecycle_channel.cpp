#include "lifecycle_channel.h"
#include <cerrno>
#include <sys/socket.h>

namespace pih::deepseek_v41 {
Result<RankLifecycleChannel> RankLifecycleChannel::Attach(int fd, std::int32_t pid, std::uint32_t uid,
    std::uint32_t rank, std::uint32_t world, RankChannelDirection direction) {
  auto endpoint = RankReceiptChannel::Attach(fd, pid, uid, rank, world, direction);
  if (!endpoint.ok()) return endpoint.status();
  return RankLifecycleChannel(std::move(*endpoint));
}
Result<std::array<std::uint8_t, 64>> RankLifecycleChannel::Frame(RankLifecycleKind kind, const SamplingIdentity& id) const {
  if (!id.epoch || id.plan_seq || !id.sequence_generation || !id.sampling_config_id ||
      (kind != RankLifecycleKind::kReady && kind != RankLifecycleKind::kRetire && kind != RankLifecycleKind::kReleased))
    return Status::InvalidArgument("Lifecycle frame requires base sequence identity and valid kind");
  std::array<std::uint8_t, 64> frame{};
  const auto put = [&](unsigned offset, std::uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i) frame[offset + i] = static_cast<std::uint8_t>(value >> (i * 8));
  };
  put(0, 0x314546494c484950ULL, 8); // PIHLIFE1
  put(8, 1, 4); put(12, frame.size(), 4); put(16, static_cast<std::uint32_t>(kind), 4);
  put(20, rank(), 4); put(24, world(), 4);
  put(32, id.epoch, 8); put(40, id.sequence_generation, 8); put(48, id.sampling_config_id, 8);
  return frame;
}
Status RankLifecycleChannel::Queue(RankLifecycleKind kind, const SamplingIdentity& id) {
  if (!sender_ready()) return Status::FailedPrecondition("Lifecycle channel cannot queue");
  auto frame = Frame(kind, id); if (!frame.ok()) return frame.status();
  frame_ = *frame; offset_ = 0; queued_ = true; return Status::Ok();
}
Result<bool> RankLifecycleChannel::PollSend(Clock::time_point deadline) {
  if (endpoint_.failed_ || endpoint_.fd_ < 0 || endpoint_.direction_ != RankChannelDirection::kSend || !queued_)
    return Status::FailedPrecondition("Lifecycle channel has no pending send");
  if (Clock::now() >= deadline) { endpoint_.failed_ = true; return Status::DeadlineExceeded("Lifecycle send expired"); }
  const auto sent = ::send(endpoint_.fd_, frame_.data() + offset_, frame_.size() - offset_, MSG_DONTWAIT | MSG_NOSIGNAL);
  if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return false;
  if (sent <= 0) { endpoint_.failed_ = true; return Status::Unavailable("Lifecycle send failed"); }
  offset_ += static_cast<std::size_t>(sent);
  if (offset_ != frame_.size()) return false;
  if (Clock::now() >= deadline) { endpoint_.failed_ = true; return Status::DeadlineExceeded("Lifecycle send completed late"); }
  queued_ = false; offset_ = 0; return true;
}
Result<bool> RankLifecycleChannel::PollReceive(RankLifecycleKind kind, const SamplingIdentity& id, Clock::time_point deadline) {
  if (endpoint_.failed_ || endpoint_.fd_ < 0 || endpoint_.direction_ != RankChannelDirection::kReceive)
    return Status::FailedPrecondition("Lifecycle channel cannot receive");
  auto expected = Frame(kind, id); if (!expected.ok()) return expected.status();
  if (offset_ && expected_ != *expected) {
    endpoint_.failed_ = true; return Status::FailedPrecondition("Lifecycle expectation changed during partial receive");
  }
  expected_ = *expected;
  if (Clock::now() >= deadline) { endpoint_.failed_ = true; return Status::DeadlineExceeded("Lifecycle receive expired"); }
  const auto received = ::recv(endpoint_.fd_, frame_.data() + offset_, frame_.size() - offset_, MSG_DONTWAIT);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return false;
  if (received <= 0) { endpoint_.failed_ = true; return Status::Unavailable("Lifecycle peer disconnected or receive failed"); }
  offset_ += static_cast<std::size_t>(received);
  if (offset_ != frame_.size()) return false;
  if (Clock::now() >= deadline || frame_ != expected_) {
    endpoint_.failed_ = true; return Status::FailedPrecondition("Lifecycle frame is late, noncanonical or has wrong identity/order");
  }
  offset_ = 0; return true;
}
}  // namespace pih::deepseek_v41
