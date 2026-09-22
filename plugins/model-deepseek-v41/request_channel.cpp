#include "request_channel.h"
#include <cerrno>
#include <sys/socket.h>

namespace pih::deepseek_v41 {
Result<RankRequestChannel> RankRequestChannel::Attach(int fd, std::int32_t pid, std::uint32_t uid,
    std::uint32_t rank, std::uint32_t world, RankChannelDirection direction) {
  auto endpoint = RankReceiptChannel::Attach(fd, pid, uid, rank, world, direction);
  if (!endpoint.ok()) return endpoint.status();
  return RankRequestChannel(std::move(*endpoint));
}
Status RankRequestChannel::Queue(const InferenceRequest& request) {
  if (endpoint_.failed_ || endpoint_.fd_ < 0 || endpoint_.direction_ != RankChannelDirection::kSend || queued_)
    return Status::FailedPrecondition("Request channel cannot queue another frame");
  auto encoded = request.Encode(); if (!encoded.ok()) return encoded.status();
  frame_ = *encoded; offset_ = 0; queued_ = true; return Status::Ok();
}
Result<bool> RankRequestChannel::PollSend(Clock::time_point deadline) {
  if (endpoint_.failed_ || endpoint_.fd_ < 0 || endpoint_.direction_ != RankChannelDirection::kSend || !queued_)
    return Status::FailedPrecondition("Request channel has no pending send");
  if (Clock::now() >= deadline) { endpoint_.failed_ = true; return Status::DeadlineExceeded("Request send timed out"); }
  const auto sent = ::send(endpoint_.fd_, frame_.data() + offset_, frame_.size() - offset_, MSG_DONTWAIT | MSG_NOSIGNAL);
  if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return false;
  if (sent <= 0) { endpoint_.failed_ = true; return Status::Unavailable("Request channel send failed"); }
  offset_ += static_cast<std::size_t>(sent);
  if (offset_ != frame_.size()) return false;
  if (Clock::now() >= deadline) { endpoint_.failed_ = true; return Status::DeadlineExceeded("Request send completed after deadline"); }
  queued_ = false; offset_ = 0; return true;
}
Result<std::optional<InferenceRequest>> RankRequestChannel::PollReceive(Clock::time_point deadline) {
  if (endpoint_.failed_ || endpoint_.fd_ < 0 || endpoint_.direction_ != RankChannelDirection::kReceive)
    return Status::FailedPrecondition("Request channel cannot receive");
  if (Clock::now() >= deadline) { endpoint_.failed_ = true; return Status::DeadlineExceeded("Request receive timed out"); }
  const auto received = ::recv(endpoint_.fd_, frame_.data() + offset_, frame_.size() - offset_, MSG_DONTWAIT);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return std::optional<InferenceRequest>{};
  if (received <= 0) { endpoint_.failed_ = true; return Status::Unavailable("Request channel disconnected or failed"); }
  offset_ += static_cast<std::size_t>(received);
  if (offset_ != frame_.size()) return std::optional<InferenceRequest>{};
  if (Clock::now() >= deadline) { endpoint_.failed_ = true; return Status::DeadlineExceeded("Request received after deadline"); }
  auto decoded = InferenceRequest::Decode(frame_);
  if (!decoded.ok()) { endpoint_.failed_ = true; return decoded.status(); }
  offset_ = 0; return std::optional<InferenceRequest>{std::move(*decoded)};
}
}  // namespace pih::deepseek_v41
