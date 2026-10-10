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
  if (endpoint_.failed_ || endpoint_.fd_ < 0 || endpoint_.direction_ != RankChannelDirection::kSend || queued_ || !packed_frame_.empty())
    return Status::FailedPrecondition("Request channel cannot queue another frame");
  auto encoded = request.Encode(); if (!encoded.ok()) return encoded.status();
  frame_ = *encoded; offset_ = 0; queued_ = true; return Status::Ok();
}
Result<bool> RankRequestChannel::PollSend(Clock::time_point deadline) {
  if (endpoint_.failed_ || endpoint_.fd_ < 0 || endpoint_.direction_ != RankChannelDirection::kSend || !queued_ || !packed_frame_.empty())
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
  if (endpoint_.failed_ || endpoint_.fd_ < 0 || endpoint_.direction_ != RankChannelDirection::kReceive || !packed_frame_.empty())
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
Status RankRequestChannel::QueuePacked(const PackedRequest& request) {
  if (!sender_ready() || offset_ || !packed_frame_.empty())
    return Status::FailedPrecondition("Packed request channel has outstanding frame");
  auto bytes = request.Encode(); if (!bytes.ok()) return bytes.status();
  packed_frame_ = std::move(*bytes); queued_ = true; return Status::Ok();
}
Result<bool> RankRequestChannel::PollSendPacked(Clock::time_point deadline) {
  if (endpoint_.failed_ || endpoint_.fd_ < 0 || endpoint_.direction_ != RankChannelDirection::kSend ||
      !queued_ || packed_frame_.empty()) return Status::FailedPrecondition("Packed request send not ready");
  if (Clock::now() >= deadline) { endpoint_.failed_ = true; return Status::DeadlineExceeded("Packed send expired"); }
  const auto sent = ::send(endpoint_.fd_, packed_frame_.data() + offset_, packed_frame_.size() - offset_, MSG_DONTWAIT | MSG_NOSIGNAL);
  if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return false;
  if (sent <= 0) { endpoint_.failed_ = true; return Status::Unavailable("Packed request send failed"); }
  offset_ += static_cast<std::size_t>(sent);
  if (offset_ != packed_frame_.size()) return false;
  if (Clock::now() >= deadline) { endpoint_.failed_ = true; return Status::DeadlineExceeded("Packed send retired after deadline"); }
  packed_frame_.clear(); queued_ = false; offset_ = 0; return true;
}
Result<std::optional<PackedRequest>> RankRequestChannel::PollReceivePacked(Clock::time_point deadline,
    std::uint32_t maximum_sequences, std::uint32_t maximum_tokens) {
  if (endpoint_.failed_ || endpoint_.fd_ < 0 || endpoint_.direction_ != RankChannelDirection::kReceive ||
      !maximum_sequences || maximum_sequences > 32 || !maximum_tokens)
    return Status::FailedPrecondition("Packed request receiver or reservation invalid");
  if (Clock::now() >= deadline) { endpoint_.failed_ = true; return Status::DeadlineExceeded("Packed receive expired"); }
  if (packed_frame_.empty()) {
    if (offset_) return Status::FailedPrecondition("Ordinary frame partially received on packed endpoint");
    packed_frame_.resize(32);
  }
  const auto received = ::recv(endpoint_.fd_, packed_frame_.data() + offset_, packed_frame_.size() - offset_, MSG_DONTWAIT);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return std::optional<PackedRequest>{};
  if (received <= 0) { endpoint_.failed_ = true; return Status::Unavailable("Packed receiver disconnected"); }
  offset_ += static_cast<std::size_t>(received);
  if (offset_ != packed_frame_.size()) return std::optional<PackedRequest>{};
  if (packed_frame_.size() == 32) {
    const auto get = [&](unsigned at, unsigned count) {
      std::uint64_t value = 0;
      for (unsigned i = 0; i < count; ++i) value |= std::uint64_t(packed_frame_[at + i]) << (i * 8);
      return value;
    };
    const auto members = get(16, 4), bytes = get(12, 4);
    if (get(0, 8) != 0x31504b4341504850ULL || get(8, 4) != 1 || !members || members > maximum_sequences ||
        bytes != 32 + members * (8 + InferenceRequest::kWireBytes) || get(20, 4) > maximum_tokens || get(24, 8)) {
      endpoint_.failed_ = true; return Status::InvalidArgument("Packed frame header invalid before allocation");
    }
    packed_frame_.resize(bytes); return std::optional<PackedRequest>{};
  }
  auto decoded = PackedRequest::Decode(packed_frame_, maximum_sequences, maximum_tokens);
  if (!decoded.ok()) { endpoint_.failed_ = true; return decoded.status(); }
  if (Clock::now() >= deadline) { endpoint_.failed_ = true; return Status::DeadlineExceeded("Packed receive completed after deadline"); }
  packed_frame_.clear(); offset_ = 0; return std::optional<PackedRequest>{std::move(*decoded)};
}
}  // namespace pih::deepseek_v41
