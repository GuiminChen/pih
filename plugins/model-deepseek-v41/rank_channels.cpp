#include "rank_channels.h"
#include <new>

namespace pih::deepseek_v41 {
Status RankChannels::Validate(std::span<const std::string_view> names, std::uint32_t rank, std::uint32_t world) {
  if (names.size() != 4 || (world != 2 && world != 4 && world != 8) || rank >= world)
    return Status::InvalidArgument("Rank channel bundle names or topology invalid");
  for (unsigned i = 0; i < 4; ++i) {
    if (names[i].size() < 32 || names[i].size() > 96)
      return Status::InvalidArgument("Rank endpoint name length invalid");
    for (const char c : names[i])
      if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '.'))
        return Status::InvalidArgument("Rank endpoint name character invalid");
    for (unsigned j = 0; j < i; ++j) if (names[i] == names[j])
      return Status::InvalidArgument("Rank protocol streams require distinct endpoint names");
  }
  return Status::Ok();
}
Result<std::unique_ptr<RankChannels>> RankChannels::Listen(std::span<const std::string_view> names,
    std::uint32_t rank, std::uint32_t world) {
  const auto valid = Validate(names, rank, world); if (!valid.ok()) return valid;
  try {
    auto bundle = std::unique_ptr<RankChannels>(new RankChannels);
    bundle->rank_ = rank; bundle->world_ = world; bundle->server_ = true;
    for (unsigned i = 0; i < 4; ++i) {
      auto socket = RankSocket::Listen(names[i]); if (!socket.ok()) return socket.status();
      bundle->sockets_[i].emplace(std::move(*socket));
    }
    bundle->state_ = State::kListening;
    return bundle;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Rank listener bundle allocation failed"); }
}
Result<std::unique_ptr<RankChannels>> RankChannels::Connect(std::span<const std::string_view> names,
    std::uint32_t rank, std::uint32_t world, std::int32_t pid, std::uint32_t uid, Clock::time_point deadline) {
  const auto valid = Validate(names, rank, world); if (!valid.ok()) return valid;
  if (pid <= 0 || deadline <= Clock::now()) return Status::InvalidArgument("Rank bundle peer or deadline invalid");
  try {
    auto bundle = std::unique_ptr<RankChannels>(new RankChannels);
    bundle->rank_ = rank; bundle->world_ = world; bundle->pid_ = pid; bundle->uid_ = uid; bundle->deadline_ = deadline;
    for (unsigned i = 0; i < 4; ++i) {
      auto socket = RankSocket::Connect(names[i], pid, uid, deadline); if (!socket.ok()) return socket.status();
      bundle->sockets_[i].emplace(std::move(*socket));
    }
    bundle->state_ = State::kConnecting;
    return bundle;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Rank connection bundle allocation failed"); }
}
Status RankChannels::BeginAccept(std::int32_t pid, std::uint32_t uid, Clock::time_point deadline) {
  if (state_ != State::kListening) return Status::FailedPrecondition("Rank bundle is not awaiting spawn identity");
  if (pid <= 0 || deadline <= Clock::now()) return Status::InvalidArgument("Rank child identity or deadline invalid");
  pid_ = pid; uid_ = uid; deadline_ = deadline; state_ = State::kAccepting;
  return Status::Ok();
}
Status RankChannels::Attach(unsigned slot, const RankSocket& socket) {
  const auto fd = socket.ConnectedDescriptor(); if (!fd.ok()) return fd.status();
  const bool supervisor_sends = slot == 0 || slot == 2;
  const auto direction = server_ == supervisor_sends ? RankChannelDirection::kSend : RankChannelDirection::kReceive;
  if (slot == 0) {
    auto channel = RankRequestChannel::Attach(*fd, pid_, uid_, rank_, world_, direction);
    if (!channel.ok()) return channel.status();
    requests_.emplace(std::move(*channel));
  } else if (slot == 1) {
    auto channel = RankReceiptChannel::Attach(*fd, pid_, uid_, rank_, world_, direction);
    if (!channel.ok()) return channel.status();
    receipts_.emplace(std::move(*channel));
  } else {
    auto channel = RankLifecycleChannel::Attach(*fd, pid_, uid_, rank_, world_, direction);
    if (!channel.ok()) return channel.status();
    (slot == 2 ? commands_ : notices_).emplace(std::move(*channel));
  }
  attached_ |= 1U << slot;
  return Status::Ok();
}
Result<bool> RankChannels::Poll() {
  if (state_ == State::kReady) return true;
  if (state_ != State::kAccepting && state_ != State::kConnecting)
    return Status::FailedPrecondition("Rank bundle has no pending admission");
  if (Clock::now() >= deadline_) { state_ = State::kFailed; return Status::DeadlineExceeded("Rank bundle admission expired"); }
  try {
    for (unsigned i = 0; i < 4; ++i) {
      if (attached_ & (1U << i)) continue;
      Status attached = Status::Ok();
      if (server_) {
        auto accepted = sockets_[i]->PollAccept(pid_, uid_, deadline_);
        if (!accepted.ok()) { state_ = State::kFailed; return accepted.status(); }
        if (!*accepted) continue;
        attached = Attach(i, **accepted);
      } else {
        auto ready = sockets_[i]->PollConnected();
        if (!ready.ok()) { state_ = State::kFailed; return ready.status(); }
        if (!*ready) continue;
        attached = Attach(i, *sockets_[i]);
      }
      if (!attached.ok()) { state_ = State::kFailed; return attached; }
      sockets_[i].reset(); // Close listener/temporary connector; channel owns duplicate.
    }
    if (Clock::now() >= deadline_) { state_ = State::kFailed; return Status::DeadlineExceeded("Rank bundle admission completed late"); }
    if (attached_ == 15U) { state_ = State::kReady; return true; }
    return false;
  } catch (const std::bad_alloc&) {
    state_ = State::kFailed; return Status::ResourceExhausted("Rank bundle admission allocation failed");
  }
}
Result<RankChannelViews> RankChannels::Views() {
  if (state_ != State::kReady) return Status::FailedPrecondition("Rank bundle admission is incomplete or failed");
  return RankChannelViews{*requests_, *receipts_, *commands_, *notices_};
}
}  // namespace pih::deepseek_v41
