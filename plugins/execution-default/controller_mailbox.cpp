#include "pih/scheduler/controller_mailbox.h"

#include <limits>

namespace pih {
namespace {
bool nonzero(const Sha256Digest& digest) {
  for (const auto value : digest.bytes) {
    if (value != std::byte{0}) return true;
  }
  return false;
}
bool valid_command(ControllerCommandKind kind) {
  return kind == ControllerCommandKind::kAdmit ||
         kind == ControllerCommandKind::kCancel ||
         kind == ControllerCommandKind::kShutdown;
}
bool valid_event(ControllerOutputEventKind kind) {
  return kind >= ControllerOutputEventKind::kAdmitted &&
         kind <= ControllerOutputEventKind::kFailed;
}
}  // namespace

ControllerMailbox::ControllerMailbox(ControllerMailboxLimits limits)
    : commands_(limits.command_capacity), events_(limits.event_capacity) {}

ControllerMailbox::ControllerMailbox(ControllerMailbox&& other) noexcept {
  std::scoped_lock lock(other.mutex_);
  commands_ = std::move(other.commands_);
  events_ = std::move(other.events_);
  command_head_ = other.command_head_;
  command_size_ = other.command_size_;
  event_head_ = other.event_head_;
  event_size_ = other.event_size_;
  next_command_sequence_ = other.next_command_sequence_;
  next_event_sequence_ = other.next_event_sequence_;
  controller_thread_ = other.controller_thread_;
}

Result<ControllerMailbox> ControllerMailbox::Create(
    ControllerMailboxLimits limits) {
  if (limits.command_capacity == 0 || limits.event_capacity == 0) {
    return Status::InvalidArgument("controller mailbox limits must be nonzero");
  }
  return ControllerMailbox(limits);
}

Result<ControllerCommand> ControllerMailbox::submit_command(
    ControllerCommandKind kind, std::uint64_t epoch,
    std::uint64_t request_generation, Sha256Digest payload_digest) {
  std::lock_guard lock(mutex_);
  if (!valid_command(kind) || epoch == 0 || request_generation == 0 ||
      !nonzero(payload_digest)) {
    return Status::InvalidArgument("controller command identity is invalid");
  }
  if (command_size_ == commands_.size()) {
    return Status::ResourceExhausted("controller command mailbox is full");
  }
  if (next_command_sequence_ == 0) {
    return Status::FailedPrecondition("controller command sequence wrapped");
  }
  const ControllerCommand command{kind, next_command_sequence_, epoch,
                                  request_generation, payload_digest};
  const auto tail = (command_head_ + command_size_) % commands_.size();
  commands_[tail] = command;
  ++command_size_;
  ++next_command_sequence_;
  return command;
}

Status ControllerMailbox::require_controller_thread_locked() {
  const auto current = std::this_thread::get_id();
  if (!controller_thread_.has_value()) {
    controller_thread_ = current;
    return Status::Ok();
  }
  if (*controller_thread_ != current) {
    return Status::FailedPrecondition("controller mailbox has another owner");
  }
  return Status::Ok();
}

Result<std::optional<ControllerCommand>>
ControllerMailbox::try_take_command() {
  std::lock_guard lock(mutex_);
  const auto owner = require_controller_thread_locked();
  if (!owner.ok()) return owner;
  if (command_size_ == 0) return std::optional<ControllerCommand>{};
  const auto command = commands_[command_head_];
  command_head_ = (command_head_ + 1) % commands_.size();
  --command_size_;
  return std::optional<ControllerCommand>{command};
}

Result<ControllerOutputEvent> ControllerMailbox::publish_event(
    const ControllerOutputEventDraft& draft) {
  std::lock_guard lock(mutex_);
  const auto owner = require_controller_thread_locked();
  if (!owner.ok()) return owner;
  const bool token_event =
      draft.kind == ControllerOutputEventKind::kTokenCommitted;
  const bool plan_bundle = draft.plan_sequence != 0 ||
                           draft.plan_event_index != 0 ||
                           draft.plan_event_count != 0;
  const bool terminal_reason =
      draft.finish_reason != ControllerFinishReason::kNone;
  if (!valid_event(draft.kind) || draft.epoch == 0 ||
      draft.request_generation == 0 || !nonzero(draft.state_digest) ||
      (token_event && (draft.token_ordinal == 0 || draft.token_id >= 151936)) ||
      (!token_event && (draft.token_ordinal != 0 || draft.token_id != 0)) ||
      (plan_bundle && (draft.plan_sequence == 0 || draft.plan_event_count == 0 ||
                       draft.plan_event_index >= draft.plan_event_count)) ||
      (token_event && (!plan_bundle || terminal_reason)) ||
      (terminal_reason &&
       (draft.kind != ControllerOutputEventKind::kDraining || !plan_bundle))) {
    return Status::InvalidArgument("controller output event is invalid");
  }
  if (event_size_ == events_.size()) {
    return Status::ResourceExhausted("controller output mailbox is full");
  }
  if (next_event_sequence_ == 0) {
    return Status::FailedPrecondition("controller event sequence wrapped");
  }
  const ControllerOutputEvent event{
      draft.kind, next_event_sequence_, draft.epoch, draft.request_generation,
      draft.token_ordinal, draft.token_id, draft.state_digest,
      draft.plan_sequence, draft.plan_event_index, draft.plan_event_count,
      draft.finish_reason};
  const auto tail = (event_head_ + event_size_) % events_.size();
  events_[tail] = event;
  ++event_size_;
  ++next_event_sequence_;
  return event;
}

Result<std::optional<ControllerOutputEvent>>
ControllerMailbox::try_take_event() {
  std::lock_guard lock(mutex_);
  if (event_size_ == 0) return std::optional<ControllerOutputEvent>{};
  const auto event = events_[event_head_];
  event_head_ = (event_head_ + 1) % events_.size();
  --event_size_;
  return std::optional<ControllerOutputEvent>{event};
}

std::uint32_t ControllerMailbox::command_size() const {
  std::lock_guard lock(mutex_);
  return command_size_;
}

std::uint32_t ControllerMailbox::event_size() const {
  std::lock_guard lock(mutex_);
  return event_size_;
}

std::uint32_t ControllerMailbox::event_available_capacity() const {
  std::lock_guard lock(mutex_);
  return static_cast<std::uint32_t>(events_.size()) - event_size_;
}

}  // namespace pih
