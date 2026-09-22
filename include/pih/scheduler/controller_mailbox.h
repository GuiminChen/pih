#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

enum class ControllerCommandKind : std::uint8_t {
  kAdmit = 1,
  kCancel = 2,
  kShutdown = 3,
};

struct ControllerCommand final {
  ControllerCommandKind kind;
  std::uint64_t command_sequence;
  std::uint64_t epoch;
  std::uint64_t request_generation;
  Sha256Digest payload_digest;
};

enum class ControllerOutputEventKind : std::uint8_t {
  kAdmitted = 1,
  kTokenCommitted = 2,
  kDraining = 3,
  kCompleted = 4,
  kCancelled = 5,
  kFailed = 6,
};

enum class ControllerFinishReason : std::uint8_t {
  kNone = 0,
  kStop,
  kLength,
};

struct ControllerOutputEventDraft final {
  ControllerOutputEventKind kind;
  std::uint64_t epoch;
  std::uint64_t request_generation;
  std::uint64_t token_ordinal;
  std::uint32_t token_id;
  Sha256Digest state_digest;
  std::uint64_t plan_sequence = 0;
  std::uint32_t plan_event_index = 0;
  std::uint32_t plan_event_count = 0;
  ControllerFinishReason finish_reason = ControllerFinishReason::kNone;
};

struct ControllerOutputEvent final {
  ControllerOutputEventKind kind;
  std::uint64_t event_sequence;
  std::uint64_t epoch;
  std::uint64_t request_generation;
  std::uint64_t token_ordinal;
  std::uint32_t token_id;
  Sha256Digest state_digest;
  std::uint64_t plan_sequence = 0;
  std::uint32_t plan_event_index = 0;
  std::uint32_t plan_event_count = 0;
  ControllerFinishReason finish_reason = ControllerFinishReason::kNone;
};

struct ControllerMailboxLimits final {
  std::uint32_t command_capacity;
  std::uint32_t event_capacity;
};

class ControllerMailbox final {
 public:
  static constexpr std::string_view kAbi = "bounded_controller_mailbox_v1";

  static Result<ControllerMailbox> Create(ControllerMailboxLimits limits);
  ControllerMailbox(const ControllerMailbox&) = delete;
  ControllerMailbox& operator=(const ControllerMailbox&) = delete;
  ControllerMailbox(ControllerMailbox&& other) noexcept;
  ControllerMailbox& operator=(ControllerMailbox&&) = delete;

  Result<ControllerCommand> submit_command(ControllerCommandKind kind,
                                           std::uint64_t epoch,
                                           std::uint64_t request_generation,
                                           Sha256Digest payload_digest);
  Result<std::optional<ControllerCommand>> try_take_command();
  Result<ControllerOutputEvent> publish_event(
      const ControllerOutputEventDraft& draft);
  Result<std::optional<ControllerOutputEvent>> try_take_event();

  [[nodiscard]] std::uint32_t command_size() const;
  [[nodiscard]] std::uint32_t event_size() const;
  [[nodiscard]] std::uint32_t event_available_capacity() const;

 private:
  explicit ControllerMailbox(ControllerMailboxLimits limits);
  Status require_controller_thread_locked();

  mutable std::mutex mutex_;
  std::vector<ControllerCommand> commands_;
  std::vector<ControllerOutputEvent> events_;
  std::uint32_t command_head_ = 0;
  std::uint32_t command_size_ = 0;
  std::uint32_t event_head_ = 0;
  std::uint32_t event_size_ = 0;
  std::uint64_t next_command_sequence_ = 1;
  std::uint64_t next_event_sequence_ = 1;
  std::optional<std::thread::id> controller_thread_;
};

}  // namespace pih
