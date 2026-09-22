#pragma once

#include <cstdint>

#include "pih/core/result.h"

namespace pih {

enum class DeepSeekRequestState : std::uint8_t {
  kSubmitted,
  kAdmitted,
  kPrepared,
  kCommitted,
  kDraining,
  kCompleted,
  kCancelled,
  kFailed,
};

class DeepSeekRequestLifecycle final {
 public:
  static Result<DeepSeekRequestLifecycle> Create(
      std::uint64_t request_id, std::uint64_t request_generation);

  Status admit();
  [[nodiscard]] Status validate_prepare(std::uint64_t plan_sequence) const;
  Status prepare(std::uint64_t plan_sequence);
  [[nodiscard]] Status validate_commit(std::uint64_t plan_sequence) const;
  Status commit(std::uint64_t plan_sequence);
  [[nodiscard]] Status validate_abort_prepare(
      std::uint64_t plan_sequence) const;
  Status abort_prepare(std::uint64_t plan_sequence);
  [[nodiscard]] Status validate_cancel(
      std::uint64_t request_generation) const;
  Status cancel(std::uint64_t request_generation);
  [[nodiscard]] Status validate_backend_complete(
      std::uint64_t plan_sequence) const;
  Status backend_complete(std::uint64_t plan_sequence,
                          bool terminal = true);
  Status finish();
  Status fail();

  [[nodiscard]] DeepSeekRequestState state() const noexcept { return state_; }
  [[nodiscard]] bool user_terminal() const noexcept { return user_terminal_; }
  [[nodiscard]] bool backend_drained() const noexcept { return backend_drained_; }
  [[nodiscard]] bool suppress_output() const noexcept { return suppress_output_; }
  [[nodiscard]] std::uint64_t request_id() const noexcept { return request_id_; }
  [[nodiscard]] std::uint64_t request_generation() const noexcept {
    return request_generation_;
  }

 private:
  DeepSeekRequestLifecycle(std::uint64_t request_id,
                           std::uint64_t request_generation)
      : request_id_(request_id), request_generation_(request_generation) {}
  [[nodiscard]] bool terminal() const noexcept;

  std::uint64_t request_id_ = 0;
  std::uint64_t request_generation_ = 0;
  std::uint64_t plan_sequence_ = 0;
  DeepSeekRequestState state_ = DeepSeekRequestState::kSubmitted;
  bool user_terminal_ = false;
  bool backend_drained_ = false;
  bool suppress_output_ = false;
  bool failure_pending_ = false;
};

}  // namespace pih
