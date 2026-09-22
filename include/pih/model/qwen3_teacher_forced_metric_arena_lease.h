#pragma once

#include <atomic>
#include <cstdint>
#include <utility>

#include "pih/core/result.h"

namespace pih {

struct QwenTeacherForcedMetricArenaLeaseState final {
  explicit QwenTeacherForcedMetricArenaLeaseState(
      std::uint64_t identity) : arena_identity(identity) {}
  const std::uint64_t arena_identity;
  std::atomic<bool> claimed{false};
  std::atomic<std::uint64_t> last_factory_generation{0};
};

// A prepared transaction may release its arena claim on destruction. Once any
// asynchronous GPU work is attempted, only authorized completion may release
// the claim; failure or abandonment therefore leaves the arena fail-stopped.
class QwenTeacherForcedMetricArenaLease final {
 public:
  QwenTeacherForcedMetricArenaLease() = default;
  static Result<QwenTeacherForcedMetricArenaLease> Acquire(
      QwenTeacherForcedMetricArenaLeaseState& state) {
    bool expected = false;
    if (!state.claimed.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel,
            std::memory_order_acquire))
      return Status::FailedPrecondition(
          "Qwen teacher-forced metric arena is already leased");
    return QwenTeacherForcedMetricArenaLease(&state);
  }

  QwenTeacherForcedMetricArenaLease(
      const QwenTeacherForcedMetricArenaLease&) = delete;
  QwenTeacherForcedMetricArenaLease& operator=(
      const QwenTeacherForcedMetricArenaLease&) = delete;
  QwenTeacherForcedMetricArenaLease(
      QwenTeacherForcedMetricArenaLease&& other) noexcept {
    move_from(std::move(other));
  }
  QwenTeacherForcedMetricArenaLease& operator=(
      QwenTeacherForcedMetricArenaLease&& other) noexcept {
    if (this != &other) {
      release_if_prepared();
      move_from(std::move(other));
    }
    return *this;
  }
  ~QwenTeacherForcedMetricArenaLease() { release_if_prepared(); }

  Status mark_submitted() {
    if (state_ == nullptr) return Status::Ok();
    if (!release_if_prepared_)
      return Status::FailedPrecondition(
          "Qwen teacher-forced arena lease is already submitted");
    release_if_prepared_ = false;
    return Status::Ok();
  }

  Status release_completed() {
    if (state_ == nullptr) return Status::Ok();
    if (release_if_prepared_)
      return Status::FailedPrecondition(
          "Qwen teacher-forced arena lease was not submitted");
    if (!state_->claimed.exchange(false, std::memory_order_acq_rel))
      return Status::FailedPrecondition(
          "Qwen teacher-forced arena lease is not claimed");
    state_ = nullptr;
    return Status::Ok();
  }

 private:
  explicit QwenTeacherForcedMetricArenaLease(
      QwenTeacherForcedMetricArenaLeaseState* state) : state_(state) {}
  void release_if_prepared() noexcept {
    if (state_ != nullptr && release_if_prepared_)
      state_->claimed.store(false, std::memory_order_release);
    state_ = nullptr;
  }
  void move_from(QwenTeacherForcedMetricArenaLease&& other) noexcept {
    state_ = other.state_;
    release_if_prepared_ = other.release_if_prepared_;
    other.state_ = nullptr;
  }

  QwenTeacherForcedMetricArenaLeaseState* state_ = nullptr;
  bool release_if_prepared_ = true;
};

}  // namespace pih
