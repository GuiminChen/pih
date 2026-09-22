#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

enum class QwenKvSlotLifecycle : std::uint8_t {
  kUninitialized = 0,
  kFreeClean = 1,
  kAdmissionReserved = 2,
  kOwned = 3,
  kReclaimPending = 4,
  kFreeDirty = 5,
  kScrubbing = 6,
};

struct QwenKvBlockHandle final {
  std::uint32_t slot;
  std::uint32_t generation;
  friend bool operator==(const QwenKvBlockHandle&,
                         const QwenKvBlockHandle&) = default;
};

struct QwenKvSlotState final {
  std::uint32_t generation;
  std::uint32_t owner_sequence_index;
  std::uint16_t valid_tokens;
  QwenKvSlotLifecycle state;
  std::uint8_t reserved_zero_u8;
  std::uint32_t reserved_zero_u32;
};

struct QwenKvCompletionEvent final {
  std::uint64_t handle;
  std::uint64_t generation;
};

struct QwenKvScrubWork final {
  QwenKvBlockHandle slot;
  std::uint64_t bytes;
};

static_assert(sizeof(QwenKvBlockHandle) == 8);
static_assert(sizeof(QwenKvSlotState) == 16);

class QwenKvSlotPool final {
 public:
  static constexpr std::uint32_t kTokensPerSlot = 16;
  static constexpr std::uint64_t kBytesPerLayerK = 32 * 1024;
  static constexpr std::uint64_t kBytesPerLayerV = 32 * 1024;
  static constexpr std::uint32_t kLayerCount = 28;
  static constexpr std::uint64_t kSlotPayloadBytes = 1'835'008;
  static constexpr std::uint32_t kNoOwner = UINT32_MAX;
  static constexpr std::uint32_t kMaximumSlots = 4'681;

  static Result<QwenKvSlotPool> Create(std::uint32_t slot_count,
                                       std::uint64_t kv_backing_bytes,
                                       std::uint64_t metadata_backing_bytes);

  Status complete_startup_sanitize(std::uint64_t cleared_kv_bytes,
                                   std::uint64_t cleared_metadata_bytes,
                                   bool completion_event_succeeded);
  Result<std::uint32_t> acquire_sequence_generation();
  Result<std::vector<QwenKvBlockHandle>> reserve(
      std::uint32_t owner_sequence_index, std::uint32_t slot_count);
  Status publish(std::uint32_t owner_sequence_index,
                 std::span<const QwenKvBlockHandle> handles);
  Status rollback(std::uint32_t owner_sequence_index,
                  std::span<const QwenKvBlockHandle> handles);
  Status commit_valid_tokens(std::uint32_t owner_sequence_index,
                             QwenKvBlockHandle handle,
                             std::uint16_t valid_tokens);
  Status commit_prefix(std::uint32_t owner_sequence_index,
                       std::span<const QwenKvBlockHandle> handles,
                       std::uint32_t committed_tokens);
  Result<std::vector<QwenKvSlotState>> project_prefix(
      std::uint32_t owner_sequence_index,
      std::span<const QwenKvBlockHandle> handles,
      std::uint32_t committed_tokens) const;
  Status release(std::uint32_t owner_sequence_index,
                 std::span<const QwenKvBlockHandle> handles,
                 QwenKvCompletionEvent last_use_event);
  Status complete_reclaim_event(QwenKvBlockHandle handle,
                                QwenKvCompletionEvent event,
                                bool event_ready, bool event_succeeded);
  Result<QwenKvScrubWork> begin_next_scrub(
      QwenKvCompletionEvent scrub_completion_event);
  Status complete_scrub(QwenKvBlockHandle handle,
                        QwenKvCompletionEvent event,
                        std::uint64_t cleared_bytes,
                        bool event_ready, bool event_succeeded);

  [[nodiscard]] bool ready() const noexcept { return ready_; }
  [[nodiscard]] bool failed() const noexcept { return failed_; }
  [[nodiscard]] std::uint32_t slot_count() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
  }
  [[nodiscard]] std::uint32_t clean_credits() const noexcept {
    return clean_credits_;
  }
  [[nodiscard]] std::uint32_t lifecycle_count(
      QwenKvSlotLifecycle state) const noexcept;
  [[nodiscard]] const QwenKvSlotState& slot(std::uint32_t ordinal) const {
    return slots_.at(ordinal);
  }

 private:
  QwenKvSlotPool(std::vector<QwenKvSlotState> slots,
                 std::vector<QwenKvCompletionEvent> pending_events,
                 std::uint64_t kv_backing_bytes,
                 std::uint64_t metadata_backing_bytes)
      : slots_(std::move(slots)),
        pending_events_(std::move(pending_events)),
        kv_backing_bytes_(kv_backing_bytes),
        metadata_backing_bytes_(metadata_backing_bytes) {}

  Status validate_handle(const QwenKvBlockHandle& handle,
                         std::uint32_t owner,
                         QwenKvSlotLifecycle expected) const;
  Status validate_exact_owner_set(
      std::uint32_t owner, std::span<const QwenKvBlockHandle> handles,
      QwenKvSlotLifecycle expected) const;

  std::vector<QwenKvSlotState> slots_;
  std::vector<QwenKvCompletionEvent> pending_events_;
  std::uint64_t kv_backing_bytes_ = 0;
  std::uint64_t metadata_backing_bytes_ = 0;
  std::uint32_t clean_credits_ = 0;
  std::uint32_t next_sequence_generation_ = 1;
  bool ready_ = false;
  bool failed_ = false;
};

}  // namespace pih
