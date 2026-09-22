#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/model/qwen3_bf16_packed_step_staging_layout.h"

namespace pih {

enum class QwenBf16PackedStepUploadState : std::uint8_t {
  kPrepared = 1,
  kSubmitted = 2,
  kPoisoned = 3,
};

class QwenBf16PackedStepUpload final {
 public:
  static constexpr std::size_t kCopyCount =
      QwenBf16PackedStepStagingLayout::kCopySpanCount;

  static Result<QwenBf16PackedStepUpload> Create(
      const QwenBf16PackedStepStagingLayout& layout,
      CudaCopyEndpoint pinned_host_arena, CudaCopyEndpoint device_arena,
      std::uintptr_t primary_context_identity, DriverStreamHandle stream,
      std::uint64_t completion_event_generation,
      std::uint64_t first_plan_id);

  Status submit(TypedCopyDriver& driver);

  [[nodiscard]] QwenBf16PackedStepUploadState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::size_t submitted_copies() const noexcept {
    return submitted_copies_;
  }
  [[nodiscard]] std::uintptr_t context_identity() const noexcept {
    return copies_.front().context_identity();
  }
  [[nodiscard]] DriverStreamHandle stream() const noexcept {
    return copies_.front().stream();
  }
  [[nodiscard]] std::uint64_t event_generation() const noexcept {
    return copies_.front().completion_event_generation();
  }
  [[nodiscard]] const CudaTypedCopyPlan& copy(std::size_t index) const {
    return copies_.at(index);
  }

 private:
  explicit QwenBf16PackedStepUpload(std::vector<CudaTypedCopyPlan> copies)
      : copies_(std::move(copies)) {}

  std::vector<CudaTypedCopyPlan> copies_;
  std::size_t submitted_copies_ = 0;
  QwenBf16PackedStepUploadState state_ =
      QwenBf16PackedStepUploadState::kPrepared;
};

}  // namespace pih
