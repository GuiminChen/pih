#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/model/qwen3_bf16_step_staging_layout.h"

namespace pih {

enum class QwenBf16StepUploadState : std::uint8_t {
  kPrepared,
  kSubmitted,
  kPoisoned,
};

class QwenBf16StepUpload final {
 public:
  static constexpr std::size_t kCopyCount = 5;

  static Result<QwenBf16StepUpload> Create(
      const QwenBf16StepStagingLayout& layout,
      CudaCopyEndpoint pinned_host_arena, CudaCopyEndpoint device_arena,
      std::uintptr_t primary_context_identity, DriverStreamHandle stream,
      std::uint64_t completion_event_generation,
      std::uint64_t first_plan_id);

  Status submit(TypedCopyDriver& driver);

  [[nodiscard]] QwenBf16StepUploadState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::size_t submitted_copies() const noexcept {
    return submitted_copies_;
  }
  [[nodiscard]] const CudaTypedCopyPlan& copy(std::size_t index) const {
    return copies_.at(index);
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

 private:
  explicit QwenBf16StepUpload(std::vector<CudaTypedCopyPlan> copies)
      : copies_(std::move(copies)) {}

  std::vector<CudaTypedCopyPlan> copies_;
  std::size_t submitted_copies_ = 0;
  QwenBf16StepUploadState state_ = QwenBf16StepUploadState::kPrepared;
};

}  // namespace pih
