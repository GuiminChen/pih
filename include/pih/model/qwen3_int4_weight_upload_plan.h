#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/model/qwen3_int4_artifact_layout.h"

namespace pih {

enum class QwenInt4WeightUploadState : std::uint8_t {
  kPrepared,
  kSubmitted,
  kPoisoned,
};

class QwenInt4WeightUploadPlan final {
 public:
  static Result<QwenInt4WeightUploadPlan> Create(
      const QwenInt4ArtifactLayout& layout,
      CudaCopyEndpoint canonical_file,
      CudaCopyEndpoint compact_device_pool,
      std::uintptr_t primary_context_identity,
      DriverStreamHandle stream,
      std::uint64_t completion_event_generation,
      std::uint64_t first_plan_id);

  Status submit(TypedCopyDriver& driver);
  [[nodiscard]] std::size_t copy_count() const noexcept {
    return copies_.size();
  }
  [[nodiscard]] std::uint64_t logical_bytes() const noexcept {
    return logical_bytes_;
  }
  [[nodiscard]] const CudaTypedCopyPlan& copy(std::size_t index) const {
    return copies_.at(index);
  }
  [[nodiscard]] QwenInt4WeightUploadState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::size_t submitted_copies() const noexcept {
    return submitted_copies_;
  }

 private:
  QwenInt4WeightUploadPlan(std::vector<CudaTypedCopyPlan> copies,
                           std::uint64_t logical_bytes)
      : copies_(std::move(copies)), logical_bytes_(logical_bytes) {}

  std::vector<CudaTypedCopyPlan> copies_;
  std::uint64_t logical_bytes_;
  std::size_t submitted_copies_ = 0;
  QwenInt4WeightUploadState state_ = QwenInt4WeightUploadState::kPrepared;
};

}  // namespace pih
