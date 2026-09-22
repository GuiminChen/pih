#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/model/qwen3_bf16_packed_result_layout.h"

namespace pih {

enum class QwenBf16PackedReadbackState : std::uint8_t {
  kPrepared = 1,
  kSubmitted = 2,
  kPoisoned = 3,
};

class QwenBf16PackedReadback final {
 public:
  static constexpr std::size_t kMaximumCopyCount = 7;

  static Result<QwenBf16PackedReadback> Create(
      const QwenBf16PackedResultLayout& layout, std::uint32_t sample_count,
      CudaCopyEndpoint sampled_tokens_device,
      CudaCopyEndpoint device_error_device,
      CudaCopyEndpoint pinned_host_result_arena,
      std::uintptr_t primary_context_identity, DriverStreamHandle stream,
      std::uint64_t completion_event_generation,
      std::uint64_t first_plan_id);
  static Result<QwenBf16PackedReadback> CreateSampling(
      const QwenBf16PackedResultLayout& layout, std::uint32_t sample_count,
      CudaCopyEndpoint sampling_result_device,
      CudaCopyEndpoint device_error_device,
      CudaCopyEndpoint pinned_host_result_arena,
      std::uintptr_t primary_context_identity, DriverStreamHandle stream,
      std::uint64_t completion_event_generation,
      std::uint64_t first_plan_id);

  Status submit(TypedCopyDriver& driver);
  [[nodiscard]] QwenBf16PackedReadbackState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::size_t submitted_copies() const noexcept {
    return submitted_copies_;
  }
  [[nodiscard]] std::size_t size() const noexcept { return copies_.size(); }
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
  explicit QwenBf16PackedReadback(std::vector<CudaTypedCopyPlan> copies)
      : copies_(std::move(copies)) {}

  std::vector<CudaTypedCopyPlan> copies_;
  std::size_t submitted_copies_ = 0;
  QwenBf16PackedReadbackState state_ =
      QwenBf16PackedReadbackState::kPrepared;
};

}  // namespace pih
