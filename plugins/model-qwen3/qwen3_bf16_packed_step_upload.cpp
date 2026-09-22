#include "pih/model/qwen3_bf16_packed_step_upload.h"

#include <array>
#include <limits>

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenBf16PackedStepUpload> QwenBf16PackedStepUpload::Create(
    const QwenBf16PackedStepStagingLayout& layout,
    CudaCopyEndpoint pinned_host_arena, CudaCopyEndpoint device_arena,
    std::uintptr_t primary_context_identity, DriverStreamHandle stream,
    std::uint64_t completion_event_generation,
    std::uint64_t first_plan_id) {
  if (first_plan_id == 0 ||
      first_plan_id >
          std::numeric_limits<std::uint64_t>::max() - (kCopyCount - 1)) {
    return Status::InvalidArgument(
        "packed step upload plan identity is invalid");
  }
  auto host_end =
      checked_add_u64(pinned_host_arena.offset, layout.total_bytes());
  auto device_end =
      checked_add_u64(device_arena.offset, layout.total_bytes());
  if (!host_end.ok()) return host_end.status();
  if (!device_end.ok()) return device_end.status();
  if (*host_end > pinned_host_arena.allocation_bytes ||
      *device_end > device_arena.allocation_bytes) {
    return Status::InvalidArgument(
        "packed step upload arena range exceeds its owner");
  }
  constexpr std::array<std::uint64_t, kCopyCount> kAlignments{
      4, 8, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 8, 4};
  std::vector<CudaTypedCopyPlan> copies;
  copies.reserve(kCopyCount);
  for (std::size_t index = 0; index < kCopyCount; ++index) {
    const auto span = layout.copy_spans()[index];
    auto host_offset =
        checked_add_u64(pinned_host_arena.offset, span.offset_bytes);
    auto device_offset =
        checked_add_u64(device_arena.offset, span.offset_bytes);
    if (!host_offset.ok()) return host_offset.status();
    if (!device_offset.ok()) return device_offset.status();
    auto source = pinned_host_arena;
    auto destination = device_arena;
    source.offset = *host_offset;
    destination.offset = *device_offset;
    auto copy = CudaTypedCopyPlan::Create(
        first_plan_id + index, CudaCopyPurpose::kInput,
        CudaCopyKind::kHostToDevice, source, destination, span.size_bytes,
        kAlignments[index], primary_context_identity, stream,
        completion_event_generation);
    if (!copy.ok()) return copy.status();
    copies.push_back(std::move(*copy));
  }
  return QwenBf16PackedStepUpload(std::move(copies));
}

Status QwenBf16PackedStepUpload::submit(TypedCopyDriver& driver) {
  if (state_ != QwenBf16PackedStepUploadState::kPrepared) {
    return Status::FailedPrecondition(
        "packed step upload is not submit-ready");
  }
  for (; submitted_copies_ < copies_.size(); ++submitted_copies_) {
    const auto status = copies_[submitted_copies_].submit(driver);
    if (!status.ok()) {
      state_ = QwenBf16PackedStepUploadState::kPoisoned;
      return status;
    }
  }
  state_ = QwenBf16PackedStepUploadState::kSubmitted;
  return Status::Ok();
}

}  // namespace pih
