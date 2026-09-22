#include "pih/model/qwen3_bf16_packed_readback.h"

#include <array>

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenBf16PackedReadback> QwenBf16PackedReadback::Create(
    const QwenBf16PackedResultLayout& layout, std::uint32_t sample_count,
    CudaCopyEndpoint sampled_tokens_device,
    CudaCopyEndpoint device_error_device,
    CudaCopyEndpoint pinned_host_result_arena,
    std::uintptr_t primary_context_identity, DriverStreamHandle stream,
    std::uint64_t completion_event_generation, std::uint64_t first_plan_id) {
  if (sample_count > layout.sample_capacity() ||
      first_plan_id == 0 || first_plan_id == UINT64_MAX) {
    return Status::InvalidArgument("packed readback identity or count is invalid");
  }
  auto host_end = checked_add_u64(pinned_host_result_arena.offset,
                                  layout.total_bytes());
  if (!host_end.ok()) return host_end.status();
  if (*host_end > pinned_host_result_arena.allocation_bytes) {
    return Status::InvalidArgument("packed readback exceeds host result owner");
  }
  const QwenBf16ArenaSpan active_tokens{
      layout.sampled_token_ids().offset_bytes,
      static_cast<std::uint64_t>(sample_count) * sizeof(std::uint32_t)};
  constexpr std::size_t kLegacyCopyCount = 2;
  const std::array<CudaCopyEndpoint, kLegacyCopyCount> sources{
      sampled_tokens_device, device_error_device};
  const std::array<QwenBf16ArenaSpan, kLegacyCopyCount> spans{
      active_tokens, layout.device_error()};
  const std::array<std::uint64_t, kLegacyCopyCount> alignments{4, 4};
  std::vector<CudaTypedCopyPlan> copies;
  copies.reserve(sample_count == 0 ? 1 : kLegacyCopyCount);
  const std::size_t first_copy = sample_count == 0 ? 1 : 0;
  for (std::size_t i = first_copy; i < kLegacyCopyCount; ++i) {
    auto destination = pinned_host_result_arena;
    auto offset = checked_add_u64(destination.offset, spans[i].offset_bytes);
    if (!offset.ok()) return offset.status();
    destination.offset = *offset;
    auto copy = CudaTypedCopyPlan::Create(
        first_plan_id + (i - first_copy), CudaCopyPurpose::kResult,
        CudaCopyKind::kDeviceToHost, sources[i], destination,
        spans[i].size_bytes, alignments[i], primary_context_identity, stream,
        completion_event_generation);
    if (!copy.ok()) return copy.status();
    copies.push_back(std::move(*copy));
  }
  return QwenBf16PackedReadback(std::move(copies));
}

Result<QwenBf16PackedReadback> QwenBf16PackedReadback::CreateSampling(
    const QwenBf16PackedResultLayout& layout, std::uint32_t sample_count,
    CudaCopyEndpoint sampling_result_device,
    CudaCopyEndpoint device_error_device,
    CudaCopyEndpoint pinned_host_result_arena,
    std::uintptr_t primary_context_identity, DriverStreamHandle stream,
    std::uint64_t completion_event_generation, std::uint64_t first_plan_id) {
  if (sample_count > layout.sample_capacity() || first_plan_id == 0 ||
      first_plan_id > UINT64_MAX - (kMaximumCopyCount - 1U)) {
    return Status::InvalidArgument(
        "packed sampling readback identity or count is invalid");
  }
  auto host_end = checked_add_u64(pinned_host_result_arena.offset,
                                  layout.total_bytes());
  auto device_end = checked_add_u64(sampling_result_device.offset,
                                    layout.device_error().offset_bytes);
  if (!host_end.ok()) return host_end.status();
  if (!device_end.ok()) return device_end.status();
  if (*host_end > pinned_host_result_arena.allocation_bytes ||
      *device_end > sampling_result_device.allocation_bytes) {
    return Status::InvalidArgument(
        "packed sampling readback exceeds result owner");
  }
  const auto scalar_bytes =
      static_cast<std::uint64_t>(sample_count) * sizeof(std::uint32_t);
  const auto top_bytes = scalar_bytes * 20U;
  const std::array<QwenBf16ArenaSpan, 6> active_spans{
      QwenBf16ArenaSpan{layout.sampled_token_ids().offset_bytes, scalar_bytes},
      QwenBf16ArenaSpan{layout.selected_logprobs().offset_bytes, scalar_bytes},
      QwenBf16ArenaSpan{layout.rng_words().offset_bytes, scalar_bytes},
      QwenBf16ArenaSpan{layout.top_logprob_token_ids().offset_bytes, top_bytes},
      QwenBf16ArenaSpan{layout.top_logprobs().offset_bytes, top_bytes},
      QwenBf16ArenaSpan{layout.top_logprob_counts().offset_bytes, scalar_bytes}};
  std::vector<CudaTypedCopyPlan> copies;
  copies.reserve(sample_count == 0 ? 1U : kMaximumCopyCount);
  if (sample_count != 0) {
    for (std::size_t i = 0; i < active_spans.size(); ++i) {
      auto source = sampling_result_device;
      auto destination = pinned_host_result_arena;
      auto source_offset = checked_add_u64(source.offset,
                                           active_spans[i].offset_bytes);
      auto destination_offset = checked_add_u64(
          destination.offset, active_spans[i].offset_bytes);
      if (!source_offset.ok()) return source_offset.status();
      if (!destination_offset.ok()) return destination_offset.status();
      source.offset = *source_offset;
      destination.offset = *destination_offset;
      auto copy = CudaTypedCopyPlan::Create(
          first_plan_id + i, CudaCopyPurpose::kResult,
          CudaCopyKind::kDeviceToHost, source, destination,
          active_spans[i].size_bytes, 4, primary_context_identity, stream,
          completion_event_generation);
      if (!copy.ok()) return copy.status();
      copies.push_back(std::move(*copy));
    }
  }
  auto error_destination = pinned_host_result_arena;
  auto error_offset = checked_add_u64(error_destination.offset,
                                      layout.device_error().offset_bytes);
  if (!error_offset.ok()) return error_offset.status();
  error_destination.offset = *error_offset;
  auto error_copy = CudaTypedCopyPlan::Create(
      first_plan_id + copies.size(), CudaCopyPurpose::kResult,
      CudaCopyKind::kDeviceToHost, device_error_device, error_destination,
      sizeof(std::uint32_t), 4, primary_context_identity, stream,
      completion_event_generation);
  if (!error_copy.ok()) return error_copy.status();
  copies.push_back(std::move(*error_copy));
  return QwenBf16PackedReadback(std::move(copies));
}

Status QwenBf16PackedReadback::submit(TypedCopyDriver& driver) {
  if (state_ != QwenBf16PackedReadbackState::kPrepared) {
    return Status::FailedPrecondition("packed readback is not submit-ready");
  }
  for (; submitted_copies_ < copies_.size(); ++submitted_copies_) {
    auto status = copies_[submitted_copies_].submit(driver);
    if (!status.ok()) {
      state_ = QwenBf16PackedReadbackState::kPoisoned;
      return status;
    }
  }
  state_ = QwenBf16PackedReadbackState::kSubmitted;
  return Status::Ok();
}

}  // namespace pih
