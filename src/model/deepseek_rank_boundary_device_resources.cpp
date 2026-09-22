#include "pih/model/deepseek_rank_boundary_device_resources.h"

#include "pih/core/checked_math.h"

namespace pih {

Result<std::unique_ptr<DeepSeekRankBoundaryDeviceResources>>
DeepSeekRankBoundaryDeviceResources::Allocate(
    std::uint32_t rank, std::uint32_t world_size,
    std::uint64_t maximum_pipeline_tokens,
    std::uintptr_t context_identity, Allocator& device_allocator,
    CudaRuntimeResourceDriver& runtime_driver) {
  if (world_size == 0 || world_size > 4 || rank >= world_size ||
      maximum_pipeline_tokens == 0 || context_identity == 0) {
    return Status::InvalidArgument(
        "DeepSeek rank boundary resource identity is invalid");
  }
  auto wire_bytes = checked_mul_u64(maximum_pipeline_tokens,
                                    kWireBytesPerToken);
  if (!wire_bytes.ok()) return wire_bytes.status();
  auto slot_bytes = checked_align_up_u64(*wire_bytes, kAlignment);
  if (!slot_bytes.ok()) return slot_bytes.status();
  auto resources = std::unique_ptr<DeepSeekRankBoundaryDeviceResources>(
      new DeepSeekRankBoundaryDeviceResources(
          *slot_bytes, context_identity, runtime_driver));

  if (rank > 0) {
    resources->incoming_slots_.reserve(kCreditCount);
    resources->incoming_events_.reserve(kCreditCount);
    for (std::uint32_t credit = 0; credit < kCreditCount; ++credit) {
      auto buffer = Buffer::Allocate(device_allocator, *slot_bytes, kAlignment);
      if (!buffer.ok()) return buffer.status();
      resources->incoming_slots_.push_back(std::move(*buffer));
      auto event = runtime_driver.create_disable_timing_event(context_identity);
      if (!event.ok()) return event.status();
      resources->incoming_events_.push_back(*event);
    }
  }
  if (rank + 1 < world_size) {
    auto warmup_buffer = Buffer::Allocate(device_allocator, *slot_bytes,
                                          kAlignment);
    if (!warmup_buffer.ok()) return warmup_buffer.status();
    resources->outgoing_warmup_buffer_ =
        std::make_unique<Buffer>(std::move(*warmup_buffer));
    resources->outgoing_events_.reserve(kCreditCount);
    for (std::uint32_t credit = 0; credit < kCreditCount; ++credit) {
      auto event = runtime_driver.create_disable_timing_event(context_identity);
      if (!event.ok()) return event.status();
      resources->outgoing_events_.push_back(*event);
    }
  }
  return resources;
}

DeepSeekRankBoundaryDeviceResources::~DeepSeekRankBoundaryDeviceResources() {
  if (runtime_driver_ == nullptr) return;
  for (auto event : outgoing_events_) runtime_driver_->destroy_event(event);
  for (auto event : incoming_events_) runtime_driver_->destroy_event(event);
}

}  // namespace pih
