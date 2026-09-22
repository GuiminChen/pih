#pragma once

#include <memory>
#include <vector>

#include "pih/backend/cuda/cuda_runtime_resources.h"
#include "pih/core/buffer.h"

namespace pih {

class DeepSeekRankBoundaryDeviceResources final {
 public:
  static constexpr std::uint32_t kCreditCount = 2;
  static constexpr std::uint64_t kWireBytesPerToken = 32768;
  static constexpr std::uint64_t kAlignment = 256;

  static Result<std::unique_ptr<DeepSeekRankBoundaryDeviceResources>> Allocate(
      std::uint32_t rank, std::uint32_t world_size,
      std::uint64_t maximum_pipeline_tokens,
      std::uintptr_t context_identity, Allocator& device_allocator,
      CudaRuntimeResourceDriver& runtime_driver);

  DeepSeekRankBoundaryDeviceResources(
      const DeepSeekRankBoundaryDeviceResources&) = delete;
  DeepSeekRankBoundaryDeviceResources& operator=(
      const DeepSeekRankBoundaryDeviceResources&) = delete;
  ~DeepSeekRankBoundaryDeviceResources();

  [[nodiscard]] std::uint64_t slot_bytes() const noexcept {
    return slot_bytes_;
  }
  [[nodiscard]] std::size_t incoming_slot_count() const noexcept {
    return incoming_slots_.size();
  }
  [[nodiscard]] std::size_t incoming_event_count() const noexcept {
    return incoming_events_.size();
  }
  [[nodiscard]] std::size_t outgoing_event_count() const noexcept {
    return outgoing_events_.size();
  }
  [[nodiscard]] Buffer* incoming_slot(std::size_t index) noexcept {
    return index < incoming_slots_.size() ? &incoming_slots_[index] : nullptr;
  }
  [[nodiscard]] DriverEventHandle incoming_event(std::size_t index) const
      noexcept {
    return index < incoming_events_.size() ? incoming_events_[index] : 0;
  }
  [[nodiscard]] DriverEventHandle outgoing_event(std::size_t index) const
      noexcept {
    return index < outgoing_events_.size() ? outgoing_events_[index] : 0;
  }
  [[nodiscard]] Buffer* outgoing_warmup_buffer() noexcept {
    return outgoing_warmup_buffer_.get();
  }

 private:
  DeepSeekRankBoundaryDeviceResources(
      std::uint64_t slot_bytes, std::uintptr_t context_identity,
      CudaRuntimeResourceDriver& runtime_driver) noexcept
      : slot_bytes_(slot_bytes), context_identity_(context_identity),
        runtime_driver_(&runtime_driver) {}

  std::uint64_t slot_bytes_ = 0;
  std::uintptr_t context_identity_ = 0;
  CudaRuntimeResourceDriver* runtime_driver_ = nullptr;
  std::vector<Buffer> incoming_slots_;
  std::unique_ptr<Buffer> outgoing_warmup_buffer_;
  std::vector<DriverEventHandle> incoming_events_;
  std::vector<DriverEventHandle> outgoing_events_;
};

}  // namespace pih
