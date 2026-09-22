#include "pih/model/qwen3_bf16_packed_staging_resources.h"

#include <array>
#include <limits>

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenBf16PackedStagingResources>
QwenBf16PackedStagingResources::Create(
    std::uint64_t request_generation, std::int32_t owning_rank,
    const QwenBf16PackedStepStagingLayout& layout,
    QwenBf16DeviceArenaOwner owner) {
  if (request_generation == 0 || owning_rank < 0 || owner.base == 0 ||
      owner.bytes < layout.total_bytes() ||
      owner.generation != request_generation) {
    return Status::InvalidArgument(
        "packed staging resource owner identity is invalid");
  }
  auto device = Device::Create(DeviceType::kCuda, owning_rank);
  if (!device.ok()) return device.status();
  QwenBf16PackedStagingResources result;
  result.request_generation_ = request_generation;
  result.owning_rank_ = owning_rank;
  for (std::size_t i = 0; i < kSlotCount; ++i) {
    const auto span = layout.copy_spans()[i];
    if (span.size_bytes == 0 ||
        span.size_bytes > static_cast<std::uint64_t>(
                              std::numeric_limits<std::int64_t>::max())) {
      return Status::InvalidArgument("packed staging resource span is invalid");
    }
    auto end = checked_add_u64(span.offset_bytes, span.size_bytes);
    auto address = checked_add_u64(owner.base, span.offset_bytes);
    if (!end.ok()) return end.status();
    if (!address.ok()) return address.status();
    if (*end > owner.bytes) {
      return Status::InvalidArgument(
          "packed staging resource exceeds its owner");
    }
    const std::array<std::int64_t, 1> shape{
        static_cast<std::int64_t>(span.size_bytes)};
    auto view = TensorView::Create(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(*address)),
        DType::kUInt8, shape, {}, *device, owner.generation);
    if (!view.ok()) return view.status();
    result.resources_[i].emplace(std::move(*view));
  }
  return result;
}

Result<TensorView> QwenBf16PackedStagingResources::view(
    QwenBf16PackedStagingSlot slot) const {
  const auto index = static_cast<std::size_t>(slot);
  if (index >= resources_.size() || !resources_[index].has_value()) {
    return Status::InvalidArgument("packed staging resource slot is invalid");
  }
  return *resources_[index];
}

}  // namespace pih
