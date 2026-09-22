#include "pih/model/qwen3_teacher_forced_metric_arenas.h"

#include <array>
#include <atomic>
#include <limits>
#include <memory>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Result<std::uint64_t> reserve_metric_arena_identity() {
  static std::atomic<std::uint64_t> last_identity{0};
  auto current = last_identity.load(std::memory_order_acquire);
  while (true) {
    if (current == std::numeric_limits<std::uint64_t>::max())
      return Status::ResourceExhausted("Qwen metric arena identity exhausted");
    if (last_identity.compare_exchange_weak(
            current, current + 1, std::memory_order_acq_rel,
            std::memory_order_acquire))
      return current + 1;
  }
}

bool valid(const Buffer& buffer, std::uint64_t bytes, DeviceType type,
           std::int32_t index) {
  return buffer.data() != nullptr && buffer.size_bytes() == bytes &&
         buffer.generation() != 0 && buffer.device().type() == type &&
         buffer.device().index() == index &&
         reinterpret_cast<std::uintptr_t>(buffer.data()) % 256 == 0;
}

}  // namespace

Result<QwenTeacherForcedMetricArenas>
QwenTeacherForcedMetricArenas::AllocateVerified(
    std::uint32_t row_capacity, Allocator& device_allocator,
    RegisteredPinnedAllocator& pinned_allocator,
    PinnedPlacementVerifier& verifier, std::int32_t owning_rank,
    std::int32_t numa_node) {
  if (owning_rank < 0 || numa_node < 0)
    return Status::InvalidArgument("Qwen metric arena placement is invalid");
  auto layout = QwenTeacherForcedMetricResultLayout::Create(row_capacity);
  if (!layout.ok()) return layout.status();
  auto target_bytes = checked_mul_u64(row_capacity, sizeof(std::uint32_t));
  if (!target_bytes.ok()) return target_bytes.status();
  auto device_sample_rows =
      Buffer::Allocate(device_allocator, *target_bytes, 256);
  if (!device_sample_rows.ok()) return device_sample_rows.status();
  auto device_targets = Buffer::Allocate(device_allocator, *target_bytes, 256);
  if (!device_targets.ok()) return device_targets.status();
  auto device_result =
      Buffer::Allocate(device_allocator, layout->total_bytes(), 256);
  if (!device_result.ok()) return device_result.status();
  auto pinned_sample_rows =
      Buffer::Allocate(pinned_allocator, *target_bytes, 256);
  if (!pinned_sample_rows.ok()) return pinned_sample_rows.status();
  auto pinned_targets = Buffer::Allocate(pinned_allocator, *target_bytes, 256);
  if (!pinned_targets.ok()) return pinned_targets.status();
  auto pinned_result =
      Buffer::Allocate(pinned_allocator, layout->total_bytes(), 256);
  if (!pinned_result.ok()) return pinned_result.status();
  if (!valid(*device_sample_rows, *target_bytes, DeviceType::kCuda,
             owning_rank) ||
      !valid(*device_targets, *target_bytes, DeviceType::kCuda, owning_rank) ||
      !valid(*device_result, layout->total_bytes(), DeviceType::kCuda,
             owning_rank) ||
      !valid(*pinned_sample_rows, *target_bytes, DeviceType::kCpu, 0) ||
      !valid(*pinned_targets, *target_bytes, DeviceType::kCpu, 0) ||
      !valid(*pinned_result, layout->total_bytes(), DeviceType::kCpu, 0))
    return Status::FailedPrecondition(
        "Qwen metric allocator returned an invalid allocation");
  auto status = verifier.verify(pinned_sample_rows->data(), *target_bytes,
                                numa_node);
  if (status.ok())
    status = verifier.verify(pinned_targets->data(), *target_bytes, numa_node);
  if (status.ok())
    status = verifier.verify(pinned_result->data(), layout->total_bytes(),
                             numa_node);
  if (!status.ok()) return status;
  auto arena_identity = reserve_metric_arena_identity();
  if (!arena_identity.ok()) return arena_identity.status();
  auto lease_state = std::make_unique<QwenTeacherForcedMetricArenaLeaseState>(
      *arena_identity);
  return QwenTeacherForcedMetricArenas(
      *layout, std::move(*device_sample_rows), std::move(*device_targets),
      std::move(*device_result), std::move(*pinned_sample_rows),
      std::move(*pinned_targets), std::move(*pinned_result), owning_rank,
      numa_node, std::move(lease_state));
}

std::span<std::byte>
QwenTeacherForcedMetricArenas::pinned_sample_rows() noexcept {
  return {static_cast<std::byte*>(pinned_sample_rows_.data()),
          static_cast<std::size_t>(pinned_sample_rows_.size_bytes())};
}

std::span<std::byte> QwenTeacherForcedMetricArenas::pinned_targets() noexcept {
  return {static_cast<std::byte*>(pinned_targets_.data()),
          static_cast<std::size_t>(pinned_targets_.size_bytes())};
}

std::span<std::byte> QwenTeacherForcedMetricArenas::pinned_result() noexcept {
  return {static_cast<std::byte*>(pinned_result_.data()),
          static_cast<std::size_t>(pinned_result_.size_bytes())};
}

Result<TensorView> QwenTeacherForcedMetricArenas::device_sample_rows(
    std::uint32_t rows) const {
  return device_sample_rows(rows, device_sample_rows_.generation());
}
Result<TensorView> QwenTeacherForcedMetricArenas::device_sample_rows(
    std::uint32_t rows, std::uint64_t generation) const {
  if (rows == 0 || rows > layout_.row_capacity())
    return Status::InvalidArgument("Qwen metric sample row count is invalid");
  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(rows)};
  return TensorView::Create(device_sample_rows_.data(), DType::kUInt32, shape,
      {}, device_sample_rows_.device(), generation);
}

Result<TensorView> QwenTeacherForcedMetricArenas::device_targets(
    std::uint32_t rows) const {
  return device_targets(rows, device_targets_.generation());
}
Result<TensorView> QwenTeacherForcedMetricArenas::device_targets(
    std::uint32_t rows, std::uint64_t generation) const {
  if (rows == 0 || rows > layout_.row_capacity())
    return Status::InvalidArgument("Qwen metric target row count is invalid");
  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(rows)};
  return TensorView::Create(device_targets_.data(), DType::kUInt32, shape, {},
      device_targets_.device(), generation);
}

Result<TensorView> QwenTeacherForcedMetricArenas::result_view(
    QwenBf16ArenaSpan span, DType dtype, std::uint32_t rows,
    std::uint64_t generation) const {
  if (rows == 0 || rows > layout_.row_capacity())
    return Status::InvalidArgument("Qwen metric result row count is invalid");
  auto bytes = dtype_size(dtype);
  if (!bytes.ok() ||
      static_cast<std::uint64_t>(rows) * *bytes > span.size_bytes)
    return Status::InvalidArgument("Qwen metric result extent is invalid");
  const auto base = reinterpret_cast<std::uintptr_t>(device_result_.data());
  const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(rows)};
  return TensorView::Create(reinterpret_cast<void*>(base + span.offset_bytes),
      dtype, shape, {}, device_result_.device(), generation);
}

Result<TensorView> QwenTeacherForcedMetricArenas::device_argmax(
    std::uint32_t rows) const {
  return device_argmax(rows, device_result_.generation());
}
Result<TensorView> QwenTeacherForcedMetricArenas::device_argmax(
    std::uint32_t rows, std::uint64_t generation) const {
  return result_view(layout_.argmax_tokens(), DType::kUInt32, rows, generation);
}
Result<TensorView> QwenTeacherForcedMetricArenas::device_nll(
    std::uint32_t rows) const {
  return device_nll(rows, device_result_.generation());
}
Result<TensorView> QwenTeacherForcedMetricArenas::device_nll(
    std::uint32_t rows, std::uint64_t generation) const {
  return result_view(layout_.target_nll(), DType::kFloat64, rows, generation);
}
Result<TensorView> QwenTeacherForcedMetricArenas::device_nonfinite(
    std::uint32_t rows) const {
  return device_nonfinite(rows, device_result_.generation());
}
Result<TensorView> QwenTeacherForcedMetricArenas::device_nonfinite(
    std::uint32_t rows, std::uint64_t generation) const {
  return result_view(layout_.nonfinite_rows(), DType::kUInt32, rows, generation);
}
Result<TensorView> QwenTeacherForcedMetricArenas::device_error() const {
  return device_error(device_result_.generation());
}
Result<TensorView> QwenTeacherForcedMetricArenas::device_error(
    std::uint64_t generation) const {
  return result_view(layout_.device_error(), DType::kUInt32, 1, generation);
}

Result<CudaCopyEndpoint> QwenTeacherForcedMetricArenas::endpoint(
    const Buffer& buffer, std::uint64_t owner_id, CudaCopyMemoryType type,
    std::int32_t device_or_numa) const {
  if (owner_id == 0)
    return Status::InvalidArgument("Qwen metric endpoint owner is invalid");
  return CudaCopyEndpoint{reinterpret_cast<std::uintptr_t>(buffer.data()),
      buffer.size_bytes(), 0, owner_id, buffer.generation(), type,
      static_cast<std::uint32_t>(owning_rank_), device_or_numa};
}
Result<CudaCopyEndpoint> QwenTeacherForcedMetricArenas::pinned_targets_endpoint(
    std::uint64_t owner_id) const {
  return endpoint(pinned_targets_, owner_id,
                  CudaCopyMemoryType::kRegisteredPinnedHost, numa_node_);
}
Result<CudaCopyEndpoint>
QwenTeacherForcedMetricArenas::pinned_sample_rows_endpoint(
    std::uint64_t owner_id) const {
  return endpoint(pinned_sample_rows_, owner_id,
                  CudaCopyMemoryType::kRegisteredPinnedHost, numa_node_);
}
Result<CudaCopyEndpoint>
QwenTeacherForcedMetricArenas::device_sample_rows_endpoint(
    std::uint64_t owner_id) const {
  return endpoint(device_sample_rows_, owner_id, CudaCopyMemoryType::kDevice,
                  owning_rank_);
}
Result<CudaCopyEndpoint> QwenTeacherForcedMetricArenas::device_targets_endpoint(
    std::uint64_t owner_id) const {
  return endpoint(device_targets_, owner_id, CudaCopyMemoryType::kDevice,
                  owning_rank_);
}
Result<CudaCopyEndpoint> QwenTeacherForcedMetricArenas::pinned_result_endpoint(
    std::uint64_t owner_id) const {
  return endpoint(pinned_result_, owner_id,
                  CudaCopyMemoryType::kRegisteredPinnedHost, numa_node_);
}
Result<CudaCopyEndpoint> QwenTeacherForcedMetricArenas::device_result_endpoint(
    std::uint64_t owner_id) const {
  return endpoint(device_result_, owner_id, CudaCopyMemoryType::kDevice,
                  owning_rank_);
}

}  // namespace pih
