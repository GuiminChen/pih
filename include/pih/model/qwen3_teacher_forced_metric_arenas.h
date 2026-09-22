#pragma once

#include <limits>
#include <memory>
#include <span>

#include "pih/backend/cuda/pinned_placement_verifier.h"
#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/core/buffer.h"
#include "pih/model/qwen3_teacher_forced_metric_result_layout.h"
#include "pih/model/qwen3_teacher_forced_metric_arena_lease.h"

namespace pih {

// Owns the bounded qualification-only metric buffers. Logits remain owned by
// the engine execution arena; only selected rows, targets and compact metric
// results live here.
class QwenTeacherForcedMetricArenas final {
 public:
  static Result<QwenTeacherForcedMetricArenas> AllocateVerified(
      std::uint32_t row_capacity, Allocator& device_allocator,
      RegisteredPinnedAllocator& pinned_allocator,
      PinnedPlacementVerifier& verifier, std::int32_t owning_rank,
      std::int32_t numa_node);

  QwenTeacherForcedMetricArenas(const QwenTeacherForcedMetricArenas&) = delete;
  QwenTeacherForcedMetricArenas& operator=(
      const QwenTeacherForcedMetricArenas&) = delete;
  QwenTeacherForcedMetricArenas(QwenTeacherForcedMetricArenas&&) noexcept =
      default;
  QwenTeacherForcedMetricArenas& operator=(
      QwenTeacherForcedMetricArenas&&) noexcept = delete;

  [[nodiscard]] const QwenTeacherForcedMetricResultLayout& layout() const
      noexcept { return layout_; }
  [[nodiscard]] std::span<std::byte> pinned_sample_rows() noexcept;
  [[nodiscard]] std::span<std::byte> pinned_targets() noexcept;
  [[nodiscard]] std::span<std::byte> pinned_result() noexcept;
  Result<TensorView> device_sample_rows(std::uint32_t rows) const;
  Result<TensorView> device_sample_rows(
      std::uint32_t rows, std::uint64_t execution_generation) const;
  Result<TensorView> device_targets(std::uint32_t rows) const;
  Result<TensorView> device_targets(
      std::uint32_t rows, std::uint64_t execution_generation) const;
  Result<TensorView> device_argmax(std::uint32_t rows) const;
  Result<TensorView> device_argmax(
      std::uint32_t rows, std::uint64_t execution_generation) const;
  Result<TensorView> device_nll(std::uint32_t rows) const;
  Result<TensorView> device_nll(
      std::uint32_t rows, std::uint64_t execution_generation) const;
  Result<TensorView> device_nonfinite(std::uint32_t rows) const;
  Result<TensorView> device_nonfinite(
      std::uint32_t rows, std::uint64_t execution_generation) const;
  Result<TensorView> device_error() const;
  Result<TensorView> device_error(std::uint64_t execution_generation) const;
  Result<CudaCopyEndpoint> pinned_sample_rows_endpoint(
      std::uint64_t owner_id) const;
  Result<CudaCopyEndpoint> device_sample_rows_endpoint(
      std::uint64_t owner_id) const;
  Result<CudaCopyEndpoint> pinned_targets_endpoint(
      std::uint64_t owner_id) const;
  Result<CudaCopyEndpoint> device_targets_endpoint(
      std::uint64_t owner_id) const;
  Result<CudaCopyEndpoint> pinned_result_endpoint(
      std::uint64_t owner_id) const;
  Result<CudaCopyEndpoint> device_result_endpoint(
      std::uint64_t owner_id) const;
  Result<QwenTeacherForcedMetricArenaLease> acquire_transaction_lease() {
    if (!lease_state_)
      return Status::FailedPrecondition(
          "Qwen teacher-forced metric arena lease state is absent");
    return QwenTeacherForcedMetricArenaLease::Acquire(*lease_state_);
  }
  Result<std::uint64_t> reserve_factory_generation() {
    if (!lease_state_)
      return Status::FailedPrecondition(
          "Qwen teacher-forced metric arena lease state is absent");
    auto current = lease_state_->last_factory_generation.load(
        std::memory_order_acquire);
    while (true) {
      if (current == std::numeric_limits<std::uint64_t>::max())
        return Status::ResourceExhausted(
            "Qwen teacher-forced factory generation exhausted");
      if (lease_state_->last_factory_generation.compare_exchange_weak(
              current, current + 1, std::memory_order_acq_rel,
              std::memory_order_acquire))
        return current + 1;
    }
  }
  [[nodiscard]] std::uint64_t arena_identity() const noexcept {
    return lease_state_ ? lease_state_->arena_identity : 0;
  }

 private:
  QwenTeacherForcedMetricArenas(
      QwenTeacherForcedMetricResultLayout layout, Buffer device_sample_rows,
      Buffer device_targets, Buffer device_result, Buffer pinned_sample_rows,
      Buffer pinned_targets, Buffer pinned_result,
      std::int32_t owning_rank, std::int32_t numa_node,
      std::unique_ptr<QwenTeacherForcedMetricArenaLeaseState> lease_state)
      : layout_(layout), device_sample_rows_(std::move(device_sample_rows)),
        device_targets_(std::move(device_targets)),
        device_result_(std::move(device_result)),
        pinned_sample_rows_(std::move(pinned_sample_rows)),
        pinned_targets_(std::move(pinned_targets)),
        pinned_result_(std::move(pinned_result)), owning_rank_(owning_rank),
        numa_node_(numa_node), lease_state_(std::move(lease_state)) {}

  Result<TensorView> result_view(QwenBf16ArenaSpan span, DType dtype,
                                 std::uint32_t rows,
                                 std::uint64_t generation) const;
  Result<CudaCopyEndpoint> endpoint(const Buffer& buffer,
      std::uint64_t owner_id, CudaCopyMemoryType type,
      std::int32_t device_or_numa) const;

  QwenTeacherForcedMetricResultLayout layout_;
  Buffer device_sample_rows_;
  Buffer device_targets_;
  Buffer device_result_;
  Buffer pinned_sample_rows_;
  Buffer pinned_targets_;
  Buffer pinned_result_;
  std::int32_t owning_rank_ = -1;
  std::int32_t numa_node_ = -1;
  std::unique_ptr<QwenTeacherForcedMetricArenaLeaseState> lease_state_;
};

}  // namespace pih
