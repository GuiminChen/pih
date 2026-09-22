#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/backend/cuda/completion_frontier.h"
#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/model/qwen3_numerical_tap_slot.h"
#include "pih/model/qwen3_bf16_prepared_execution.h"

namespace pih {

enum class QwenNumericalTapTransferState : std::uint8_t {
  kProducerPending = 0,
  kSnapshotReady,
  kSnapshotInFlight,
  kHostReady,
  kHostInFlight,
  kHostComplete,
  kSealed,
  kPoisoned,
};

class QwenNumericalTapTransfer final {
 public:
  static Result<QwenNumericalTapTransfer> Create(
      std::size_t capture_index, std::uint64_t capture_generation,
      std::uint64_t bytes, std::uint64_t producer_plan_generation,
      CudaTypedCopyPlan snapshot_copy, CudaTypedCopyPlan host_copy);

  QwenNumericalTapTransfer(const QwenNumericalTapTransfer&) = delete;
  QwenNumericalTapTransfer& operator=(const QwenNumericalTapTransfer&) = delete;
  QwenNumericalTapTransfer(QwenNumericalTapTransfer&&) noexcept = default;
  QwenNumericalTapTransfer& operator=(QwenNumericalTapTransfer&&) noexcept = default;

  Status complete_producer(const CudaCompletionFrontier& frontier);
  Status submit_inline_snapshot(TypedCopyDriver& driver,
                                std::uint64_t producer_plan_generation);
  Status submit_snapshot(TypedCopyDriver& driver);
  Status complete_snapshot(const CudaCompletionFrontier& frontier);
  Status submit_host(TypedCopyDriver& driver);
  Status complete_host(const CudaCompletionFrontier& frontier);
  Result<QwenNumericalTapReceipt> seal(
      std::span<const std::byte> observed_bytes);

  [[nodiscard]] QwenNumericalTapTransferState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::size_t capture_index() const noexcept {
    return capture_index_;
  }
  [[nodiscard]] std::uint64_t producer_plan_generation() const noexcept {
    return producer_plan_generation_;
  }
  [[nodiscard]] DriverStreamHandle snapshot_stream() const noexcept {
    return snapshot_copy_.stream();
  }

 private:
  QwenNumericalTapTransfer(std::size_t capture_index,
                           std::uint64_t capture_generation,
                           std::uint64_t bytes,
                           std::uint64_t producer_plan_generation,
                           CudaTypedCopyPlan snapshot_copy,
                           CudaTypedCopyPlan host_copy)
      : capture_index_(capture_index),
        capture_generation_(capture_generation), bytes_(bytes),
        producer_plan_generation_(producer_plan_generation),
        snapshot_copy_(std::move(snapshot_copy)),
        host_copy_(std::move(host_copy)) {}
  Status poison(const char* message);
  Status validate_copy_frontier(const CudaCompletionFrontier& frontier,
                                const CudaTypedCopyPlan& plan,
                                const char* message);

  std::size_t capture_index_;
  std::uint64_t capture_generation_;
  std::uint64_t bytes_;
  std::uint64_t producer_plan_generation_;
  CudaTypedCopyPlan snapshot_copy_;
  CudaTypedCopyPlan host_copy_;
  QwenNumericalTapTransferState state_ =
      QwenNumericalTapTransferState::kProducerPending;
};

class QwenNumericalTapInlineSnapshotDriver final
    : public QwenBf16TapSnapshotDriver {
 public:
  static Result<QwenNumericalTapInlineSnapshotDriver> Create(
      std::span<QwenNumericalTapTransfer> transfers,
      TypedCopyDriver& copy_driver,
      std::uint64_t producer_plan_generation);

  Status snapshot(const QwenBf16TapBinding& binding,
                  DriverStreamHandle stream) override;

 private:
  QwenNumericalTapInlineSnapshotDriver(
      std::span<QwenNumericalTapTransfer> transfers,
      TypedCopyDriver& copy_driver,
      std::uint64_t producer_plan_generation)
      : transfers_(transfers), copy_driver_(&copy_driver),
        producer_plan_generation_(producer_plan_generation) {}

  std::span<QwenNumericalTapTransfer> transfers_;
  TypedCopyDriver* copy_driver_;
  std::uint64_t producer_plan_generation_;
};

}  // namespace pih
