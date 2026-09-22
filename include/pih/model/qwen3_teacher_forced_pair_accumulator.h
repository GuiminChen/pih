#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "pih/model/qwen3_teacher_forced_chunk_plan.h"
#include "pih/model/qwen3_teacher_forced_metric_oracle.h"

namespace pih {

class QwenPairedDeltaBuffer final {
 public:
  QwenPairedDeltaBuffer() = default;
  explicit QwenPairedDeltaBuffer(std::vector<double> values)
      : values_(std::move(values)) {}

  [[nodiscard]] std::span<const double> values() const noexcept {
    return values_;
  }

 private:
  std::vector<double> values_;
};

struct QwenTeacherForcedCategoryAggregate final {
  QwenTeacherForcedCategory category;
  std::uint64_t evaluated_positions = 0;
  std::uint64_t equal_argmax_positions = 0;
  double bf16_nll_sum = 0.0;
  double int4_nll_sum = 0.0;
  std::uint64_t nonfinite_count = 0;
  QwenPairedDeltaBuffer paired_nll_deltas{};
};

enum class QwenTeacherForcedPairAccumulatorState : std::uint8_t {
  kCollecting, kFinalized, kPoisoned,
};

class QwenTeacherForcedPairAccumulator final {
 public:
  static Result<QwenTeacherForcedPairAccumulator> Create(
      std::span<const QwenTeacherForcedChunk> expected_chunks);

  Status append(const QwenTeacherForcedChunk& chunk,
                const QwenTeacherForcedMetricBatch& bf16,
                const QwenTeacherForcedMetricBatch& int4);
  Result<std::array<QwenTeacherForcedCategoryAggregate, 6>> finalize();
  [[nodiscard]] QwenTeacherForcedPairAccumulatorState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::size_t consumed_chunks() const noexcept { return next_; }

 private:
  struct Sum final { double value = 0.0; double correction = 0.0; };
  struct MutableCategory final {
    std::uint64_t positions = 0;
    std::uint64_t equal = 0;
    Sum bf16;
    Sum int4;
    std::uint64_t nonfinite = 0;
    std::vector<double> paired_nll_deltas;
  };

  explicit QwenTeacherForcedPairAccumulator(
      std::vector<QwenTeacherForcedChunk> expected)
      : expected_(std::move(expected)) {}
  Status poison(std::string message);

  std::vector<QwenTeacherForcedChunk> expected_;
  std::array<MutableCategory, 6> categories_{};
  std::size_t next_ = 0;
  QwenTeacherForcedPairAccumulatorState state_ =
      QwenTeacherForcedPairAccumulatorState::kCollecting;
};

class QwenTeacherForcedPairRunCollector final {
 public:
  static Result<QwenTeacherForcedPairRunCollector> Create(
      std::span<const QwenTeacherForcedChunk> expected_chunks);

  Status stage_bf16(const QwenTeacherForcedChunk& chunk,
                    QwenTeacherForcedMetricBatch batch);
  Status stage_int4(const QwenTeacherForcedChunk& chunk,
                    QwenTeacherForcedMetricBatch batch);
  Result<std::array<QwenTeacherForcedCategoryAggregate, 6>> finalize();

 private:
  explicit QwenTeacherForcedPairRunCollector(
      QwenTeacherForcedPairAccumulator accumulator)
      : accumulator_(std::move(accumulator)) {}
  Status poison(std::string message);

  QwenTeacherForcedPairAccumulator accumulator_;
  std::optional<QwenTeacherForcedChunk> staged_chunk_;
  std::optional<QwenTeacherForcedMetricBatch> staged_bf16_;
  bool poisoned_ = false;
};

}  // namespace pih
