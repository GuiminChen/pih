#include "pih/model/qwen3_teacher_forced_pair_accumulator.h"

#include <cmath>
#include <utility>

namespace pih {
namespace {

bool same_chunk(const QwenTeacherForcedChunk& left,
                const QwenTeacherForcedChunk& right) {
  return left.category == right.category &&
         left.category_row_begin == right.category_row_begin &&
         left.rows == right.rows && left.logits_bytes == right.logits_bytes;
}

bool valid_row(const QwenTeacherForcedMetricRow& row) {
  return row.argmax_token < QwenTeacherForcedChunkPlan::kVocabularySize &&
         ((row.finite && std::isfinite(row.target_nll) &&
           row.target_nll >= 0.0) || (!row.finite && row.target_nll == 0.0));
}

}  // namespace

Result<QwenTeacherForcedPairAccumulator>
QwenTeacherForcedPairAccumulator::Create(
    std::span<const QwenTeacherForcedChunk> expected_chunks) {
  if (expected_chunks.empty())
    return Status::InvalidArgument("Qwen paired chunk plan is empty");
  std::array<bool, 6> seen{};
  std::array<std::uint64_t, 6> next_rows{};
  std::size_t previous_category = 0;
  bool first = true;
  for (const auto& chunk : expected_chunks) {
    const auto category = static_cast<std::size_t>(chunk.category);
    const auto expected_bytes = static_cast<std::uint64_t>(chunk.rows) *
        QwenTeacherForcedChunkPlan::kVocabularySize * sizeof(float);
    if (category >= seen.size() || chunk.rows == 0 ||
        chunk.rows > QwenTeacherForcedChunkPlan::kMaximumRowsPerChunk ||
        chunk.logits_bytes != expected_bytes ||
        chunk.category_row_begin != next_rows[category] ||
        (!first && category < previous_category) ||
        (!first && category > previous_category + 1)) {
      return Status::InvalidArgument("Qwen paired chunk plan is invalid");
    }
    seen[category] = true;
    next_rows[category] += chunk.rows;
    previous_category = category;
    first = false;
  }
  for (bool category_seen : seen)
    if (!category_seen)
      return Status::InvalidArgument("Qwen paired category set is incomplete");
  return QwenTeacherForcedPairAccumulator(
      std::vector<QwenTeacherForcedChunk>(expected_chunks.begin(),
                                          expected_chunks.end()));
}

Status QwenTeacherForcedPairAccumulator::poison(std::string message) {
  state_ = QwenTeacherForcedPairAccumulatorState::kPoisoned;
  return Status::InvalidArgument(std::move(message));
}

Status QwenTeacherForcedPairAccumulator::append(
    const QwenTeacherForcedChunk& chunk,
    const QwenTeacherForcedMetricBatch& bf16,
    const QwenTeacherForcedMetricBatch& int4) {
  if (state_ != QwenTeacherForcedPairAccumulatorState::kCollecting)
    return Status::FailedPrecondition("Qwen paired accumulator is not collecting");
  if (next_ >= expected_.size() || !same_chunk(chunk, expected_[next_]) ||
      bf16.rows.size() != chunk.rows || int4.rows.size() != chunk.rows)
    return poison("Qwen paired chunk identity is invalid");
  std::uint64_t bf16_nonfinite = 0, int4_nonfinite = 0;
  for (std::size_t row = 0; row < chunk.rows; ++row) {
    if (!valid_row(bf16.rows[row]) || !valid_row(int4.rows[row]))
      return poison("Qwen paired metric row is invalid");
    bf16_nonfinite += !bf16.rows[row].finite;
    int4_nonfinite += !int4.rows[row].finite;
  }
  if (bf16_nonfinite != bf16.nonfinite_count ||
      int4_nonfinite != int4.nonfinite_count)
    return poison("Qwen paired nonfinite count drifted");

  auto& category = categories_[static_cast<std::size_t>(chunk.category)];
  const auto add = [](double value, Sum& sum) {
    const double adjusted = value - sum.correction;
    const double next = sum.value + adjusted;
    sum.correction = (next - sum.value) - adjusted;
    sum.value = next;
  };
  category.positions += chunk.rows;
  for (std::size_t row = 0; row < chunk.rows; ++row) {
    category.equal +=
        bf16.rows[row].argmax_token == int4.rows[row].argmax_token;
    if (bf16.rows[row].finite) add(bf16.rows[row].target_nll, category.bf16);
    if (int4.rows[row].finite) add(int4.rows[row].target_nll, category.int4);
    if (bf16.rows[row].finite && int4.rows[row].finite) {
      category.paired_nll_deltas.push_back(
          int4.rows[row].target_nll - bf16.rows[row].target_nll);
    }
    category.nonfinite += !bf16.rows[row].finite || !int4.rows[row].finite;
  }
  ++next_;
  return Status::Ok();
}

Result<std::array<QwenTeacherForcedCategoryAggregate, 6>>
QwenTeacherForcedPairAccumulator::finalize() {
  if (state_ != QwenTeacherForcedPairAccumulatorState::kCollecting ||
      next_ != expected_.size())
    return Status::FailedPrecondition("Qwen paired accumulation is incomplete");
  std::array<QwenTeacherForcedCategoryAggregate, 6> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    const auto& value = categories_[index];
    result[index] = {static_cast<QwenTeacherForcedCategory>(index),
                     value.positions, value.equal, value.bf16.value,
                     value.int4.value, value.nonfinite,
                     QwenPairedDeltaBuffer(
                         std::move(categories_[index].paired_nll_deltas))};
  }
  state_ = QwenTeacherForcedPairAccumulatorState::kFinalized;
  return result;
}

Result<QwenTeacherForcedPairRunCollector>
QwenTeacherForcedPairRunCollector::Create(
    std::span<const QwenTeacherForcedChunk> expected_chunks) {
  auto accumulator = QwenTeacherForcedPairAccumulator::Create(expected_chunks);
  if (!accumulator.ok()) return accumulator.status();
  return QwenTeacherForcedPairRunCollector(std::move(*accumulator));
}

Status QwenTeacherForcedPairRunCollector::poison(std::string message) {
  poisoned_ = true;
  staged_chunk_.reset();
  staged_bf16_.reset();
  return Status::InvalidArgument(std::move(message));
}

Status QwenTeacherForcedPairRunCollector::stage_bf16(
    const QwenTeacherForcedChunk& chunk, QwenTeacherForcedMetricBatch batch) {
  if (poisoned_ || staged_bf16_.has_value())
    return poison("Qwen paired BF16 role order is invalid");
  staged_chunk_ = chunk;
  staged_bf16_ = std::move(batch);
  return Status::Ok();
}

Status QwenTeacherForcedPairRunCollector::stage_int4(
    const QwenTeacherForcedChunk& chunk, QwenTeacherForcedMetricBatch batch) {
  if (poisoned_ || !staged_chunk_.has_value() ||
      !staged_bf16_.has_value() || !same_chunk(*staged_chunk_, chunk))
    return poison("Qwen paired INT4 role or chunk identity is invalid");
  auto status = accumulator_.append(chunk, *staged_bf16_, batch);
  staged_chunk_.reset();
  staged_bf16_.reset();
  if (!status.ok()) poisoned_ = true;
  return status;
}

Result<std::array<QwenTeacherForcedCategoryAggregate, 6>>
QwenTeacherForcedPairRunCollector::finalize() {
  if (poisoned_ || staged_bf16_.has_value())
    return Status::FailedPrecondition("Qwen paired run collection is incomplete");
  return accumulator_.finalize();
}

}  // namespace pih
