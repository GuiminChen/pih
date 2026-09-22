#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "pih/model/qwen3_numerical_tap_plan.h"
#include "pih/model/qwen3_numerical_tap_slot.h"

namespace pih {

enum class QwenNumericalTapRunState : std::uint8_t {
  kCollecting = 0,
  kSealed,
  kPoisoned,
};

struct QwenNumericalTapRunReceipt final {
  std::uint64_t run_generation;
  std::size_t capture_count;
  Sha256Digest plan_digest;
  Sha256Digest writer_root;
};

class QwenNumericalTapRun final {
 public:
  static Result<QwenNumericalTapRun> Create(
      const QwenNumericalTapPlan& plan, std::uint64_t run_generation);

  Status record(const QwenNumericalTapReceipt& receipt);
  Result<QwenNumericalTapRunReceipt> seal();

  [[nodiscard]] QwenNumericalTapRunState state() const noexcept {
    return state_;
  }

 private:
  QwenNumericalTapRun(std::uint64_t generation, Sha256Digest plan_digest,
                      std::vector<std::uint64_t> expected_bytes)
      : generation_(generation), plan_digest_(plan_digest),
        expected_bytes_(std::move(expected_bytes)),
        receipts_(expected_bytes_.size()) {}
  Status poison(const char* message);

  std::uint64_t generation_;
  Sha256Digest plan_digest_;
  std::vector<std::uint64_t> expected_bytes_;
  std::vector<std::optional<QwenNumericalTapReceipt>> receipts_;
  std::size_t recorded_count_ = 0;
  QwenNumericalTapRunState state_ = QwenNumericalTapRunState::kCollecting;
};

}  // namespace pih
