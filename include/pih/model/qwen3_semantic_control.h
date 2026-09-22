#pragma once

#include <cstdint>

#include "pih/model/qwen3_numerical_run.h"
#include "pih/model/qwen3_bf16_tap_suite_run.h"

namespace pih {

struct QwenSemanticOutcome final {
  Sha256Digest dispatch_root;
  Sha256Digest final_logits_root;
  Sha256Digest greedy_trajectory_root;
  Sha256Digest kv_state_root;
  Sha256Digest accepted_token_ledger_root;
  std::uint64_t device_peak_bytes;
  std::uint64_t pinned_peak_bytes;
};

class QwenSemanticControlReceipt final {
 public:
  static Result<QwenSemanticControlReceipt> Create(
      const QwenNumericalRunIdentity& identity,
      const QwenBf16TapSuiteRunReceipt& tap_suite,
      const QwenSemanticOutcome& instrumented,
      const QwenSemanticOutcome& control);

  [[nodiscard]] const Sha256Digest& semantic_digest() const noexcept {
    return semantic_digest_;
  }
  [[nodiscard]] std::uint64_t instrumented_device_peak_bytes() const noexcept {
    return instrumented_device_peak_bytes_;
  }
  [[nodiscard]] std::uint64_t control_device_peak_bytes() const noexcept {
    return control_device_peak_bytes_;
  }
  [[nodiscard]] std::uint64_t instrumented_pinned_peak_bytes() const noexcept {
    return instrumented_pinned_peak_bytes_;
  }
  [[nodiscard]] std::uint64_t control_pinned_peak_bytes() const noexcept {
    return control_pinned_peak_bytes_;
  }
  [[nodiscard]] const QwenBf16TapSuiteRunReceipt& tap_suite() const noexcept {
    return tap_suite_;
  }
  [[nodiscard]] const Sha256Digest& dispatch_root() const noexcept {
    return semantic_.dispatch_root;
  }
  [[nodiscard]] const Sha256Digest& final_logits_root() const noexcept {
    return semantic_.final_logits_root;
  }
  [[nodiscard]] const Sha256Digest& greedy_trajectory_root() const noexcept {
    return semantic_.greedy_trajectory_root;
  }
  [[nodiscard]] const Sha256Digest& kv_state_root() const noexcept {
    return semantic_.kv_state_root;
  }
  [[nodiscard]] const Sha256Digest& accepted_token_ledger_root()
      const noexcept {
    return semantic_.accepted_token_ledger_root;
  }

 private:
  QwenSemanticControlReceipt(Sha256Digest semantic_digest,
                             QwenBf16TapSuiteRunReceipt tap_suite,
                             const QwenSemanticOutcome& instrumented,
                             const QwenSemanticOutcome& control)
      : semantic_digest_(semantic_digest),
        tap_suite_(std::move(tap_suite)), semantic_(instrumented),
        instrumented_device_peak_bytes_(instrumented.device_peak_bytes),
        control_device_peak_bytes_(control.device_peak_bytes),
        instrumented_pinned_peak_bytes_(instrumented.pinned_peak_bytes),
        control_pinned_peak_bytes_(control.pinned_peak_bytes) {}

  Sha256Digest semantic_digest_;
  QwenBf16TapSuiteRunReceipt tap_suite_;
  QwenSemanticOutcome semantic_;
  std::uint64_t instrumented_device_peak_bytes_;
  std::uint64_t control_device_peak_bytes_;
  std::uint64_t instrumented_pinned_peak_bytes_;
  std::uint64_t control_pinned_peak_bytes_;
};

}  // namespace pih
