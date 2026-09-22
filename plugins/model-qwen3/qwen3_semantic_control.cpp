#include "pih/model/qwen3_semantic_control.h"

#include <array>
#include <string_view>

namespace pih {
namespace {

bool nonzero(const Sha256Digest& digest) {
  for (const auto byte : digest.bytes) {
    if (byte != std::byte{0}) return true;
  }
  return false;
}

bool equal_semantics(const QwenSemanticOutcome& lhs,
                     const QwenSemanticOutcome& rhs) {
  return lhs.dispatch_root == rhs.dispatch_root &&
         lhs.final_logits_root == rhs.final_logits_root &&
         lhs.greedy_trajectory_root == rhs.greedy_trajectory_root &&
         lhs.kv_state_root == rhs.kv_state_root &&
         lhs.accepted_token_ledger_root == rhs.accepted_token_ledger_root;
}

Status update_u64(Sha256& digest, std::uint64_t value) {
  std::array<std::byte, 8> wire{};
  for (std::size_t index = 0; index < wire.size(); ++index) {
    wire[index] = static_cast<std::byte>(value >> (index * 8U));
  }
  return digest.update(wire);
}

}  // namespace

Result<QwenSemanticControlReceipt> QwenSemanticControlReceipt::Create(
    const QwenNumericalRunIdentity& identity,
    const QwenBf16TapSuiteRunReceipt& tap_suite,
    const QwenSemanticOutcome& instrumented,
    const QwenSemanticOutcome& control) {
  constexpr std::size_t kRequiredCaptureCount = 369;
  constexpr std::size_t kRequiredFixtureCount = 5;
  if (tap_suite.suite_generation != identity.run_generation() ||
      tap_suite.fixture_count != kRequiredFixtureCount ||
      tap_suite.capture_count != kRequiredCaptureCount ||
      !nonzero(tap_suite.suite_plan_digest) ||
      !nonzero(tap_suite.writer_root)) {
    return Status::FailedPrecondition(
        "Qwen instrumented tap run is incomplete or spliced");
  }
  const std::array roots{
      instrumented.dispatch_root, instrumented.final_logits_root,
      instrumented.greedy_trajectory_root, instrumented.kv_state_root,
      instrumented.accepted_token_ledger_root, control.dispatch_root,
      control.final_logits_root, control.greedy_trajectory_root,
      control.kv_state_root, control.accepted_token_ledger_root};
  for (const auto& root : roots) {
    if (!nonzero(root)) {
      return Status::FailedPrecondition(
          "Qwen semantic control outcome lacks a required root");
    }
  }
  if (!equal_semantics(instrumented, control)) {
    return Status::FailedPrecondition(
        "Qwen instrumented and control semantic outcomes drifted");
  }
  if (instrumented.device_peak_bytes == 0 ||
      instrumented.pinned_peak_bytes == 0 || control.device_peak_bytes == 0 ||
      control.pinned_peak_bytes == 0) {
    return Status::FailedPrecondition(
        "Qwen semantic control lacks separate capacity observations");
  }

  auto identity_digest = identity.semantic_digest();
  if (!identity_digest.ok()) return identity_digest.status();
  Sha256 digest;
  constexpr std::string_view domain =
      "pih.paired_instrumented_semantic_control.v1";
  Status status = digest.update(std::as_bytes(std::span(domain)));
  if (status.ok()) status = digest.update(identity_digest->bytes);
  if (status.ok()) status = digest.update(tap_suite.suite_plan_digest.bytes);
  if (status.ok()) status = digest.update(tap_suite.writer_root.bytes);
  for (std::size_t index = 0; status.ok() && index < 5; ++index) {
    status = digest.update(roots[index].bytes);
  }
  if (status.ok()) status = update_u64(digest, instrumented.device_peak_bytes);
  if (status.ok()) status = update_u64(digest, instrumented.pinned_peak_bytes);
  if (status.ok()) status = update_u64(digest, control.device_peak_bytes);
  if (status.ok()) status = update_u64(digest, control.pinned_peak_bytes);
  if (!status.ok()) return status;
  auto semantic_digest = digest.finalize();
  if (!semantic_digest.ok()) return semantic_digest.status();
  return QwenSemanticControlReceipt(*semantic_digest, tap_suite, instrumented,
                                    control);
}

}  // namespace pih
