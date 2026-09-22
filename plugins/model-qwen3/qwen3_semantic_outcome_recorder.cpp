#include "pih/model/qwen3_semantic_outcome_recorder.h"

#include <algorithm>
#include <array>
#include <string_view>

namespace pih {
namespace {

Status update_u64(Sha256& digest, std::uint64_t value) {
  std::array<std::byte, 8> wire{};
  for (std::size_t index = 0; index < wire.size(); ++index) {
    wire[index] = static_cast<std::byte>(value >> (index * 8U));
  }
  return digest.update(wire);
}

}  // namespace

Status QwenSemanticOutcomeRecorder::poison(const char* message) {
  state_ = QwenSemanticOutcomeRecorderState::kPoisoned;
  return Status::FailedPrecondition(message);
}

Status QwenSemanticOutcomeRecorder::record(
    std::span<const std::byte> bytes, std::uint64_t maximum,
    const char* domain, Sha256Digest& destination, bool& present) {
  if (state_ != QwenSemanticOutcomeRecorderState::kCollecting || present) {
    return poison("Qwen semantic outcome field has invalid lifecycle");
  }
  if (bytes.empty() || bytes.size() > maximum) {
    return poison("Qwen semantic outcome field has invalid extent");
  }
  Sha256 digest;
  const std::string_view domain_view(domain);
  Status status = digest.update(std::as_bytes(std::span(domain_view)));
  if (status.ok()) status = update_u64(digest, bytes.size());
  if (status.ok()) status = digest.update(bytes);
  if (!status.ok()) {
    state_ = QwenSemanticOutcomeRecorderState::kPoisoned;
    return status;
  }
  auto root = digest.finalize();
  if (!root.ok()) {
    state_ = QwenSemanticOutcomeRecorderState::kPoisoned;
    return root.status();
  }
  destination = *root;
  present = true;
  return Status::Ok();
}

Status QwenSemanticOutcomeRecorder::record_dispatch(
    std::span<const std::byte> bytes) {
  return record(bytes, kMaximumDispatchBytes,
                "pih.qwen_semantic.dispatch.v1", outcome_.dispatch_root,
                dispatch_present_);
}

Status QwenSemanticOutcomeRecorder::record_dispatch(
    const QwenBf16CommandBuffer& commands) {
  const auto wire = commands.canonical_wire();
  return record_dispatch(wire);
}

Status QwenSemanticOutcomeRecorder::record_final_logits(
    std::span<const std::byte> bytes) {
  return record(bytes, kMaximumLogitsBytes,
                "pih.qwen_semantic.final_logits.v1",
                outcome_.final_logits_root, logits_present_);
}

Status QwenSemanticOutcomeRecorder::record_greedy_trajectory(
    std::span<const std::byte> bytes) {
  return record(bytes, kMaximumTrajectoryBytes,
                "pih.qwen_semantic.greedy_trajectory.v1",
                outcome_.greedy_trajectory_root, trajectory_present_);
}

Status QwenSemanticOutcomeRecorder::record_kv_state(
    std::span<const std::byte> bytes) {
  return record(bytes, kMaximumKvStateBytes,
                "pih.qwen_semantic.kv_state.v1", outcome_.kv_state_root,
                kv_present_);
}

Status QwenSemanticOutcomeRecorder::record_accepted_token_ledger(
    std::span<const std::byte> bytes) {
  return record(bytes, kMaximumLedgerBytes,
                "pih.qwen_semantic.accepted_token_ledger.v1",
                outcome_.accepted_token_ledger_root, ledger_present_);
}

Status QwenSemanticOutcomeRecorder::observe_capacity(
    std::uint64_t device_bytes, std::uint64_t pinned_bytes) {
  if (state_ != QwenSemanticOutcomeRecorderState::kCollecting) {
    return poison("Qwen semantic capacity observation has invalid lifecycle");
  }
  if (device_bytes == 0 || pinned_bytes == 0) {
    return poison("Qwen semantic capacity observation is empty");
  }
  outcome_.device_peak_bytes =
      std::max(outcome_.device_peak_bytes, device_bytes);
  outcome_.pinned_peak_bytes =
      std::max(outcome_.pinned_peak_bytes, pinned_bytes);
  return Status::Ok();
}

Result<QwenSemanticOutcome> QwenSemanticOutcomeRecorder::seal() {
  if (state_ != QwenSemanticOutcomeRecorderState::kCollecting ||
      !dispatch_present_ || !logits_present_ || !trajectory_present_ ||
      !kv_present_ || !ledger_present_ || outcome_.device_peak_bytes == 0 ||
      outcome_.pinned_peak_bytes == 0) {
    return poison("Qwen semantic outcome is incomplete");
  }
  state_ = QwenSemanticOutcomeRecorderState::kSealed;
  return outcome_;
}

}  // namespace pih
