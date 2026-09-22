#include "pih/model/qwen3_numerical_tap_run.h"

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

Result<QwenNumericalTapRun> QwenNumericalTapRun::Create(
    const QwenNumericalTapPlan& plan, std::uint64_t run_generation) {
  if (run_generation == 0) {
    return Status::InvalidArgument("Qwen numerical tap run generation is zero");
  }
  auto plan_digest = plan.semantic_digest();
  if (!plan_digest.ok()) return plan_digest.status();
  std::vector<std::uint64_t> expected_bytes;
  expected_bytes.reserve(plan.captures().size());
  for (const auto& capture : plan.captures()) {
    expected_bytes.push_back(capture.size_bytes);
  }
  return QwenNumericalTapRun(run_generation, *plan_digest,
                             std::move(expected_bytes));
}

Status QwenNumericalTapRun::poison(const char* message) {
  state_ = QwenNumericalTapRunState::kPoisoned;
  return Status::FailedPrecondition(message);
}

Status QwenNumericalTapRun::record(const QwenNumericalTapReceipt& receipt) {
  if (state_ != QwenNumericalTapRunState::kCollecting) {
    return Status::FailedPrecondition("Qwen numerical tap run is not collecting");
  }
  if (receipt.capture_generation != generation_ ||
      receipt.capture_index >= receipts_.size() ||
      receipt.bytes != expected_bytes_[receipt.capture_index] ||
      receipts_[receipt.capture_index].has_value()) {
    return poison("Qwen numerical tap receipt identity is invalid");
  }
  receipts_[receipt.capture_index] = receipt;
  ++recorded_count_;
  return Status::Ok();
}

Result<QwenNumericalTapRunReceipt> QwenNumericalTapRun::seal() {
  if (state_ != QwenNumericalTapRunState::kCollecting) {
    return Status::FailedPrecondition("Qwen numerical tap run cannot be sealed");
  }
  if (recorded_count_ != receipts_.size()) {
    return poison("Qwen numerical tap run has dropped captures");
  }
  Sha256 root;
  constexpr std::string_view domain = "pih.qwen_numerical_tap_writer.v1";
  Status status = root.update(std::as_bytes(std::span(domain)));
  if (status.ok()) status = root.update(plan_digest_.bytes);
  if (status.ok()) status = update_u64(root, generation_);
  if (status.ok()) status = update_u64(root, receipts_.size());
  for (std::size_t index = 0; status.ok() && index < receipts_.size(); ++index) {
    status = update_u64(root, index);
    if (status.ok()) status = update_u64(root, receipts_[index]->bytes);
    if (status.ok()) status = root.update(receipts_[index]->digest.bytes);
  }
  if (!status.ok()) {
    state_ = QwenNumericalTapRunState::kPoisoned;
    return status;
  }
  auto writer_root = root.finalize();
  if (!writer_root.ok()) {
    state_ = QwenNumericalTapRunState::kPoisoned;
    return writer_root.status();
  }
  state_ = QwenNumericalTapRunState::kSealed;
  return QwenNumericalTapRunReceipt{generation_, receipts_.size(), plan_digest_,
                                    *writer_root};
}

}  // namespace pih
