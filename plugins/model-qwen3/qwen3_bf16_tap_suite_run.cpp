#include "pih/model/qwen3_bf16_tap_suite_run.h"

#include <limits>
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

Result<QwenBf16TapSuiteRun> QwenBf16TapSuiteRun::Create(
    const QwenBf16TapSuitePlan& suite,
    std::uint64_t suite_generation,
    std::uint64_t first_fixture_generation) {
  if (suite_generation == 0 || first_fixture_generation == 0 ||
      first_fixture_generation >
          std::numeric_limits<std::uint64_t>::max() -
              QwenBf16TapSuitePlan::kFixtureCount ||
      !validate_qwen_bf16_tap_suite_coverage(suite).ok()) {
    return Status::InvalidArgument("Qwen tap suite run identity is invalid");
  }
  auto suite_digest = suite.semantic_digest();
  if (!suite_digest.ok()) return suite_digest.status();
  std::array<Sha256Digest, QwenBf16TapSuitePlan::kFixtureCount> digests{};
  std::array<std::size_t, QwenBf16TapSuitePlan::kFixtureCount> counts{};
  for (std::size_t index = 0; index < suite.size(); ++index) {
    auto digest = suite[index].taps.semantic_digest();
    if (!digest.ok()) return digest.status();
    digests[index] = *digest;
    counts[index] = suite[index].taps.captures().size();
  }
  return QwenBf16TapSuiteRun(suite_generation, first_fixture_generation,
                             *suite_digest, digests, counts);
}

Status QwenBf16TapSuiteRun::poison(const char* message) {
  state_ = QwenBf16TapSuiteRunState::kPoisoned;
  return Status::FailedPrecondition(message);
}

Status QwenBf16TapSuiteRun::record(
    std::size_t fixture_index,
    const QwenNumericalTapRunReceipt& receipt) {
  if (state_ != QwenBf16TapSuiteRunState::kCollecting) {
    return Status::FailedPrecondition("Qwen tap suite run is not collecting");
  }
  if (fixture_index >= receipts_.size() ||
      receipts_[fixture_index].has_value() ||
      receipt.run_generation != first_fixture_generation_ + fixture_index ||
      receipt.capture_count != fixture_capture_counts_[fixture_index] ||
      !(receipt.plan_digest == fixture_plan_digests_[fixture_index])) {
    return poison("Qwen tap suite fixture receipt identity is invalid");
  }
  receipts_[fixture_index] = receipt;
  ++recorded_count_;
  return Status::Ok();
}

Result<QwenBf16TapSuiteRunReceipt> QwenBf16TapSuiteRun::seal() {
  if (state_ != QwenBf16TapSuiteRunState::kCollecting) {
    return Status::FailedPrecondition("Qwen tap suite run cannot be sealed");
  }
  if (recorded_count_ != receipts_.size()) {
    return poison("Qwen tap suite run has a missing fixture");
  }
  Sha256 root;
  constexpr std::string_view domain =
      "pih.qwen_bf16_tap_suite_writer.v1";
  Status status = root.update(std::as_bytes(std::span(domain)));
  if (status.ok()) status = root.update(suite_plan_digest_.bytes);
  if (status.ok()) status = update_u64(root, suite_generation_);
  if (status.ok()) status = update_u64(root, receipts_.size());
  std::size_t capture_count = 0;
  for (std::size_t index = 0; index < receipts_.size(); ++index) {
    capture_count += receipts_[index]->capture_count;
    if (status.ok()) status = update_u64(root, index);
    if (status.ok()) {
      status = update_u64(root, receipts_[index]->run_generation);
    }
    if (status.ok()) status = root.update(receipts_[index]->writer_root.bytes);
  }
  if (!status.ok()) {
    state_ = QwenBf16TapSuiteRunState::kPoisoned;
    return status;
  }
  auto writer_root = root.finalize();
  if (!writer_root.ok()) {
    state_ = QwenBf16TapSuiteRunState::kPoisoned;
    return writer_root.status();
  }
  state_ = QwenBf16TapSuiteRunState::kSealed;
  return QwenBf16TapSuiteRunReceipt{
      suite_generation_, receipts_.size(), capture_count,
      suite_plan_digest_, *writer_root};
}

}  // namespace pih
