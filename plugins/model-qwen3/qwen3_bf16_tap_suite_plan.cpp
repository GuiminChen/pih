#include "pih/model/qwen3_bf16_tap_suite_plan.h"

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

Result<QwenNumericalTapPlan> activation_plan() {
  std::vector<QwenNumericalTapRequest> requests;
  requests.reserve(145);
  for (const std::uint32_t layer : {0U, 13U, 27U}) {
    requests.push_back(
        {QwenNumericalTapPoint::kLayerHidden, layer, 0, 1});
  }
  for (std::uint32_t layer = 0; layer < 28; ++layer) {
    for (const auto point : {QwenNumericalTapPoint::kQueryAfterNorm,
                             QwenNumericalTapPoint::kKeyAfterNorm,
                             QwenNumericalTapPoint::kQueryAfterRope,
                             QwenNumericalTapPoint::kKeyAfterRope,
                             QwenNumericalTapPoint::kPrefillAttention}) {
      requests.push_back({point, layer, 0, 1});
    }
  }
  requests.push_back({QwenNumericalTapPoint::kFinalNorm, 28, 0, 1});
  requests.push_back({QwenNumericalTapPoint::kLogits, 28, 0, 1});
  return QwenNumericalTapPlan::Create(
      requests, QwenNumericalTapPlan::kMaximumArenaBytes);
}

Result<QwenNumericalTapPlan> kv_plan(std::uint32_t position) {
  std::vector<QwenNumericalTapRequest> requests;
  requests.reserve(56);
  for (std::uint32_t layer = 0; layer < 28; ++layer) {
    requests.push_back({QwenNumericalTapPoint::kKvKey, layer, position, 1});
    requests.push_back(
        {QwenNumericalTapPoint::kKvValue, layer, position, 1});
  }
  return QwenNumericalTapPlan::Create(
      requests, QwenNumericalTapPlan::kMaximumArenaBytes);
}

}  // namespace

Result<QwenBf16TapSuitePlan> QwenBf16TapSuitePlan::Create() {
  std::vector<QwenBf16TapFixturePlan> fixtures;
  fixtures.reserve(kFixtureCount);
  auto activations = activation_plan();
  if (!activations.ok()) return activations.status();
  fixtures.push_back({QwenBf16TapFixtureKind::kPositionZeroActivations, 0,
                      std::move(*activations)});
  constexpr std::array positions{2U, 17U, 129U, 4097U};
  constexpr std::array kinds{QwenBf16TapFixtureKind::kKvPosition2,
                             QwenBf16TapFixtureKind::kKvPosition17,
                             QwenBf16TapFixtureKind::kKvPosition129,
                             QwenBf16TapFixtureKind::kKvPosition4097};
  for (std::size_t index = 0; index < positions.size(); ++index) {
    auto taps = kv_plan(positions[index]);
    if (!taps.ok()) return taps.status();
    fixtures.push_back({kinds[index], positions[index], std::move(*taps)});
  }
  return QwenBf16TapSuitePlan(std::move(fixtures));
}

Result<Sha256Digest> QwenBf16TapSuitePlan::semantic_digest() const {
  Sha256 digest;
  constexpr std::string_view domain =
      "pih.qwen_bf16_tap_suite_plan.v1";
  Status status = digest.update(std::as_bytes(std::span(domain)));
  if (status.ok()) status = update_u64(digest, fixtures_.size());
  for (const auto& fixture : fixtures_) {
    auto plan_digest = fixture.taps.semantic_digest();
    if (!plan_digest.ok()) return plan_digest.status();
    if (status.ok()) {
      status = update_u64(digest, static_cast<std::uint8_t>(fixture.kind));
    }
    if (status.ok()) status = update_u64(digest, fixture.first_position);
    if (status.ok()) status = digest.update(plan_digest->bytes);
  }
  if (!status.ok()) return status;
  return digest.finalize();
}

Status validate_qwen_bf16_tap_suite_coverage(
    const QwenBf16TapSuitePlan& suite) {
  if (suite.size() != QwenBf16TapSuitePlan::kFixtureCount) {
    return Status::FailedPrecondition(
        "Qwen BF16 tap suite fixture count is incomplete");
  }
  std::vector<QwenNumericalTapRequest> requests;
  requests.reserve(QwenBf16TapSuitePlan::kCaptureCount);
  for (const auto& fixture : suite) {
    for (const auto& capture : fixture.taps.captures()) {
      requests.push_back(capture.request);
    }
  }
  if (requests.size() != QwenBf16TapSuitePlan::kCaptureCount) {
    return Status::FailedPrecondition(
        "Qwen BF16 tap suite capture count is incomplete");
  }
  auto aggregate = QwenNumericalTapPlan::Create(
      requests, QwenNumericalTapPlan::kMaximumArenaBytes);
  if (!aggregate.ok()) return aggregate.status();
  return validate_qwen_bf16_tap_coverage(*aggregate);
}

}  // namespace pih
