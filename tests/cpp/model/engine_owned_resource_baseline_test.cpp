#include "pih/model/engine_owned_resource_baseline.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest resource_digest(std::uint8_t value) {
  Sha256Digest result{}; result.bytes.fill(static_cast<std::byte>(value));
  return result;
}
std::vector<EngineOwnedResourceBaselineEntry> baselines() {
  std::vector<EngineOwnedResourceBaselineEntry> values;
  for (std::uint8_t kind = 0; kind < 6; ++kind)
    values.push_back({static_cast<EngineOwnedResourceKind>(kind),
                      resource_digest(kind + 1)});
  return values;
}
EngineOwnedResourceBaselineReceipt receipt() {
  EngineOwnedResourceBaselineReceipt value{
      7, resource_digest(9), 1, 100, 101, {}};
  for (const auto& baseline : baselines())
    value.resources.push_back(
        {baseline.kind, true, baseline.baseline_digest});
  return value;
}

TEST(EngineOwnedResourceBaselineTest, DistinguishesUnknownDriftAndBaseline) {
  auto gate = EngineOwnedResourceBaselineGate::Create(
      7, resource_digest(9), baselines()).value();
  auto unknown = receipt(); unknown.resources[1].visibility_complete = false;
  ASSERT_EQ(*gate.accept(unknown), EngineOwnedResourceBaselineState::kUnknown);
  auto drift = receipt(); drift.sample_identity = 2;
  drift.sample_started_ns = 101; drift.sample_completed_ns = 102;
  drift.resources[2].observed_digest = resource_digest(99);
  ASSERT_EQ(*gate.accept(drift), EngineOwnedResourceBaselineState::kDrifted);
  auto clean = receipt(); clean.sample_identity = 3;
  clean.sample_started_ns = 102; clean.sample_completed_ns = 103;
  ASSERT_EQ(*gate.accept(clean), EngineOwnedResourceBaselineState::kBaseline);
}

TEST(EngineOwnedResourceBaselineTest, RejectsInvalidBaselineManifest) {
  EXPECT_FALSE(EngineOwnedResourceBaselineGate::Create(
      0, resource_digest(9), baselines()).ok());
  EXPECT_FALSE(EngineOwnedResourceBaselineGate::Create(
      7, {}, baselines()).ok());
  auto missing = baselines(); missing.pop_back();
  EXPECT_FALSE(EngineOwnedResourceBaselineGate::Create(
      7, resource_digest(9), missing).ok());
  auto duplicate = baselines(); duplicate[5].kind = duplicate[4].kind;
  EXPECT_FALSE(EngineOwnedResourceBaselineGate::Create(
      7, resource_digest(9), duplicate).ok());
  auto zero = baselines(); zero[0].baseline_digest = {};
  EXPECT_FALSE(EngineOwnedResourceBaselineGate::Create(
      7, resource_digest(9), zero).ok());
}

TEST(EngineOwnedResourceBaselineTest, ReceiptIdentityCoverageAndTimePoison) {
  for (int mutation = 0; mutation < 7; ++mutation) {
    auto gate = EngineOwnedResourceBaselineGate::Create(
        7, resource_digest(9), baselines()).value();
    auto value = receipt();
    if (mutation == 0) value.engine_generation = 8;
    if (mutation == 1) value.allocation_lease_digest = resource_digest(8);
    if (mutation == 2) value.sample_identity = 2;
    if (mutation == 3) value.sample_completed_ns = 99;
    if (mutation == 4) value.resources.pop_back();
    if (mutation == 5) value.resources[1].kind = value.resources[0].kind;
    if (mutation == 6) value.resources[0].observed_digest = {};
    EXPECT_FALSE(gate.accept(value).ok()) << mutation;
    EXPECT_TRUE(gate.poisoned()) << mutation;
  }
}

}  // namespace
}  // namespace pih
