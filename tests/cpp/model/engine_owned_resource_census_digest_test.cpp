#include "pih/model/engine_owned_resource_census_digest.h"

#include <array>
#include <vector>
#include <gtest/gtest.h>

namespace pih {
namespace {

EngineOwnedResourceCensusRecord census_record(std::string_view value) {
  return {std::as_bytes(std::span(value))};
}

TEST(EngineOwnedResourceCensusDigestTest, IsOrderIndependentAndStable) {
  const std::array first{census_record("owner-b"), census_record("owner-a")};
  const std::array second{census_record("owner-a"), census_record("owner-b")};

  auto left = engine_owned_resource_census_digest(
      EngineOwnedResourceKind::kShm, first);
  auto right = engine_owned_resource_census_digest(
      EngineOwnedResourceKind::kShm, second);
  ASSERT_TRUE(left.ok());
  ASSERT_TRUE(right.ok());
  EXPECT_EQ(*left, *right);
  EXPECT_EQ(left->hex(),
            "2f69a443f4f7f9f62638f584bacc3d0bf7deb6285f2e3a4cf0bc70f14d2bbbd2");
}

TEST(EngineOwnedResourceCensusDigestTest, SeparatesKindsAndRecordBoundaries) {
  const std::array joined{census_record("ab")};
  const std::array split{census_record("a"), census_record("b")};
  auto shm = engine_owned_resource_census_digest(
      EngineOwnedResourceKind::kShm, joined).value();
  auto network = engine_owned_resource_census_digest(
      EngineOwnedResourceKind::kNetwork, joined).value();
  auto two_records = engine_owned_resource_census_digest(
      EngineOwnedResourceKind::kShm, split).value();

  EXPECT_NE(shm, network);
  EXPECT_NE(shm, two_records);
}

TEST(EngineOwnedResourceCensusDigestTest, RepresentsAnEmptyVisibleCensus) {
  const std::array<EngineOwnedResourceCensusRecord, 0> empty{};
  auto digest = engine_owned_resource_census_digest(
      EngineOwnedResourceKind::kArtifact, empty);
  ASSERT_TRUE(digest.ok());
  EXPECT_NE(*digest, Sha256Digest{});
}

TEST(EngineOwnedResourceCensusDigestTest, RejectsInvalidAmbiguousInput) {
  const std::array empty_record{EngineOwnedResourceCensusRecord{}};
  EXPECT_FALSE(engine_owned_resource_census_digest(
                   EngineOwnedResourceKind::kShm, empty_record).ok());

  const std::array duplicate{census_record("same"), census_record("same")};
  EXPECT_FALSE(engine_owned_resource_census_digest(
                   EngineOwnedResourceKind::kShm, duplicate).ok());
  EXPECT_FALSE(engine_owned_resource_census_digest(
                   static_cast<EngineOwnedResourceKind>(6), duplicate).ok());

  const std::vector<std::byte> oversized(64U * 1024U + 1U);
  const std::array oversized_record{
      EngineOwnedResourceCensusRecord{oversized}};
  auto exhausted = engine_owned_resource_census_digest(
      EngineOwnedResourceKind::kShm, oversized_record);
  ASSERT_FALSE(exhausted.ok());
  EXPECT_EQ(exhausted.status().code(), StatusCode::kResourceExhausted);
}

}  // namespace
}  // namespace pih
