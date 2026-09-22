#include "pih/model/engine_owned_resource_census_provider.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

class CensusSource final : public EngineOwnedResourceCensusSource {
 public:
  Result<EngineOwnedResourceCensus> capture() override {
    ++calls;
    return result;
  }

  int calls = 0;
  Result<EngineOwnedResourceCensus> result = EngineOwnedResourceCensus{};
};

TEST(EngineOwnedResourceCensusProviderTest, DigestsCapturedOwnedRecords) {
  CensusSource source;
  source.result = EngineOwnedResourceCensus{
      {std::byte{'b'}}, {std::byte{'a'}}};
  auto provider = EngineOwnedResourceCensusProvider::Create(
      EngineOwnedResourceKind::kListener, source).value();

  auto observed = provider.observe();
  ASSERT_TRUE(observed.ok());
  const std::array records{
      EngineOwnedResourceCensusRecord{source.result->at(1)},
      EngineOwnedResourceCensusRecord{source.result->at(0)}};
  EXPECT_EQ(*observed, *engine_owned_resource_census_digest(
                           EngineOwnedResourceKind::kListener, records));
  EXPECT_EQ(source.calls, 1);
}

TEST(EngineOwnedResourceCensusProviderTest, PreservesSourceFailureAndRecovers) {
  CensusSource source;
  source.result = Status::Unavailable("listener namespace is unavailable");
  auto provider = EngineOwnedResourceCensusProvider::Create(
      EngineOwnedResourceKind::kListener, source).value();

  auto unavailable = provider.observe();
  ASSERT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), StatusCode::kUnavailable);
  source.result = EngineOwnedResourceCensus{{std::byte{'x'}}};
  EXPECT_TRUE(provider.observe().ok());
  EXPECT_EQ(source.calls, 2);
}

TEST(EngineOwnedResourceCensusProviderTest, PreservesCanonicalInputRejection) {
  CensusSource source;
  source.result = EngineOwnedResourceCensus{{}, {std::byte{'x'}}};
  auto provider = EngineOwnedResourceCensusProvider::Create(
      EngineOwnedResourceKind::kShm, source).value();

  auto observed = provider.observe();
  ASSERT_FALSE(observed.ok());
  EXPECT_EQ(observed.status().code(), StatusCode::kInvalidArgument);
}

TEST(EngineOwnedResourceCensusProviderTest, RejectsInvalidKindBeforeCapture) {
  CensusSource source;
  auto provider = EngineOwnedResourceCensusProvider::Create(
      static_cast<EngineOwnedResourceKind>(6), source);
  EXPECT_FALSE(provider.ok());
  EXPECT_EQ(source.calls, 0);
}

}  // namespace
}  // namespace pih
