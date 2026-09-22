#include "pih/model/deepseek_bound_expert_plan_provider.h"
#include "pih/model/deepseek_hash_router.h"
#include "pih/model/deepseek_route_scratch_arena.h"

#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace pih {
namespace {

std::vector<DeepSeekExpertRoute> routes(std::uint32_t tokens,
                                        std::uint16_t expert_base) {
  std::vector<DeepSeekExpertRoute> result;
  result.reserve(tokens * DeepSeekExpertSubwavePlan::kRoutesPerToken);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint8_t ordinal = 0;
         ordinal < DeepSeekExpertSubwavePlan::kRoutesPerToken; ++ordinal) {
      result.push_back({static_cast<std::uint16_t>(expert_base + ordinal),
                        token, ordinal, 1.0F / 6.0F});
    }
  }
  return result;
}

TEST(DeepSeekBoundExpertPlanProviderTest,
     PublishesCompleteLayerSetForExactPlanIdentity) {
  auto provider = DeepSeekBoundExpertPlanProvider::Create({3, 4}, 8);
  ASSERT_TRUE(provider.ok());
  auto layer3 = routes(2, 0);
  auto layer4 = routes(2, 10);
  const std::array<DeepSeekLayerExpertRoutes, 2> layers{{
      {4, layer4}, {3, layer3}}};
  const DeepSeekPipelinePlanDescriptor descriptor{
      7, 9, DeepSeekPlanPhase::kDecode, 2, 2};
  ASSERT_TRUE(provider->bind(descriptor, layers).ok());

  auto resolved3 = provider->resolve(3, descriptor);
  auto resolved4 = provider->resolve(4, descriptor);
  ASSERT_TRUE(resolved3.ok());
  ASSERT_TRUE(resolved4.ok());
  EXPECT_EQ((*resolved3)->token_count(), 2U);
  EXPECT_EQ((*resolved4)->routes().front().expert_id, 10U);
  auto stale = descriptor;
  ++stale.plan_sequence;
  EXPECT_EQ(provider->resolve(3, stale).status().code(),
            StatusCode::kFailedPrecondition);
  EXPECT_EQ(provider->resolve(2, descriptor).status().code(),
            StatusCode::kFailedPrecondition);
}

TEST(DeepSeekBoundExpertPlanProviderTest,
     RejectsSyntheticDsparkLayerIdentity) {
  EXPECT_FALSE(DeepSeekBoundExpertPlanProvider::Create({43, 43}, 8).ok());
}

TEST(DeepSeekBoundExpertPlanProviderTest,
     FailedRebindDoesNotPublishPartialNewBank) {
  auto provider = DeepSeekBoundExpertPlanProvider::Create({3, 4}, 8);
  ASSERT_TRUE(provider.ok());
  auto layer3 = routes(1, 0);
  auto layer4 = routes(1, 10);
  std::array<DeepSeekLayerExpertRoutes, 2> valid{{
      {3, layer3}, {4, layer4}}};
  const DeepSeekPipelinePlanDescriptor first{
      7, 9, DeepSeekPlanPhase::kDecode, 1, 1};
  ASSERT_TRUE(provider->bind(first, valid).ok());

  auto invalid_routes = routes(2, 20);
  invalid_routes.pop_back();
  std::array<DeepSeekLayerExpertRoutes, 2> invalid{{
      {3, routes(2, 0)}, {4, invalid_routes}}};
  const DeepSeekPipelinePlanDescriptor second{
      7, 10, DeepSeekPlanPhase::kDecode, 2, 1};
  EXPECT_FALSE(provider->bind(second, invalid).ok());
  EXPECT_TRUE(provider->resolve(3, first).ok());
  EXPECT_FALSE(provider->resolve(3, second).ok());
}

TEST(DeepSeekBoundExpertPlanProviderTest,
     RejectsMissingDuplicateAndForeignLayers) {
  auto provider = DeepSeekBoundExpertPlanProvider::Create({3, 4}, 4);
  ASSERT_TRUE(provider.ok());
  auto work = routes(1, 0);
  const DeepSeekPipelinePlanDescriptor descriptor{
      7, 9, DeepSeekPlanPhase::kDecode, 1, 1};
  std::array<DeepSeekLayerExpertRoutes, 1> missing{{{3, work}}};
  std::array<DeepSeekLayerExpertRoutes, 2> duplicate{{
      {3, work}, {3, work}}};
  std::array<DeepSeekLayerExpertRoutes, 2> foreign{{
      {3, work}, {5, work}}};
  EXPECT_FALSE(provider->bind(descriptor, missing).ok());
  EXPECT_FALSE(provider->bind(descriptor, duplicate).ok());
  EXPECT_FALSE(provider->bind(descriptor, foreign).ok());
}

TEST(DeepSeekBoundExpertPlanProviderTest,
     HashRouterPublishesAndRoutedConsumerResolvesSameStore) {
  auto provider = DeepSeekBoundExpertPlanProvider::Create({0, 1}, 4);
  auto scratch = DeepSeekRouteScratchArena::Create(4);
  ASSERT_TRUE(provider.ok()); ASSERT_TRUE(scratch.ok());
  const DeepSeekPipelinePlanDescriptor descriptor{
      7, 9, DeepSeekPlanPhase::kDecode, 1, 1};
  ASSERT_TRUE(provider->begin(descriptor).ok());
  std::vector<std::uint32_t> tokens{1};
  std::vector<float> scores(256);
  std::vector<std::uint16_t> table(12);
  for (std::uint16_t ordinal = 0; ordinal < 6; ++ordinal) {
    table[ordinal] = ordinal;
    table[6 + ordinal] = static_cast<std::uint16_t>(10 + ordinal);
  }
  ASSERT_TRUE(DeepSeekHashRouter::RouteInto(
      0, tokens, scores, 2, table, *scratch, *provider).ok());
  auto resolved = provider->resolve(0, descriptor);
  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ((*resolved)->routes().size(), 6U);
  EXPECT_FALSE(provider->resolve(1, descriptor).ok());
  EXPECT_EQ(DeepSeekHashRouter::RouteInto(
      0, tokens, scores, 2, table, *scratch, *provider).code(),
      StatusCode::kFailedPrecondition);

  auto next = descriptor;
  ++next.plan_sequence;
  ASSERT_TRUE(provider->begin(next).ok());
  EXPECT_FALSE(provider->resolve(0, descriptor).ok());
  EXPECT_FALSE(provider->resolve(0, next).ok());
}

}  // namespace
}  // namespace pih
