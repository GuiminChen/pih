#include "pih/model/deepseek_hash_router.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace pih {
namespace {

std::vector<DeepSeekExpertRoute> by_ordinal(
    const DeepSeekExpertSubwavePlan& plan, std::uint32_t token) {
  std::vector<DeepSeekExpertRoute> routes;
  for (const auto& route : plan.routes()) {
    if (route.token_index == token) routes.push_back(route);
  }
  std::sort(routes.begin(), routes.end(), [](const auto& left, const auto& right) {
    return left.route_ordinal < right.route_ordinal;
  });
  return routes;
}

TEST(DeepSeekHashRouterTest, TableSelectsExpertsAndGateScoresWeightThem) {
  std::vector<std::uint16_t> table = {
      1, 3, 5, 7, 9, 11,
      2, 4, 6, 8, 10, 12,
  };
  std::vector<float> raw(256, -5.0F);
  raw[1] = 6.0F;
  auto plan = DeepSeekHashRouter::Route({0}, raw, 2, table);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  const auto routes = by_ordinal(*plan, 0);
  ASSERT_EQ(routes.size(), 6U);
  EXPECT_EQ(routes[0].expert_id, 1U);
  EXPECT_EQ(routes[5].expert_id, 11U);
  EXPECT_GT(routes[0].weight, routes[1].weight);
  float sum = 0.0F;
  for (const auto& route : routes) sum += route.weight;
  EXPECT_NEAR(sum, 1.5F, 1e-6F);
}

TEST(DeepSeekHashRouterTest, DifferentTokenIdsUseDifferentRows) {
  std::vector<std::uint16_t> table = {
      1, 3, 5, 7, 9, 11,
      2, 4, 6, 8, 10, 12,
  };
  std::vector<float> raw(512, 0.0F);
  auto plan = DeepSeekHashRouter::Route({1, 0}, raw, 2, table);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(by_ordinal(*plan, 0).front().expert_id, 2U);
  EXPECT_EQ(by_ordinal(*plan, 1).front().expert_id, 1U);
}

TEST(DeepSeekHashRouterTest, RejectsMissingIdsAndMalformedTable) {
  std::vector<float> raw(256, 0.0F);
  std::vector<std::uint16_t> table = {1, 3, 5, 7, 9, 11};
  EXPECT_FALSE(DeepSeekHashRouter::Route({1}, raw, 1, table).ok());
  table[5] = table[0];
  EXPECT_FALSE(DeepSeekHashRouter::Route({0}, raw, 1, table).ok());
  table[5] = 256;
  EXPECT_FALSE(DeepSeekHashRouter::Route({0}, raw, 1, table).ok());
}

TEST(DeepSeekHashRouterTest, RejectsNonFiniteSelectedGateScore) {
  std::vector<float> raw(256, 0.0F);
  std::vector<std::uint16_t> table = {1, 3, 5, 7, 9, 11};
  raw[7] = std::nanf("");
  EXPECT_FALSE(DeepSeekHashRouter::Route({0}, raw, 1, table).ok());
}

TEST(DeepSeekHashRouterTest, PublishesIntoPreallocatedTransactionPlan) {
  std::vector<std::uint16_t> table={1,3,5,7,9,11};
  std::vector<float> raw(256,0.0F); raw[1]=6.0F;
  std::vector<std::uint32_t> token_ids{0};
  auto scratch=DeepSeekRouteScratchArena::Create(1);
  DeepSeekPipelinePlanDescriptor descriptor{
      7,13,DeepSeekPlanPhase::kDecode,1,1};
  auto catalog=DeepSeekExpertPlanCatalog::Create(descriptor,{4,4});
  ASSERT_TRUE(scratch.ok()); ASSERT_TRUE(catalog.ok());
  ASSERT_TRUE(DeepSeekHashRouter::RouteInto(
      4,token_ids,raw,1,table,*scratch,*catalog).ok());
  auto resolved=catalog->resolve(4,descriptor); ASSERT_TRUE(resolved.ok());
  auto expected=DeepSeekHashRouter::Route(token_ids,raw,1,table);
  ASSERT_TRUE(expected.ok());
  EXPECT_EQ((*resolved)->routes(),expected->routes());
}

}  // namespace
}  // namespace pih
