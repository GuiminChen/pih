#include "pih/model/deepseek_expert_subwave_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace pih {
namespace {

std::vector<DeepSeekExpertRoute> two_token_routes() {
  return {
      {9, 1, 2, 0.12F}, {3, 0, 5, 0.15F}, {7, 1, 0, 0.10F},
      {1, 0, 2, 0.20F}, {5, 1, 4, 0.11F}, {9, 0, 0, 0.25F},
      {2, 1, 5, 0.13F}, {5, 0, 1, 0.10F}, {1, 1, 3, 0.30F},
      {7, 0, 4, 0.12F}, {3, 1, 1, 0.24F}, {2, 0, 3, 0.18F},
  };
}

TEST(DeepSeekExpertSubwavePlanTest, SortsCanonicalExpertReductionOrder) {
  auto plan = DeepSeekExpertSubwavePlan::Create(2, two_token_routes());
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  ASSERT_EQ(plan->routes().size(), 12U);
  for (std::size_t i = 1; i < plan->routes().size(); ++i) {
    const auto& previous = plan->routes()[i - 1];
    const auto& current = plan->routes()[i];
    EXPECT_TRUE(previous.expert_id < current.expert_id ||
                (previous.expert_id == current.expert_id &&
                 previous.token_index < current.token_index));
  }
  EXPECT_EQ(plan->expert_offsets()[0], 0U);
  EXPECT_EQ(plan->expert_offsets()[1], 0U);
  EXPECT_EQ(plan->expert_offsets()[2], 2U);
  EXPECT_EQ(plan->expert_offsets()[3], 4U);
  EXPECT_EQ(plan->expert_offsets()[10], 12U);
  EXPECT_EQ(plan->expert_offsets()[256], 12U);
  EXPECT_EQ(plan->workspace_bytes(), 34240U);
}

TEST(DeepSeekExpertSubwavePlanTest, IsIndependentOfInputRouteOrder) {
  auto forward = two_token_routes();
  auto reverse = forward;
  std::reverse(reverse.begin(), reverse.end());
  auto first = DeepSeekExpertSubwavePlan::Create(2, std::move(forward));
  auto second = DeepSeekExpertSubwavePlan::Create(2, std::move(reverse));
  ASSERT_TRUE(first.ok()); ASSERT_TRUE(second.ok());
  ASSERT_EQ(first->routes().size(), second->routes().size());
  for (std::size_t i = 0; i < first->routes().size(); ++i) {
    EXPECT_EQ(first->routes()[i].expert_id, second->routes()[i].expert_id);
    EXPECT_EQ(first->routes()[i].token_index, second->routes()[i].token_index);
    EXPECT_EQ(first->routes()[i].route_ordinal,
              second->routes()[i].route_ordinal);
    EXPECT_FLOAT_EQ(first->routes()[i].weight, second->routes()[i].weight);
  }
}

TEST(DeepSeekExpertSubwavePlanTest, RejectsDuplicateExpertForToken) {
  auto routes = two_token_routes();
  routes[1].expert_id = routes[3].expert_id;
  EXPECT_FALSE(DeepSeekExpertSubwavePlan::Create(2, std::move(routes)).ok());
}

TEST(DeepSeekExpertSubwavePlanTest, RejectsDuplicateOrMissingOrdinal) {
  auto routes = two_token_routes();
  routes[1].route_ordinal = routes[3].route_ordinal;
  EXPECT_FALSE(DeepSeekExpertSubwavePlan::Create(2, std::move(routes)).ok());
}

TEST(DeepSeekExpertSubwavePlanTest, RejectsMalformedRouteTable) {
  auto routes = two_token_routes();
  routes.pop_back();
  EXPECT_FALSE(DeepSeekExpertSubwavePlan::Create(2, std::move(routes)).ok());
  routes = two_token_routes();
  routes[0].weight = std::nanf("");
  EXPECT_FALSE(DeepSeekExpertSubwavePlan::Create(2, std::move(routes)).ok());
  routes = two_token_routes();
  routes[0].expert_id = 256;
  EXPECT_FALSE(DeepSeekExpertSubwavePlan::Create(2, std::move(routes)).ok());
}

TEST(DeepSeekExpertSubwavePlanTest, ReusesReservedRouteBacking) {
  auto plan = DeepSeekExpertSubwavePlan::Reserve(2);
  ASSERT_TRUE(plan.ok());
  auto routes = two_token_routes();
  ASSERT_TRUE(plan->materialize(2, routes).ok());
  const auto* backing = plan->routes().data();
  std::reverse(routes.begin(), routes.end());
  ASSERT_TRUE(plan->materialize(2, routes).ok());
  EXPECT_EQ(plan->routes().data(), backing);
  EXPECT_EQ(plan->maximum_token_count(), 2U);
  EXPECT_EQ(plan->workspace_bytes(), 34240U);
  EXPECT_FALSE(plan->materialize(3, routes).ok());
}

TEST(DeepSeekExpertSubwavePlanTest, InvalidRematerializationCannotReuseOldPlan) {
  auto plan = DeepSeekExpertSubwavePlan::Reserve(2);
  auto routes = two_token_routes();
  ASSERT_TRUE(plan.ok());
  ASSERT_TRUE(plan->materialize(2, routes).ok());
  routes[0].weight = -1.0F;
  EXPECT_FALSE(plan->materialize(2, routes).ok());
  EXPECT_EQ(plan->token_count(), 0U);
  EXPECT_EQ(plan->workspace_bytes(), 0U);
  EXPECT_EQ(plan->expert_offsets()[256], 0U);
}

}  // namespace
}  // namespace pih
