#include "pih/model/deepseek_learned_router.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace pih {
namespace {

std::vector<DeepSeekExpertRoute> selection_order(
    const DeepSeekExpertSubwavePlan& plan, std::uint32_t token) {
  std::vector<DeepSeekExpertRoute> result;
  for (const auto& route : plan.routes()) {
    if (route.token_index == token) result.push_back(route);
  }
  std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
    return left.route_ordinal < right.route_ordinal;
  });
  return result;
}

TEST(DeepSeekLearnedRouterTest, SelectsTopSixAndNormalizesToScalingFactor) {
  std::vector<float> raw(256, -10.0F);
  for (std::uint16_t expert = 0; expert < 8; ++expert) raw[expert] = expert;
  std::array<float, 256> bias{};
  auto plan = DeepSeekLearnedRouter::Route(1, raw, bias);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  const auto selected = selection_order(*plan, 0);
  ASSERT_EQ(selected.size(), 6U);
  EXPECT_EQ(selected[0].expert_id, 7U);
  EXPECT_EQ(selected[1].expert_id, 6U);
  EXPECT_EQ(selected[5].expert_id, 2U);
  float sum = 0.0F;
  for (const auto& route : selected) sum += route.weight;
  EXPECT_NEAR(sum, 1.5F, 1e-6F);
}

TEST(DeepSeekLearnedRouterTest, BiasChangesSelectionButNotBaseWeightFormula) {
  std::vector<float> raw(256, -20.0F);
  for (std::uint16_t expert = 0; expert < 6; ++expert) raw[expert] = 2.0F;
  raw[9] = -2.0F;
  std::array<float, 256> bias{};
  bias[9] = 100.0F;
  auto plan = DeepSeekLearnedRouter::Route(1, raw, bias);
  ASSERT_TRUE(plan.ok());
  const auto selected = selection_order(*plan, 0);
  EXPECT_EQ(selected[0].expert_id, 9U);
  const auto promoted = selected[0].weight;
  const auto regular = selected[1].weight;
  EXPECT_LT(promoted, regular);
}

TEST(DeepSeekLearnedRouterTest, ExactTiePrefersSmallerExpertId) {
  std::vector<float> raw(256, 0.0F);
  std::array<float, 256> bias{};
  auto plan = DeepSeekLearnedRouter::Route(1, raw, bias);
  ASSERT_TRUE(plan.ok());
  const auto selected = selection_order(*plan, 0);
  for (std::uint16_t ordinal = 0; ordinal < 6; ++ordinal) {
    EXPECT_EQ(selected[ordinal].expert_id, ordinal);
  }
}

TEST(DeepSeekLearnedRouterTest, SelectsAcrossAllLegacyExpertGroups) {
  // Flash-0731's sqrt(softplus) router is ungrouped.  Place the six best
  // candidates in six different 32-expert legacy groups: a grouped-top-k
  // implementation with a four-group limit would silently drop two of them.
  std::vector<float> raw(256, -20.0F);
  constexpr std::array<std::uint16_t, 6> experts{0, 32, 64, 96, 128, 160};
  for (std::uint16_t ordinal = 0; ordinal < experts.size(); ++ordinal) {
    raw[experts[ordinal]] = 16.0F - static_cast<float>(ordinal);
  }
  std::array<float, 256> bias{};
  auto plan = DeepSeekLearnedRouter::Route(1, raw, bias);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  const auto selected = selection_order(*plan, 0);
  ASSERT_EQ(selected.size(), experts.size());
  for (std::uint16_t ordinal = 0; ordinal < experts.size(); ++ordinal) {
    EXPECT_EQ(selected[ordinal].expert_id, experts[ordinal]);
  }
}

TEST(DeepSeekLearnedRouterTest, RoutesEachTokenIndependently) {
  std::vector<float> raw(512, -10.0F);
  for (std::uint16_t expert = 0; expert < 6; ++expert) {
    raw[expert] = 10.0F - expert;
    raw[256 + 100 + expert] = 10.0F - expert;
  }
  std::array<float, 256> bias{};
  auto plan = DeepSeekLearnedRouter::Route(2, raw, bias);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(selection_order(*plan, 0).front().expert_id, 0U);
  EXPECT_EQ(selection_order(*plan, 1).front().expert_id, 100U);
}

TEST(DeepSeekLearnedRouterTest, RejectsShapeAndNonFiniteInputs) {
  std::array<float, 256> bias{};
  EXPECT_FALSE(DeepSeekLearnedRouter::Route(1, std::vector<float>(255), bias).ok());
  auto raw = std::vector<float>(256, 0.0F);
  raw[17] = std::nanf("");
  EXPECT_FALSE(DeepSeekLearnedRouter::Route(1, raw, bias).ok());
  raw[17] = 0.0F;
  bias[3] = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(DeepSeekLearnedRouter::Route(1, raw, bias).ok());
}

TEST(DeepSeekLearnedRouterTest, PublishesIntoPreallocatedTransactionPlan) {
  std::vector<float> raw(256,-10.0F);
  for (std::uint16_t expert=0;expert<8;++expert) raw[expert]=expert;
  std::array<float,256> bias{};
  auto scratch=DeepSeekRouteScratchArena::Create(1);
  DeepSeekPipelinePlanDescriptor descriptor{
      7,13,DeepSeekPlanPhase::kDecode,1,1};
  auto catalog=DeepSeekExpertPlanCatalog::Create(descriptor,{4,4});
  ASSERT_TRUE(scratch.ok()); ASSERT_TRUE(catalog.ok());
  ASSERT_TRUE(DeepSeekLearnedRouter::RouteInto(
      4,1,raw,bias,*scratch,*catalog).ok());
  auto resolved=catalog->resolve(4,descriptor); ASSERT_TRUE(resolved.ok());
  auto expected=DeepSeekLearnedRouter::Route(1,raw,bias); ASSERT_TRUE(expected.ok());
  EXPECT_EQ((*resolved)->routes(),expected->routes());
  EXPECT_FALSE(DeepSeekLearnedRouter::RouteInto(
      4,1,raw,bias,*scratch,*catalog).ok());
}

}  // namespace
}  // namespace pih
