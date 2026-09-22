#include "pih/model/deepseek_expert_accumulator_oracle.h"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace pih {
namespace {

std::vector<float> constant_rows(std::size_t rows, float value) {
  return std::vector<float>(
      rows * DeepSeekExpertAccumulatorOracle::kHiddenSize, value);
}

TEST(DeepSeekExpertAccumulatorOracleTest, ReducesExpertsThenAddsSharedOnce) {
  auto oracle = DeepSeekExpertAccumulatorOracle::Create(2);
  ASSERT_TRUE(oracle.ok());
  const std::vector<DeepSeekExpertRoute> expert_one = {
      {1, 0, 0, 0.25F}, {1, 1, 2, 0.5F}};
  const std::vector<DeepSeekExpertRoute> expert_three = {
      {3, 0, 1, 0.75F}};
  auto expert_one_output = constant_rows(2, 0.5F);
  std::fill(expert_one_output.begin() + 4096, expert_one_output.end(), 1.0F);
  ASSERT_TRUE(oracle->add_expert(1, expert_one, expert_one_output).ok());
  ASSERT_TRUE(oracle->add_expert(3, expert_three,
                                 constant_rows(1, 3.0F)).ok());
  auto result = oracle->finalize(constant_rows(2, 1.0F));
  ASSERT_TRUE(result.ok());
  EXPECT_FLOAT_EQ((*result)[0], 4.5F);
  EXPECT_FLOAT_EQ((*result)[4096], 2.0F);
}

TEST(DeepSeekExpertAccumulatorOracleTest, DoesNotApplyRouteWeightAfterW2) {
  auto oracle = DeepSeekExpertAccumulatorOracle::Create(1);
  ASSERT_TRUE(oracle.ok());
  const std::vector<DeepSeekExpertRoute> route = {{1, 0, 0, 0.25F}};
  ASSERT_TRUE(oracle->add_expert(1, route, constant_rows(1, 2.0F)).ok());
  auto result = oracle->finalize(constant_rows(1, 1.0F));
  ASSERT_TRUE(result.ok());
  EXPECT_FLOAT_EQ((*result)[0], 3.0F);
}

TEST(DeepSeekExpertAccumulatorOracleTest, RejectsExpertOrderRegression) {
  auto oracle = DeepSeekExpertAccumulatorOracle::Create(1);
  ASSERT_TRUE(oracle.ok());
  const std::vector<DeepSeekExpertRoute> route = {{4, 0, 0, 1.0F}};
  ASSERT_TRUE(oracle->add_expert(4, route, constant_rows(1, 1.0F)).ok());
  const std::vector<DeepSeekExpertRoute> earlier = {{3, 0, 0, 1.0F}};
  EXPECT_FALSE(oracle->add_expert(3, earlier,
                                  constant_rows(1, 1.0F)).ok());
}

TEST(DeepSeekExpertAccumulatorOracleTest, RejectsDuplicateTokenWithinExpert) {
  auto oracle = DeepSeekExpertAccumulatorOracle::Create(1);
  ASSERT_TRUE(oracle.ok());
  const std::vector<DeepSeekExpertRoute> routes = {
      {4, 0, 0, 0.5F}, {4, 0, 1, 0.5F}};
  EXPECT_FALSE(oracle->add_expert(4, routes,
                                  constant_rows(2, 1.0F)).ok());
}

TEST(DeepSeekExpertAccumulatorOracleTest, FinalizesExactlyOnce) {
  auto oracle = DeepSeekExpertAccumulatorOracle::Create(1);
  ASSERT_TRUE(oracle.ok());
  auto shared = constant_rows(1, 1.0F);
  EXPECT_TRUE(oracle->finalize(shared).ok());
  EXPECT_FALSE(oracle->finalize(shared).ok());
}

TEST(DeepSeekExpertAccumulatorOracleTest, FailedExpertDoesNotPartiallyPublish) {
  auto oracle = DeepSeekExpertAccumulatorOracle::Create(1);
  ASSERT_TRUE(oracle.ok());
  const std::vector<DeepSeekExpertRoute> route = {{4, 0, 0, 1.0F}};
  auto output = constant_rows(1, 2.0F);
  output.back() = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(oracle->add_expert(4, route, output).ok());
  for (const auto value : oracle->accumulator()) EXPECT_EQ(value, 0.0F);
  output.back() = 2.0F;
  EXPECT_TRUE(oracle->add_expert(4, route, output).ok());
}

}  // namespace
}  // namespace pih
