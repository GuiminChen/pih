#include "pih/backend/cuda/deepseek_indexer_projection.h"
#include "pih/model/deepseek_indexer_projection_oracle.h"

#include <gtest/gtest.h>

#include <cmath>

namespace pih { namespace {

TEST(DeepSeekIndexerProjectionTest, ValidatesExactProductionGeometry) {
  DeepSeekIndexerProjectionLaunch launch{
      1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 8, 9, 10, 2, 1024};
  EXPECT_TRUE(validate_deepseek_indexer_projection_launch(launch).ok());
  launch.token_count = 0;
  EXPECT_FALSE(validate_deepseek_indexer_projection_launch(launch).ok());
  launch.token_count = 2;
  launch.query_bf16 = launch.qr_bf16;
  EXPECT_FALSE(validate_deepseek_indexer_projection_launch(launch).ok());
}

TEST(DeepSeekIndexerProjectionTest,
     RejectsMissingScaleAndAliasedQuantizationScratch) {
  const DeepSeekIndexerProjectionLaunch valid{
      1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 8, 9, 10, 2, 1024};
  auto launch = valid;
  launch.wq_b_scale_bits = 0;
  EXPECT_FALSE(validate_deepseek_indexer_projection_launch(launch).ok());
  launch = valid;
  launch.qr_e4m3 = launch.qr_bf16;
  EXPECT_FALSE(validate_deepseek_indexer_projection_launch(launch).ok());
  launch = valid;
  launch.qr_scale_bits = launch.qr_e4m3;
  EXPECT_FALSE(validate_deepseek_indexer_projection_launch(launch).ok());
  launch = valid;
  launch.query_bf16 = launch.wq_b_e4m3;
  EXPECT_FALSE(validate_deepseek_indexer_projection_launch(launch).ok());
}

TEST(DeepSeekIndexerProjectionTest,
     OracleProjectsWeightsAndRotatesOnlyTrailingDimensions) {
  std::vector<BFloat16> qr(1024, BFloat16::FromFloat(0.0F));
  std::vector<BFloat16> hidden(4096, BFloat16::FromFloat(0.0F));
  std::vector<BFloat16> wq(8192U * 1024U, BFloat16::FromFloat(0.0F));
  std::vector<BFloat16> weights(64U * 4096U,
                                BFloat16::FromFloat(0.0F));
  std::vector<float> frequencies(2U * 64U, 0.0F);
  qr[0] = BFloat16::FromFloat(2.0F);
  hidden[0] = BFloat16::FromFloat(3.0F);
  wq[0] = BFloat16::FromFloat(4.0F);
  wq[64U * 1024U] = BFloat16::FromFloat(5.0F);
  wq[65U * 1024U] = BFloat16::FromFloat(6.0F);
  weights[0] = BFloat16::FromFloat(7.0F);
  frequencies[64] = 0.0F;
  frequencies[64 + 32] = 1.0F;
  const std::vector<std::uint32_t> positions{1};
  auto result = DeepSeekIndexerProjectionOracle::Evaluate(
      qr, hidden, wq, weights, frequencies, positions, 2);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_FLOAT_EQ(result->query[0].to_float(), 8.0F);
  EXPECT_FLOAT_EQ(result->query[64].to_float(), -12.0F);
  EXPECT_FLOAT_EQ(result->query[65].to_float(), 10.0F);
  EXPECT_NEAR(result->head_weight[0], 21.0F / std::sqrt(8192.0F), 1.0e-6F);
}

} }  // namespace pih
