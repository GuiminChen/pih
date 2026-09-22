#include "pih/model/deepseek_routed_expert_oracle.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace pih {
namespace {

struct OwnedMatrix final {
  std::vector<std::byte> packed;
  std::vector<std::byte> scales;

  DeepSeekFp4MatrixView view() const { return {packed, scales}; }
};

OwnedMatrix first_column_matrix(std::uint32_t rows, std::uint32_t columns) {
  OwnedMatrix matrix;
  matrix.packed.resize(static_cast<std::size_t>(rows) * columns / 2U,
                       std::byte{0});
  matrix.scales.resize(static_cast<std::size_t>(rows) * columns / 32U,
                       std::byte{127});
  for (std::uint32_t row = 0; row < rows; ++row) {
    matrix.packed[static_cast<std::size_t>(row) * columns / 2U] =
        std::byte{0x02};  // w[row, 0] = +1
  }
  return matrix;
}

TEST(DeepSeekRoutedExpertOracleTest, ExecutesThreeProjectionExpert) {
  constexpr std::uint32_t kWidth = 128;
  std::vector<BFloat16> input(kWidth);
  input[0] = BFloat16::FromFloat(1.0F);
  const auto w1 = first_column_matrix(kWidth, kWidth);
  const auto w3 = first_column_matrix(kWidth, kWidth);
  const auto w2 = first_column_matrix(kWidth, kWidth);
  const std::vector<float> route_weight = {1.0F};
  auto result = DeepSeekRoutedExpertOracle::Forward(
      input, route_weight, 1, kWidth, kWidth, w1.view(), w2.view(), w3.view());
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->size(), kWidth);
  for (const auto value : *result) {
    // SwiGLU rounds to BF16, then W2 activation quantization maps it to the
    // nearest E4M3 value (0.75) before the identity-like projection.
    EXPECT_EQ(value, BFloat16::FromFloat(0.75F));
  }
}

TEST(DeepSeekRoutedExpertOracleTest, AppliesRouteWeightBeforeW2Quantization) {
  constexpr std::uint32_t kWidth = 128;
  std::vector<BFloat16> input(kWidth);
  input[0] = BFloat16::FromFloat(1.0F);
  const auto w1 = first_column_matrix(kWidth, kWidth);
  const auto w3 = first_column_matrix(kWidth, kWidth);
  const auto w2 = first_column_matrix(kWidth, kWidth);
  auto weighted = DeepSeekRoutedExpertOracle::Forward(
      input, std::vector<float>{0.3F}, 1, kWidth, kWidth, w1.view(), w2.view(),
      w3.view());
  auto unweighted = DeepSeekRoutedExpertOracle::Forward(
      input, std::vector<float>{1.0F}, 1, kWidth, kWidth, w1.view(), w2.view(),
      w3.view());
  ASSERT_TRUE(weighted.ok());
  ASSERT_TRUE(unweighted.ok());
  EXPECT_NE((*weighted)[0],
            BFloat16::FromFloat((*unweighted)[0].to_float() * 0.3F));
}

TEST(DeepSeekRoutedExpertOracleTest, RejectsMalformedRouteAndMatrix) {
  constexpr std::uint32_t kWidth = 128;
  const std::vector<BFloat16> input(kWidth);
  const auto matrix = first_column_matrix(kWidth, kWidth);
  EXPECT_FALSE(DeepSeekRoutedExpertOracle::Forward(
                   input, std::vector<float>{}, 1, kWidth, kWidth,
                   matrix.view(), matrix.view(), matrix.view()).ok());
  auto malformed = matrix;
  malformed.scales.pop_back();
  EXPECT_FALSE(DeepSeekRoutedExpertOracle::Forward(
                   input, std::vector<float>{1.0F}, 1, kWidth, kWidth,
                   malformed.view(), matrix.view(), matrix.view()).ok());
}

}  // namespace
}  // namespace pih
