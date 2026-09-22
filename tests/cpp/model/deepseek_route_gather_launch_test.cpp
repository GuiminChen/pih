#include "pih/backend/cuda/deepseek_route_gather.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekRouteGatherLaunch valid_gather() {
  return {.source_hidden_bf16 = 1,
          .token_indices_u32 = 2,
          .route_hidden_bf16 = 3,
          .error_flag = 4,
          .stream = 5,
          .route_count = 3,
          .packed_token_count = 8};
}

TEST(DeepSeekRouteGatherLaunchTest, AcceptsBoundedRouteSlice) {
  EXPECT_TRUE(validate_deepseek_route_gather_launch(valid_gather()).ok());
  EXPECT_EQ(DeepSeekRouteGatherLaunch::kHiddenSize, 4096U);
}

TEST(DeepSeekRouteGatherLaunchTest, RejectsMissingOrOversizedSlice) {
  auto launch = valid_gather();
  launch.source_hidden_bf16 = 0;
  EXPECT_FALSE(validate_deepseek_route_gather_launch(launch).ok());
  launch = valid_gather();
  launch.token_indices_u32 = 0;
  EXPECT_FALSE(validate_deepseek_route_gather_launch(launch).ok());
  launch = valid_gather();
  launch.stream = 0;
  EXPECT_FALSE(validate_deepseek_route_gather_launch(launch).ok());
  launch = valid_gather();
  launch.route_count = 0;
  EXPECT_FALSE(validate_deepseek_route_gather_launch(launch).ok());
  launch = valid_gather();
  launch.route_count = 9;
  EXPECT_FALSE(validate_deepseek_route_gather_launch(launch).ok());
}

}  // namespace
}  // namespace pih
