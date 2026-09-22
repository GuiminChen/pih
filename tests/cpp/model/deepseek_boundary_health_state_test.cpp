#include "pih/model/deepseek_boundary_health_state.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

TEST(DeepSeekBoundaryHealthStateTest, PublishesFirstDeviceErrorAndPoisons) {
  DeepSeekBoundaryHealthState health;
  EXPECT_EQ(health.device_error_code(), 0U);
  EXPECT_FALSE(health.engine_poisoned());
  ASSERT_TRUE(health.report_device_error(17).ok());
  EXPECT_EQ(health.device_error_code(), 17U);
  EXPECT_TRUE(health.engine_poisoned());
  EXPECT_FALSE(health.report_device_error(19).ok());
  EXPECT_EQ(health.device_error_code(), 17U);
}

TEST(DeepSeekBoundaryHealthStateTest, RejectsZeroAndAllowsIdempotentReport) {
  DeepSeekBoundaryHealthState health;
  EXPECT_FALSE(health.report_device_error(0).ok());
  ASSERT_TRUE(health.report_device_error(23).ok());
  EXPECT_TRUE(health.report_device_error(23).ok());
}

}  // namespace
}  // namespace pih
