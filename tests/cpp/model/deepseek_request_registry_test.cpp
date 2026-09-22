#include "pih/model/deepseek_request_registry.h"

#include <gtest/gtest.h>

namespace pih { namespace {

TEST(DeepSeekRequestRegistryTest, OwnsGenerationUntilTerminalDrainRetires) {
  DeepSeekRequestRegistry registry;
  EXPECT_TRUE(registry.submit(10, 3).ok());
  EXPECT_FALSE(registry.submit(10, 4).ok());
  EXPECT_EQ(registry.active_count(), 1U);
  EXPECT_FALSE(registry.cancel(10, 2).ok());
  EXPECT_TRUE(registry.cancel(10, 3).ok());
  EXPECT_EQ(registry.state(10, 3).value(), DeepSeekRequestState::kCancelled);
  EXPECT_TRUE(registry.validate_retire(10, 3).ok());
  EXPECT_EQ(registry.active_count(), 1U);
  EXPECT_TRUE(registry.retire(10, 3).ok());
  EXPECT_EQ(registry.active_count(), 0U);
  EXPECT_TRUE(registry.submit(10, 4).ok());
}

TEST(DeepSeekRequestRegistryTest, FailAllPreservesCommittedDrainOwnership) {
  DeepSeekRequestRegistry registry;
  ASSERT_TRUE(registry.submit(10, 3).ok());
  ASSERT_TRUE(registry.prepare(10, 3, 5).ok());
  ASSERT_TRUE(registry.commit(10, 3, 5).ok());
  ASSERT_TRUE(registry.submit(11, 1).ok());
  registry.fail_all();
  EXPECT_EQ(registry.state(10, 3).value(), DeepSeekRequestState::kDraining);
  EXPECT_EQ(registry.state(11, 1).value(), DeepSeekRequestState::kFailed);
  EXPECT_FALSE(registry.validate_retire(10, 3).ok());
  EXPECT_FALSE(registry.retire(10, 3).ok());
  EXPECT_TRUE(registry.backend_complete(10, 3, 5).ok());
  EXPECT_EQ(registry.state(10, 3).value(), DeepSeekRequestState::kFailed);
  EXPECT_TRUE(registry.retire(10, 3).ok());
}

TEST(DeepSeekRequestRegistryTest, CancelAllDrainsOnlyCommittedRequests) {
  DeepSeekRequestRegistry registry;
  ASSERT_TRUE(registry.submit(10, 3).ok());
  ASSERT_TRUE(registry.prepare(10, 3, 5).ok());
  ASSERT_TRUE(registry.commit(10, 3, 5).ok());
  ASSERT_TRUE(registry.submit(11, 1).ok());
  registry.cancel_all();
  EXPECT_EQ(registry.state(10, 3).value(), DeepSeekRequestState::kDraining);
  EXPECT_EQ(registry.state(11, 1).value(), DeepSeekRequestState::kCancelled);
  ASSERT_TRUE(registry.backend_complete(10, 3, 5).ok());
  EXPECT_EQ(registry.state(10, 3).value(), DeepSeekRequestState::kCancelled);
}

} }  // namespace pih
