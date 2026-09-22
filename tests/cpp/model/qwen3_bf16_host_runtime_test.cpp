#include "pih/model/qwen3_bf16_host_runtime.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

static_assert(std::is_base_of_v<QwenInt4StartupClock,QwenBf16SteadyClock>);
static_assert(std::is_base_of_v<QwenInt4StartupWaiter,QwenBf16YieldWaiter>);

TEST(QwenBf16HostRuntimeTest, SteadyClockNeverRegressesAcrossYield) {
  QwenBf16SteadyClock clock;
  QwenBf16YieldWaiter waiter;
  auto before = clock.now_ns();
  ASSERT_TRUE(before.ok());
  ASSERT_TRUE(waiter.wait().ok());
  auto after = clock.now_ns();
  ASSERT_TRUE(after.ok());
  EXPECT_GE(*after, *before);
}

}  // namespace
}  // namespace pih
