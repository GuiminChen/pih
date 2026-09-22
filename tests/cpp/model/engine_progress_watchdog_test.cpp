#include "pih/model/engine_progress_watchdog.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

EngineProgressHeartbeat heartbeat(std::uint64_t controller,
                                  std::vector<std::uint64_t> ranks,
                                  std::uint64_t observed = 101) {
  return {7, controller, std::move(ranks), observed};
}

TEST(EngineProgressWatchdogTest, RenewsOnlyWhenControllerAndEveryRankAdvance) {
  auto watchdog = EngineProgressWatchdog::Create(7, 2, 10).value();
  ASSERT_TRUE(watchdog.arm(100).ok());
  EXPECT_EQ(watchdog.deadline_ns(), 110U);
  ASSERT_TRUE(watchdog.accept(heartbeat(1, {1, 1}), 101).ok());
  EXPECT_EQ(watchdog.deadline_ns(), 111U);
  ASSERT_TRUE(watchdog.accept(heartbeat(2, {2, 2}, 105), 105).ok());
  EXPECT_EQ(watchdog.deadline_ns(), 115U);
  EXPECT_TRUE(watchdog.poll(114).ok());
  EXPECT_EQ(watchdog.poll(115).code(), StatusCode::kDeadlineExceeded);
  EXPECT_TRUE(watchdog.expired());
}

TEST(EngineProgressWatchdogTest, RejectsReplayPartialAndWrongGeneration) {
  for (int mutation = 0; mutation < 4; ++mutation) {
    auto watchdog = EngineProgressWatchdog::Create(7, 2, 10).value();
    ASSERT_TRUE(watchdog.arm(100).ok());
    ASSERT_TRUE(watchdog.accept(heartbeat(1, {1, 1}), 101).ok());
    auto value = heartbeat(2, {2, 2}, 102);
    if (mutation == 0) value.controller_progress = 1;
    if (mutation == 1) value.rank_progress[1] = 1;
    if (mutation == 2) value.rank_progress.pop_back();
    if (mutation == 3) value.engine_generation = 8;
    EXPECT_FALSE(watchdog.accept(value, 102).ok()) << mutation;
    EXPECT_TRUE(watchdog.expired()) << mutation;
  }
}

TEST(EngineProgressWatchdogTest, RejectsClockAndReceiptTimeDrift) {
  for (int mutation = 0; mutation < 3; ++mutation) {
    auto watchdog = EngineProgressWatchdog::Create(7, 1, 10).value();
    ASSERT_TRUE(watchdog.arm(100).ok());
    auto value = heartbeat(1, {1}, 101);
    std::uint64_t now = 101;
    if (mutation == 0) value.observed_at_ns = 99;
    if (mutation == 1) value.observed_at_ns = 102;
    if (mutation == 2) now = 99;
    EXPECT_FALSE(watchdog.accept(value, now).ok()) << mutation;
    EXPECT_TRUE(watchdog.expired()) << mutation;
  }
}

TEST(EngineProgressWatchdogTest, RejectsInvalidConfigurationAndOverflow) {
  EXPECT_FALSE(EngineProgressWatchdog::Create(0, 1, 10).ok());
  EXPECT_FALSE(EngineProgressWatchdog::Create(7, 0, 10).ok());
  EXPECT_FALSE(EngineProgressWatchdog::Create(7, 5, 10).ok());
  EXPECT_FALSE(EngineProgressWatchdog::Create(7, 1, 0).ok());
  auto watchdog = EngineProgressWatchdog::Create(7, 1, 10).value();
  EXPECT_FALSE(watchdog.poll(1).ok());
  EXPECT_FALSE(watchdog.arm(UINT64_MAX - 9).ok());
  EXPECT_TRUE(watchdog.expired());
}

}  // namespace
}  // namespace pih
