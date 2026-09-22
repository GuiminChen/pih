#include "pih/model/deepseek_index_selection_driver.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace pih { namespace {

class FakeIndexSelectionOperations final
    : public DeepSeekIndexSelectionOperations {
 public:
  Status validate_host_staging(
      const DeepSeekIndexSelectionHostStaging&) override {
    return Status::Ok();
  }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status score(DeepSeekIndexScoreLaunch launch) override {
    last_launch = launch;
    tile_base = score_calls * DeepSeekIndexScoreLaunch::kMaximumSlotTile;
    ++score_calls;
    return fail_score ? Status::Internal("injected score failure") : Status::Ok();
  }
  Status copy_h2d_async(std::uintptr_t, const void* host, std::size_t bytes,
                        std::uintptr_t) override {
    const auto* slots = static_cast<const std::uint32_t*>(host);
    copied_page_slots.assign(slots, slots + bytes / sizeof(std::uint32_t));
    return Status::Ok();
  }
  Status copy_d2h_async(void* host, std::uintptr_t, std::size_t bytes,
                        std::uintptr_t) override {
    if (bytes == sizeof(std::uint32_t)) {
      *static_cast<std::uint32_t*>(host) = device_error;
      return Status::Ok();
    }
    auto* scores = static_cast<float*>(host);
    for (std::uint32_t query = 0; query < last_launch.query_count; ++query) {
      for (std::uint32_t slot = 0; slot < last_launch.slot_count; ++slot) {
        scores[static_cast<std::size_t>(query) * last_launch.slot_count + slot] =
            static_cast<float>(tile_base + slot) + query * 0.25F;
      }
    }
    return Status::Ok();
  }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return event_status;
  }

  DeepSeekIndexScoreLaunch last_launch;
  DeepSeekExpertAsyncStatus event_status =
      DeepSeekExpertAsyncStatus::kSuccess;
  std::uint32_t score_calls = 0;
  std::uint32_t tile_base = 0;
  std::uint32_t device_error = 0;
  bool fail_score = false;
  std::vector<std::uint32_t> copied_page_slots;
};

struct DriverFixture final {
  std::vector<float> scores = std::vector<float>(2U * 4096U);
  std::uint32_t error = 0;
  std::vector<std::uint32_t> page_slots = std::vector<std::uint32_t>(128);
  FakeIndexSelectionOperations operations;

  DeepSeekIndexSelectionDriver create() {
    return DeepSeekIndexSelectionDriver::Create(
               operations, {scores.data(), &error,
                            static_cast<std::uint32_t>(scores.size()),
                            page_slots.data(),
                            static_cast<std::uint32_t>(page_slots.size())}, 99)
        .value();
  }
};

TEST(DeepSeekIndexSelectionDriverTest,
     StreamsCausalQueriesAcrossTilesWithoutCollectives) {
  DriverFixture fixture;
  auto driver = fixture.create();
  const std::vector<std::uint32_t> visible{5000, 3};
  ASSERT_TRUE(driver.begin({1000, 2000, 3000, {4000, 5000}, visible,
                            5000, 2, 64, 6000})
                  .ok());
  while (!driver.complete()) {
    ASSERT_TRUE(driver.launch_next_tile().ok());
    EXPECT_EQ(*driver.poll_tile(), DeepSeekExpertAsyncStatus::kSuccess);
  }
  auto result = driver.finish();
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->size(), 2U);
  ASSERT_EQ((*result)[0].size(), 512U);
  EXPECT_EQ((*result)[0].front(), 4999U);
  EXPECT_EQ((*result)[0].back(), 4488U);
  EXPECT_EQ((*result)[1], std::vector<std::uint32_t>({2, 1, 0}));
  EXPECT_EQ(fixture.operations.score_calls, 2U);
}

TEST(DeepSeekIndexSelectionDriverTest,
     StagesPagedTableAndCarriesGlobalTileBase) {
  DriverFixture fixture;
  auto driver = fixture.create();
  const std::vector<std::uint32_t> visible{5000};
  std::vector<std::uint32_t> pages(79);
  for (std::uint32_t page = 0; page < pages.size(); ++page) {
    pages[page] = (page * 3U) % 97U;
  }
  DeepSeekIndexSelectionSubmission submission{
      1000, 2000, 3000, {4000, 5000, 6000, 128}, visible,
      5000, 1, 64, 7000};
  ASSERT_TRUE(driver.begin_paged(submission, pages, 97).ok());
  EXPECT_EQ(fixture.operations.copied_page_slots, pages);
  ASSERT_TRUE(driver.launch_next_tile().ok());
  EXPECT_EQ(fixture.operations.last_launch.page_slots_u32, 6000U);
  EXPECT_EQ(fixture.operations.last_launch.slot_base, 0U);
  EXPECT_EQ(fixture.operations.last_launch.logical_page_count, 79U);
  EXPECT_EQ(fixture.operations.last_launch.physical_page_count, 97U);
  ASSERT_EQ(driver.poll_tile().value(), DeepSeekExpertAsyncStatus::kSuccess);
  ASSERT_TRUE(driver.launch_next_tile().ok());
  EXPECT_EQ(fixture.operations.last_launch.slot_base, 4096U);
  EXPECT_EQ(fixture.operations.last_launch.index_kv_bf16, 2000U);
}

TEST(DeepSeekIndexSelectionDriverTest, SupportsZeroVisiblePrefillQuery) {
  DriverFixture fixture;
  auto driver = fixture.create();
  const std::vector<std::uint32_t> visible{0, 1};
  ASSERT_TRUE(driver.begin({1, 2, 3, {4, 5}, visible, 1, 2, 64, 6}).ok());
  ASSERT_TRUE(driver.launch_next_tile().ok());
  ASSERT_EQ(*driver.poll_tile(), DeepSeekExpertAsyncStatus::kSuccess);
  auto result = driver.finish();
  ASSERT_TRUE(result.ok());
  EXPECT_TRUE((*result)[0].empty());
  EXPECT_EQ((*result)[1], std::vector<std::uint32_t>({0}));
}

TEST(DeepSeekIndexSelectionDriverTest, ValidationDoesNotBeginOrLaunchWork) {
  DriverFixture fixture;
  auto driver = fixture.create();
  const std::vector<std::uint32_t> visible{1};
  EXPECT_TRUE(driver.validate(
      {1, 2, 3, {4, 5}, visible, 1, 1, 64, 6}).ok());
  EXPECT_EQ(fixture.operations.score_calls, 0U);
  EXPECT_FALSE(driver.launch_next_tile().ok());
}

TEST(DeepSeekIndexSelectionDriverTest, PreservesInflightAndPoisonsErrors) {
  DriverFixture fixture;
  auto driver = fixture.create();
  const std::vector<std::uint32_t> visible{1};
  ASSERT_TRUE(driver.begin({1, 2, 3, {4, 5}, visible, 1, 1, 64, 6}).ok());
  ASSERT_TRUE(driver.launch_next_tile().ok());
  fixture.operations.event_status = DeepSeekExpertAsyncStatus::kInProgress;
  EXPECT_EQ(*driver.poll_tile(), DeepSeekExpertAsyncStatus::kInProgress);
  EXPECT_FALSE(driver.launch_next_tile().ok());
  fixture.operations.event_status = DeepSeekExpertAsyncStatus::kSuccess;
  fixture.error = 1;
  EXPECT_EQ(*driver.poll_tile(), DeepSeekExpertAsyncStatus::kError);
  EXPECT_FALSE(driver.begin({1, 2, 3, {4, 5}, visible, 1, 1, 64, 6}).ok());
}

TEST(DeepSeekIndexSelectionDriverTest, RequiresCompletePpLocalHeadSet) {
  DriverFixture fixture;
  auto driver = fixture.create();
  const std::vector<std::uint32_t> visible{1};
  EXPECT_FALSE(
      driver.begin({1, 2, 3, {4, 5}, visible, 1, 1, 63, 6}).ok());
  EXPECT_TRUE(
      driver.begin({1, 2, 3, {4, 5}, visible, 1, 1, 64, 6}).ok());
}

} }  // namespace pih
