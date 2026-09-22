#include "pih/model/deepseek_expert_transfer_driver.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih {
namespace {

class FixedExpertSource final : public DeepSeekExpertHostSource {
 public:
  Result<DeepSeekPinnedExpertExtent> resolve(
      DeepSeekExpertIdentity identity, std::uint64_t bytes) override {
    if (fail) return Status::Internal("source mapping failed");
    last_identity = identity;
    last_bytes = bytes;
    return DeepSeekPinnedExpertExtent{
        base + static_cast<std::uintptr_t>(identity.expert) * bytes,
        malformed_extent ? bytes - 1 : bytes, 17};
  }
  Status release(DeepSeekExpertIdentity identity,
                 DeepSeekPinnedExpertExtent extent) override {
    ++release_count;
    released_identity = identity;
    released_extent = extent;
    return fail_release ? Status::Internal("source release failed")
                        : Status::Ok();
  }
  bool fail = false;
  bool fail_release = false;
  bool malformed_extent = false;
  std::uintptr_t base = 0x100000;
  DeepSeekExpertIdentity last_identity{};
  std::uint64_t last_bytes = 0;
  std::uint64_t release_count = 0;
  DeepSeekExpertIdentity released_identity{};
  DeepSeekPinnedExpertExtent released_extent{};
};

class ScriptedH2dRuntime final : public DeepSeekH2dRuntime {
 public:
  Status copy_async(std::uintptr_t destination, std::uintptr_t source,
                    std::uint64_t bytes, std::uintptr_t stream) override {
    if (fail_copy) return Status::Internal("copy failed");
    copied_destination = destination;
    copied_source = source;
    copied_bytes = bytes;
    copied_stream = stream;
    return Status::Ok();
  }
  Status record_event(std::uintptr_t event, std::uintptr_t stream) override {
    if (fail_record) return Status::Internal("record failed");
    recorded_event = event;
    recorded_stream = stream;
    return Status::Ok();
  }
  Result<DeepSeekTransferEventStatus> query_event(
      std::uintptr_t event) override {
    if (fail_query) return Status::Internal("query failed");
    queried_event = event;
    return query_status;
  }
  bool fail_copy = false;
  bool fail_record = false;
  bool fail_query = false;
  DeepSeekTransferEventStatus query_status =
      DeepSeekTransferEventStatus::kNotReady;
  std::uintptr_t copied_destination = 0;
  std::uintptr_t copied_source = 0;
  std::uint64_t copied_bytes = 0;
  std::uintptr_t copied_stream = 0;
  std::uintptr_t recorded_event = 0;
  std::uintptr_t recorded_stream = 0;
  std::uintptr_t queried_event = 0;
};

std::vector<DeepSeekExpertDeviceSlot> transfer_slots() {
  return {{0x200000, 0x300000}, {0x400000, 0x500000}};
}

TEST(DeepSeekExpertTransferDriverTest, CopiesCanonicalBundleAndRecordsEvent) {
  FixedExpertSource source;
  ScriptedH2dRuntime runtime;
  auto driver = DeepSeekExpertTransferStateDriver::Create(
      source, runtime, transfer_slots(), 0x600000);
  ASSERT_TRUE(driver.ok());
  ASSERT_TRUE(driver->start({4, 7}, 1, 9,
                            DeepSeekExpertPager::kBundleBytes).ok());
  EXPECT_EQ(source.last_identity, (DeepSeekExpertIdentity{4, 7}));
  EXPECT_EQ(runtime.copied_destination, 0x400000U);
  EXPECT_EQ(runtime.copied_source,
            source.base + 7U * DeepSeekExpertPager::kBundleBytes);
  EXPECT_EQ(runtime.copied_bytes, DeepSeekExpertPager::kBundleBytes);
  EXPECT_EQ(runtime.copied_stream, 0x600000U);
  EXPECT_EQ(runtime.recorded_event, 0x500000U);
}

TEST(DeepSeekExpertTransferDriverTest, PollsGenerationToCompletionAndReusesSlot) {
  FixedExpertSource source;
  ScriptedH2dRuntime runtime;
  auto driver = DeepSeekExpertTransferStateDriver::Create(
      source, runtime, transfer_slots(), 0x600000);
  ASSERT_TRUE(driver.ok());
  ASSERT_TRUE(driver->start({4, 7}, 0, 9,
                            DeepSeekExpertPager::kBundleBytes).ok());
  auto pending = driver->poll({4, 7}, 9);
  ASSERT_TRUE(pending.ok());
  EXPECT_EQ(*pending, DeepSeekExpertAsyncStatus::kInProgress);
  EXPECT_EQ(source.release_count, 0U);
  runtime.query_status = DeepSeekTransferEventStatus::kSuccess;
  auto complete = driver->poll({4, 7}, 9);
  ASSERT_TRUE(complete.ok());
  EXPECT_EQ(*complete, DeepSeekExpertAsyncStatus::kSuccess);
  EXPECT_EQ(source.release_count, 1U);
  EXPECT_EQ(source.released_identity, (DeepSeekExpertIdentity{4, 7}));
  EXPECT_EQ(source.released_extent.address, source.base + 7U * DeepSeekExpertPager::kBundleBytes);
  EXPECT_FALSE(driver->poll({4, 7}, 9).ok());
  EXPECT_TRUE(driver->start({4, 8}, 0, 10,
                            DeepSeekExpertPager::kBundleBytes).ok());
}

TEST(DeepSeekExpertTransferDriverTest, ReleasesExtentOnSynchronousCopyFailure) {
  FixedExpertSource source;
  ScriptedH2dRuntime runtime;
  runtime.fail_copy = true;
  auto driver = DeepSeekExpertTransferStateDriver::Create(
      source, runtime, transfer_slots(), 0x600000);
  ASSERT_TRUE(driver.ok());
  EXPECT_FALSE(driver->start({4, 7}, 0, 9,
                             DeepSeekExpertPager::kBundleBytes).ok());
  EXPECT_EQ(source.release_count, 1U);
}

TEST(DeepSeekExpertTransferDriverTest, ReleaseFailurePoisonsTransferEpoch) {
  FixedExpertSource source;
  source.fail_release = true;
  ScriptedH2dRuntime runtime;
  auto driver = DeepSeekExpertTransferStateDriver::Create(
      source, runtime, transfer_slots(), 0x600000);
  ASSERT_TRUE(driver.ok());
  ASSERT_TRUE(driver->start({4, 7}, 0, 9,
                            DeepSeekExpertPager::kBundleBytes).ok());
  runtime.query_status = DeepSeekTransferEventStatus::kSuccess;
  EXPECT_FALSE(driver->poll({4, 7}, 9).ok());
  EXPECT_TRUE(driver->poisoned());
}

TEST(DeepSeekExpertTransferDriverTest, RejectsSlotRebindBeforeEventCompletion) {
  FixedExpertSource source;
  ScriptedH2dRuntime runtime;
  auto driver = DeepSeekExpertTransferStateDriver::Create(
      source, runtime, transfer_slots(), 0x600000);
  ASSERT_TRUE(driver.ok());
  ASSERT_TRUE(driver->start({4, 7}, 0, 9,
                            DeepSeekExpertPager::kBundleBytes).ok());
  EXPECT_FALSE(driver->start({4, 8}, 0, 10,
                             DeepSeekExpertPager::kBundleBytes).ok());
  EXPECT_FALSE(driver->poll({4, 7}, 10).ok());
  EXPECT_FALSE(driver->poisoned());
}

TEST(DeepSeekExpertTransferDriverTest, RuntimeFailurePoisonsTransferEpoch) {
  FixedExpertSource source;
  ScriptedH2dRuntime runtime;
  runtime.fail_copy = true;
  auto driver = DeepSeekExpertTransferStateDriver::Create(
      source, runtime, transfer_slots(), 0x600000);
  ASSERT_TRUE(driver.ok());
  EXPECT_FALSE(driver->start({4, 7}, 0, 9,
                             DeepSeekExpertPager::kBundleBytes).ok());
  EXPECT_TRUE(driver->poisoned());
  EXPECT_EQ(driver->poll({4, 7}, 9).status().code(), StatusCode::kUnavailable);
}

TEST(DeepSeekExpertTransferDriverTest, UnregisteredExtentPoisonsTransferEpoch) {
  FixedExpertSource source;
  source.malformed_extent = true;
  ScriptedH2dRuntime runtime;
  auto driver = DeepSeekExpertTransferStateDriver::Create(
      source, runtime, transfer_slots(), 0x600000);
  ASSERT_TRUE(driver.ok());
  EXPECT_FALSE(driver->start({4, 7}, 0, 9,
                             DeepSeekExpertPager::kBundleBytes).ok());
  EXPECT_TRUE(driver->poisoned());
  EXPECT_EQ(runtime.copied_bytes, 0U);
}

TEST(DeepSeekExpertTransferDriverTest, RequiresFixedTwoSlotTopology) {
  FixedExpertSource source;
  ScriptedH2dRuntime runtime;
  EXPECT_FALSE(DeepSeekExpertTransferStateDriver::Create(
      source, runtime, {{0x200000, 0x300000}}, 0x600000).ok());
  EXPECT_FALSE(DeepSeekExpertTransferStateDriver::Create(
      source, runtime, transfer_slots(), 0).ok());
}

}  // namespace
}  // namespace pih
