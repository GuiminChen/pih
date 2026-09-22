#include "pih/model/deepseek_expert_transfer_lane.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class ResourceDriver final : public CudaRuntimeResourceDriver {
 public:
  Result<std::uintptr_t> retain_primary_context(std::int32_t,std::uint32_t) override { return std::uintptr_t{1}; }
  Status bind_runtime(std::int32_t,std::uintptr_t) override { return Status::Ok(); }
  Result<DriverStreamHandle> create_nonblocking_stream(std::uintptr_t) override { return DriverStreamHandle{1}; }
  Result<DriverEventHandle> create_disable_timing_event(std::uintptr_t context) override {
    EXPECT_EQ(context,77U); ++create_calls;
    if (fail_at==create_calls) return Status::Unavailable("event failure");
    return static_cast<DriverEventHandle>(100+create_calls);
  }
  void destroy_event(DriverEventHandle event) noexcept override { destroyed.push_back(event); }
  void destroy_stream(DriverStreamHandle) noexcept override {}
  void release_primary_context(std::int32_t,std::uintptr_t) noexcept override {}
  int create_calls=0; int fail_at=0; std::vector<DriverEventHandle> destroyed;
};

class Source final : public DeepSeekExpertHostSource {
 public:
  Result<DeepSeekPinnedExpertExtent> resolve(DeepSeekExpertIdentity,
                                              std::uint64_t bytes) override {
    return DeepSeekPinnedExpertExtent{0x900000,bytes,5};
  }
};

class Runtime final : public DeepSeekH2dRuntime {
 public:
  Status copy_async(std::uintptr_t destination,std::uintptr_t,
                    std::uint64_t,std::uintptr_t stream) override {
    copied_destination=destination; copied_stream=stream; return Status::Ok();
  }
  Status record_event(std::uintptr_t event,std::uintptr_t stream) override {
    recorded_event=event; recorded_stream=stream; return Status::Ok();
  }
  Result<DeepSeekTransferEventStatus> query_event(std::uintptr_t) override {
    return DeepSeekTransferEventStatus::kSuccess;
  }
  std::uintptr_t copied_destination=0,copied_stream=0,recorded_event=0,recorded_stream=0;
};

CudaRuntimeResourceIdentity identity() {
  CudaRuntimeResourceIdentity value{};
  value.context=77; value.deepseek_paging_stream=88;
  return value;
}

TEST(DeepSeekExpertTransferLaneTest, OwnsPerSlotEventsAcrossMove) {
  ResourceDriver resources; Source source; Runtime runtime;
  {
    auto lane=DeepSeekExpertTransferLane::Create(
        source,runtime,{0x10000000,0x11000000},identity(),resources);
    ASSERT_TRUE(lane.ok()); EXPECT_EQ(lane->slot_count(),2U);
    DeepSeekExpertTransferLane moved=std::move(*lane);
    ASSERT_TRUE(moved.transfer().start({4,3},1,1,
        DeepSeekExpertPager::kBundleBytes).ok());
    EXPECT_EQ(runtime.copied_destination,0x11000000U);
    EXPECT_EQ(runtime.copied_stream,88U);
    EXPECT_EQ(runtime.recorded_event,102U);
    EXPECT_EQ(runtime.recorded_stream,88U);
    auto status=moved.transfer().poll({4,3},1); ASSERT_TRUE(status.ok());
    EXPECT_EQ(*status,DeepSeekExpertAsyncStatus::kSuccess);
  }
  EXPECT_EQ(resources.destroyed,
            (std::vector<DriverEventHandle>{102,101}));
}

TEST(DeepSeekExpertTransferLaneTest, RollsBackEventsWhenCreationFails) {
  ResourceDriver resources; resources.fail_at=2; Source source; Runtime runtime;
  auto lane=DeepSeekExpertTransferLane::Create(
      source,runtime,{0x10000000,0x11000000},identity(),resources);
  EXPECT_FALSE(lane.ok());
  EXPECT_EQ(resources.destroyed,(std::vector<DriverEventHandle>{101}));
}

} }  // namespace pih
