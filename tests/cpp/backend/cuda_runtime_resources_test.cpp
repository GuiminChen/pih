#include "pih/backend/cuda/cuda_runtime_resources.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

class Driver final : public CudaRuntimeResourceDriver {
 public:
  Result<std::uintptr_t> retain_primary_context(std::int32_t,
                                                std::uint32_t) override {
    calls.push_back("retain");
    return context_result;
  }
  Status bind_runtime(std::int32_t, std::uintptr_t) override {
    calls.push_back("bind");
    return bind_result;
  }
  Result<DriverStreamHandle> create_nonblocking_stream(
      std::uintptr_t) override {
    calls.push_back("stream");
    if (!stream_result.ok()) return stream_result.status();
    return stream_result.value() + stream_calls++;
  }
  Result<DriverEventHandle> create_disable_timing_event(
      std::uintptr_t) override {
    calls.push_back("event");
    if (!event_result.ok()) return event_result.status();
    return event_result.value() + event_calls++;
  }
  void destroy_event(DriverEventHandle) noexcept override {
    calls.push_back("destroy_event");
  }
  void destroy_stream(DriverStreamHandle) noexcept override {
    calls.push_back("destroy_stream");
  }
  void release_primary_context(std::int32_t,
                               std::uintptr_t) noexcept override {
    calls.push_back("release");
  }

  Result<std::uintptr_t> context_result{std::uintptr_t{11}};
  Status bind_result = Status::Ok();
  Result<DriverStreamHandle> stream_result{DriverStreamHandle{13}};
  Result<DriverEventHandle> event_result{DriverEventHandle{17}};
  std::vector<std::string> calls;
  DriverStreamHandle stream_calls = 0;
  DriverEventHandle event_calls = 0;
};

TEST(CudaRuntimeResourcesTest, OwnsIdentityAndDestroysInReverseOrder) {
  Driver driver;
  {
    auto resources = CudaRuntimeResources::Create(2, 3, 5, 7, driver);
    ASSERT_TRUE(resources.ok());
    EXPECT_EQ(resources->identity().device_ordinal, 2);
    EXPECT_EQ(resources->identity().rank, 3U);
    EXPECT_EQ(resources->identity().worker_generation, 5U);
    EXPECT_EQ(resources->identity().context_flags, 7U);
    EXPECT_EQ(resources->identity().context, 11U);
    EXPECT_EQ(resources->identity().stream, 13U);
    EXPECT_EQ(resources->identity().event, 17U);
    EXPECT_EQ(resources->identity().scrub_stream, 14U);
    EXPECT_EQ(resources->identity().scrub_event, 18U);
    EXPECT_EQ(resources->identity().diagnostic_stream, 15U);
    EXPECT_EQ(resources->identity().deepseek_paging_stream, 16U);
    EXPECT_EQ(resources->identity().diagnostic_event, 19U);
    EXPECT_EQ(resources->identity().deepseek_expert_event, 20U);
  }
  EXPECT_EQ(driver.calls,
            (std::vector<std::string>{
                "retain", "bind", "stream", "stream", "stream", "stream", "event",
                "event", "event", "event", "destroy_event", "destroy_event",
                "destroy_event", "destroy_event", "destroy_stream", "destroy_stream",
                "destroy_stream", "destroy_stream", "release"}));
}

TEST(CudaRuntimeResourcesTest, EventFailureRollsBackStreamAndContext) {
  Driver driver;
  driver.event_result = Status::Unavailable("event unavailable");
  auto resources = CudaRuntimeResources::Create(0, 0, 1, 0, driver);
  ASSERT_FALSE(resources.ok());
  EXPECT_EQ(driver.calls,
            (std::vector<std::string>{
                "retain", "bind", "stream", "stream", "stream", "stream", "event",
                "destroy_stream", "destroy_stream", "destroy_stream", "destroy_stream",
                "release"}));
}

TEST(CudaRuntimeResourcesTest, BindFailureReleasesOnlyContext) {
  Driver driver;
  driver.bind_result = Status::FailedPrecondition("runtime mismatch");
  auto resources = CudaRuntimeResources::Create(0, 0, 1, 0, driver);
  ASSERT_FALSE(resources.ok());
  EXPECT_EQ(driver.calls,
            (std::vector<std::string>{"retain", "bind", "release"}));
}

TEST(CudaRuntimeResourcesTest, RejectsInvalidIdentityWithoutDriverCalls) {
  Driver driver;
  EXPECT_FALSE(CudaRuntimeResources::Create(-1, 0, 1, 0, driver).ok());
  EXPECT_FALSE(CudaRuntimeResources::Create(0, UINT32_MAX, 1, 0, driver).ok());
  EXPECT_FALSE(CudaRuntimeResources::Create(0, 0, 0, 0, driver).ok());
  EXPECT_TRUE(driver.calls.empty());
}

}  // namespace
}  // namespace pih
