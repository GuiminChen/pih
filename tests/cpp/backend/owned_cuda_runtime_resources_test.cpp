#include "pih/backend/cuda/owned_cuda_runtime_resources.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace pih {
namespace {

struct State final { std::vector<std::string> calls; };

class Driver final : public CudaRuntimeResourceDriver {
 public:
  explicit Driver(std::shared_ptr<State> state) : state_(std::move(state)) {}
  ~Driver() override { state_->calls.push_back("driver_destroy"); }
  Result<std::uintptr_t> retain_primary_context(
      std::int32_t, std::uint32_t) override {
    state_->calls.push_back("retain");
    return std::uintptr_t{11};
  }
  Status bind_runtime(std::int32_t, std::uintptr_t) override {
    state_->calls.push_back("bind");
    return Status::Ok();
  }
  Result<DriverStreamHandle> create_nonblocking_stream(
      std::uintptr_t) override {
    return DriverStreamHandle{20 + next_stream_++};
  }
  Result<DriverEventHandle> create_disable_timing_event(
      std::uintptr_t) override {
    return DriverEventHandle{40 + next_event_++};
  }
  void destroy_event(DriverEventHandle) noexcept override {
    state_->calls.push_back("destroy_event");
  }
  void destroy_stream(DriverStreamHandle) noexcept override {
    state_->calls.push_back("destroy_stream");
  }
  void release_primary_context(std::int32_t, std::uintptr_t) noexcept override {
    state_->calls.push_back("release");
  }
 private:
  std::shared_ptr<State> state_;
  DriverStreamHandle next_stream_ = 0;
  DriverEventHandle next_event_ = 0;
};

TEST(OwnedCudaRuntimeResourcesTest,
     MoveKeepsDriverAliveUntilEveryRuntimeResourceIsReleased) {
  auto state = std::make_shared<State>();
  {
    auto owner = OwnedCudaRuntimeResources::Create(
        2, 3, 5, 7, std::make_unique<Driver>(state));
    ASSERT_TRUE(owner.ok());
    OwnedCudaRuntimeResources moved(std::move(*owner));
    EXPECT_EQ(moved.identity().context, 11U);
    EXPECT_EQ(moved.identity().deepseek_paging_stream, 23U);
    EXPECT_EQ(moved.identity().deepseek_expert_event, 43U);
  }
  ASSERT_GE(state->calls.size(), 3U);
  EXPECT_EQ(state->calls[state->calls.size() - 2], "release");
  EXPECT_EQ(state->calls.back(), "driver_destroy");
  EXPECT_EQ(std::count(state->calls.begin(), state->calls.end(),
                       "destroy_event"), 4);
  EXPECT_EQ(std::count(state->calls.begin(), state->calls.end(),
                       "destroy_stream"), 4);
}

TEST(OwnedCudaRuntimeResourcesTest, RejectsMissingDriver) {
  EXPECT_EQ(OwnedCudaRuntimeResources::Create(0, 0, 1, 0, nullptr)
                .status().code(),
            StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pih
