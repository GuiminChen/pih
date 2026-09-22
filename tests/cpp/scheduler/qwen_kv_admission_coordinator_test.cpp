#include "pih/scheduler/qwen_kv_admission_coordinator.h"

#include <array>

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest coordinator_resource_root() {
  constexpr std::string_view value = "qwen-coordinator-resources";
  return sha256(std::as_bytes(std::span(value))).value();
}

ControllerRuntime coordinator_runtime() {
  return ControllerRuntime::Create(
      {7, 2, {{2, 4}, {2, 4, 8}}, {2, 2, 4, 2, 2, 100},
       {2, 8}, 2, false, "profile-r1", coordinator_resource_root()}).value();
}

QwenKvSlotPool coordinator_pool(std::uint32_t slots) {
  auto pool = QwenKvSlotPool::Create(
      slots, static_cast<std::uint64_t>(slots) *
                 QwenKvSlotPool::kSlotPayloadBytes,
      static_cast<std::uint64_t>(slots) * sizeof(QwenKvSlotState)).value();
  EXPECT_TRUE(pool.complete_startup_sanitize(
      static_cast<std::uint64_t>(slots) * QwenKvSlotPool::kSlotPayloadBytes,
      static_cast<std::uint64_t>(slots) * sizeof(QwenKvSlotState), true).ok());
  return pool;
}

class CoordinatorBackend final : public QwenBf16PackedBatchBackend {
 public:
  Result<QwenBf16PackedBatchExecutionView> execute_packed(
      const PackedTokenPlan&, PackedTokenMetadataView,
      QwenBf16PackedKvMetadataView,
      std::span<const QwenBf16PackedKvBinding>,
      std::span<const QwenKvAppendPlan>,
      std::span<const Qwen3SamplingDescriptor>) override {
    return Status::Internal("coordinator test must not execute GPU work");
  }
};

class CoordinatorRecycler final : public QwenBf16KvRecycler {
 public:
  Status recycle(QwenKvSlotPool&,
                 std::span<const QwenKvBlockHandle> handles,
                 QwenKvCompletionEvent event) override {
    observed.assign(handles.begin(), handles.end());
    last_event = event;
    return Status::Ok();
  }
  std::vector<QwenKvBlockHandle> observed;
  QwenKvCompletionEvent last_event{};
};

TEST(QwenKvAdmissionCoordinatorTest,
     BindsPublishedLifetimeStateBeforeAdmittedEvent) {
  auto runtime = coordinator_runtime();
  auto pool = coordinator_pool(4);
  CoordinatorBackend backend;
  auto driver = ControllerPackedDriver::Create(runtime, backend, 2, {2, 8, 4});
  ASSERT_TRUE(driver.ok());
  auto coordinator = QwenKvAdmissionCoordinator::Create(
      pool, *driver, 2, {30, 1});
  ASSERT_TRUE(coordinator.ok());
  const std::array<std::uint32_t, 3> prompt{1, 2, 3};
  ASSERT_TRUE(runtime.submit_admit(1, prompt, 2).ok());
  ASSERT_EQ(runtime.step(10, *coordinator).value(),
            ControllerRuntimeStep::kAdmitted);
  EXPECT_EQ(coordinator->active_admission_count(), 1U);
  EXPECT_EQ(pool.lifecycle_count(QwenKvSlotLifecycle::kOwned), 1U);
  auto event = runtime.try_take_event().value();
  ASSERT_TRUE(event.has_value());
  EXPECT_EQ(event->kind, ControllerOutputEventKind::kAdmitted);

  ASSERT_TRUE(runtime.submit_cancel(1).ok());
  ASSERT_EQ(runtime.step(11, *coordinator).value(),
            ControllerRuntimeStep::kCancelRequested);
  ASSERT_TRUE(runtime.try_take_event().value().has_value());
  CoordinatorRecycler recycler;
  ASSERT_TRUE(coordinator->drain(1, recycler).ok());
  EXPECT_EQ(coordinator->active_admission_count(), 0U);
  EXPECT_EQ(recycler.observed.size(), 1U);
  EXPECT_EQ(recycler.last_event.handle, 30U);
}

TEST(QwenKvAdmissionCoordinatorTest,
     CleanCreditShortageDoesNotPublishOrFailController) {
  auto runtime = coordinator_runtime();
  auto pool = coordinator_pool(1);
  auto occupied = QwenKvAdmissionTransaction::Reserve(pool, 99, 1);
  ASSERT_TRUE(occupied.ok());
  ASSERT_TRUE(occupied->publish().ok());
  CoordinatorBackend backend;
  auto driver = ControllerPackedDriver::Create(runtime, backend, 2, {2, 8, 2});
  ASSERT_TRUE(driver.ok());
  auto coordinator = QwenKvAdmissionCoordinator::Create(
      pool, *driver, 2, {30, 1});
  ASSERT_TRUE(coordinator.ok());
  const std::array<std::uint32_t, 1> prompt{};
  ASSERT_TRUE(runtime.submit_admit(2, prompt, 1).ok());
  auto step = runtime.step(10, *coordinator);
  ASSERT_TRUE(step.ok());
  EXPECT_EQ(*step, ControllerRuntimeStep::kAdmissionRejected);
  EXPECT_EQ(runtime.state(), ControllerRuntimeState::kReady);
  EXPECT_EQ(coordinator->active_admission_count(), 0U);
  EXPECT_EQ(pool.clean_credits(), 0U);
  auto rejected = runtime.try_take_event().value();
  ASSERT_TRUE(rejected.has_value());
  EXPECT_EQ(rejected->kind, ControllerOutputEventKind::kFailed);
}

}  // namespace
}  // namespace pih
