#include "pih/model/qwen3_bf16_step_transaction.h"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

namespace pih {
namespace {

struct Trace final { std::vector<char> events; };

QwenBf16StepStagingLayout staging_layout() {
  const QwenKvBlockHandle handles[] = {{4, 9}};
  auto table = QwenKvBlockTable::Create(2, 6, 16, handles).value();
  auto append = table.prepare_append(1).value();
  const std::int64_t token[] = {4};
  auto input = QwenBf16StepInputPlan::Create(token, 0, table, append).value();
  return QwenBf16StepStagingLayout::Create(input).value();
}

CudaCopyEndpoint endpoint(std::uintptr_t address, std::uint64_t bytes,
                          std::uint64_t owner, CudaCopyMemoryType type) {
  return {address, bytes, 0, owner, 3, type, 0, 0};
}

class CopyDriver final : public TypedCopyDriver {
 public:
  explicit CopyDriver(Trace& trace) : trace_(&trace) {}
  std::uintptr_t context_identity() const noexcept override { return 17; }
  Status copy(CudaCopyKind kind, std::uintptr_t, std::uintptr_t,
              std::uint64_t, DriverStreamHandle) override {
    trace_->events.push_back(kind == CudaCopyKind::kHostToDevice ? 'U' : 'D');
    return Status::Ok();
  }
 private:
  Trace* trace_;
};

class Compute final : public QwenBf16StepCompute {
 public:
  explicit Compute(Trace& trace) : trace_(&trace) {}
  Status submit(QwenBf16DeviceErrorClearDriver&, KernelLaunchDriver&,
                QwenBf16LinearExecutionDriver&,
                DriverStreamHandle) override {
    trace_->events.push_back('C');
    return result;
  }
  Status result = Status::Ok();
 private:
  Trace* trace_;
};

class ClearDriver final : public QwenBf16DeviceErrorClearDriver {
 public:
  Status clear_u32_async(const TensorView&, std::int32_t,
                         DriverStreamHandle) override { return Status::Ok(); }
};
class KernelDriver final : public KernelLaunchDriver {
 public:
  Status launch(DriverFunctionHandle, const KernelLaunchGeometry&,
                DriverStreamHandle, void**) override { return Status::Ok(); }
};
class LinearDriver final : public QwenBf16LinearExecutionDriver {
 public:
  Status execute(const QwenBf16LinearBinding&,
                 DriverStreamHandle) override { return Status::Ok(); }
};

class EventDriver final : public CompletionEventDriver {
 public:
  explicit EventDriver(Trace& trace) : trace_(&trace) {}
  Status record(DriverEventHandle, DriverStreamHandle) override {
    trace_->events.push_back('E');
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    return query_result;
  }
  CudaEventQueryResult query_result = CudaEventQueryResult::kNotReady;
 private:
  Trace* trace_;
};

class Health final : public QwenBf16StepHealthProvider {
 public:
  Result<QwenBf16StepHealth> collect() override {
    ++calls;
    return QwenBf16StepHealth{true, false};
  }
  int calls = 0;
};

Result<QwenBf16StepTransaction> transaction(
    Compute& compute, Health& health, std::span<std::byte> result) {
  auto layout = staging_layout();
  auto upload = QwenBf16StepUpload::Create(
      layout,
      endpoint(0x100000, layout.total_bytes(), 11,
               CudaCopyMemoryType::kRegisteredPinnedHost),
      endpoint(0x200000, layout.total_bytes(), 12,
               CudaCopyMemoryType::kDevice),
      17, 19, 23, 100).value();
  auto readback = QwenBf16StepReadback::Create(
      endpoint(0x300000, 8, 13, CudaCopyMemoryType::kDevice),
      endpoint(0x400000, 4, 14, CudaCopyMemoryType::kDevice),
      endpoint(reinterpret_cast<std::uintptr_t>(result.data()), result.size(),
               15, CudaCopyMemoryType::kRegisteredPinnedHost),
      17, 19, 23, 200).value();
  auto slot = CompletionEventSlot::Create(31, 17).value();
  auto frontier = CudaCompletionFrontier::Create(
      {1, 0, 2, CudaCompletionPhase::kDecode, 3}, 23, 100, 200).value();
  return QwenBf16StepTransaction::Create(
      std::move(upload), compute, std::move(readback), std::move(slot),
      std::move(frontier), result, 19, 23, health);
}

template <typename T>
void write(std::span<std::byte> backing, QwenBf16ArenaSpan span, T value) {
  std::memcpy(backing.data() + span.offset_bytes, &value, sizeof(value));
}

TEST(QwenBf16StepTransactionTest, OrdersSubmissionAndPublishesAfterEvent) {
  alignas(256) std::array<std::byte, QwenBf16StepResultLayout::kTotalBytes>
      result{};
  Trace trace;
  Compute compute(trace);
  Health health;
  auto step = transaction(compute, health, result);
  ASSERT_TRUE(step.ok()) << step.status().message();
  CopyDriver copies(trace);
  ClearDriver clear;
  KernelDriver kernels;
  LinearDriver linears;
  EventDriver events(trace);
  ASSERT_TRUE(step->submit(copies, clear, kernels, linears, events).ok());
  EXPECT_EQ(trace.events,
            (std::vector<char>{'U','U','U','U','U','C','D','D','E'}));
  EXPECT_EQ(step->poll(events).status().code(), StatusCode::kUnavailable);
  EXPECT_EQ(health.calls, 0);
  write(std::span<std::byte>(result), QwenBf16StepResultLayout::sampled_token(),
        std::int64_t{42});
  write(std::span<std::byte>(result), QwenBf16StepResultLayout::device_error(),
        std::uint32_t{0});
  events.query_result = CudaEventQueryResult::kSuccess;
  auto token = step->poll(events);
  ASSERT_TRUE(token.ok());
  EXPECT_EQ(*token, 42);
  EXPECT_EQ(health.calls, 1);
  EXPECT_EQ(step->state(), QwenBf16StepTransactionState::kCompleted);
  EXPECT_TRUE(step->release_completion().ok());
}

TEST(QwenBf16StepTransactionTest, ComputeFailurePreventsReadbackAndEvent) {
  alignas(256) std::array<std::byte, QwenBf16StepResultLayout::kTotalBytes>
      result{};
  Trace trace;
  Compute compute(trace);
  compute.result = Status::Internal("compute failed");
  Health health;
  auto step = transaction(compute, health, result).value();
  CopyDriver copies(trace);
  ClearDriver clear;
  KernelDriver kernels;
  LinearDriver linears;
  EventDriver events(trace);
  EXPECT_FALSE(step.submit(copies, clear, kernels, linears, events).ok());
  EXPECT_EQ(trace.events,
            (std::vector<char>{'U','U','U','U','U','C'}));
  EXPECT_EQ(step.state(), QwenBf16StepTransactionState::kPoisoned);
  EXPECT_FALSE(step.submit(copies, clear, kernels, linears, events).ok());
}

TEST(QwenBf16StepTransactionTest, DeadlinePoisonsPendingTransaction) {
  alignas(256) std::array<std::byte, QwenBf16StepResultLayout::kTotalBytes>
      result{};
  Trace trace;
  Compute compute(trace);
  Health health;
  auto step = transaction(compute, health, result).value();
  CopyDriver copies(trace);
  ClearDriver clear;
  KernelDriver kernels;
  LinearDriver linears;
  EventDriver events(trace);
  ASSERT_TRUE(step.submit(copies, clear, kernels, linears, events).ok());
  EXPECT_EQ(step.expire(199).code(), StatusCode::kUnavailable);
  EXPECT_FALSE(step.expire(200).ok());
  EXPECT_EQ(step.state(), QwenBf16StepTransactionState::kPoisoned);
  EXPECT_FALSE(step.poll(events).ok());
}

TEST(QwenBf16StepTransactionTest, RejectsCrossGenerationComposition) {
  alignas(256) std::array<std::byte, QwenBf16StepResultLayout::kTotalBytes>
      result{};
  Trace trace;
  Compute compute(trace);
  Health health;
  auto layout = staging_layout();
  auto upload = QwenBf16StepUpload::Create(
      layout,
      endpoint(0x100000, layout.total_bytes(), 11,
               CudaCopyMemoryType::kRegisteredPinnedHost),
      endpoint(0x200000, layout.total_bytes(), 12,
               CudaCopyMemoryType::kDevice),
      17, 19, 22, 100).value();
  auto readback = QwenBf16StepReadback::Create(
      endpoint(0x300000, 8, 13, CudaCopyMemoryType::kDevice),
      endpoint(0x400000, 4, 14, CudaCopyMemoryType::kDevice),
      endpoint(reinterpret_cast<std::uintptr_t>(result.data()), result.size(),
               15, CudaCopyMemoryType::kRegisteredPinnedHost),
      17, 19, 23, 200).value();
  auto slot = CompletionEventSlot::Create(31, 17).value();
  auto frontier = CudaCompletionFrontier::Create(
      {1, 0, 2, CudaCompletionPhase::kDecode, 3}, 23, 100, 200).value();
  EXPECT_FALSE(QwenBf16StepTransaction::Create(
                   std::move(upload), compute, std::move(readback),
                   std::move(slot), std::move(frontier), result, 19, 23,
                   health)
                   .ok());
}

}  // namespace
}  // namespace pih
