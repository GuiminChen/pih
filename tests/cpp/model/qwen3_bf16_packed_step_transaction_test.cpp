#include "pih/model/qwen3_bf16_packed_step_transaction.h"

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace pih {
namespace {

struct PackedTrace final { std::vector<char> events; };

CudaCopyEndpoint tx_endpoint(std::uintptr_t address, std::uint64_t bytes,
                             std::uint64_t owner, CudaCopyMemoryType type) {
  return {address, bytes, 0, owner, 3, type, 0, 0};
}

class TxCopy final : public TypedCopyDriver {
 public:
  explicit TxCopy(PackedTrace& trace) : trace_(&trace) {}
  std::uintptr_t context_identity() const noexcept override { return 17; }
  Status copy(CudaCopyKind kind, std::uintptr_t, std::uintptr_t,
              std::uint64_t, DriverStreamHandle) override {
    trace_->events.push_back(kind == CudaCopyKind::kHostToDevice ? 'U' : 'D');
    return Status::Ok();
  }
 private:
  PackedTrace* trace_;
};

class TxCompute final : public QwenBf16StepCompute {
 public:
  explicit TxCompute(PackedTrace& trace) : trace_(&trace) {}
  Status submit(QwenBf16DeviceErrorClearDriver&, KernelLaunchDriver&,
                QwenBf16LinearExecutionDriver&, DriverStreamHandle) override {
    trace_->events.push_back('C');
    return result;
  }
  Status result = Status::Ok();
 private:
  PackedTrace* trace_;
};

class TxClear final : public QwenBf16DeviceErrorClearDriver {
 public:
  Status clear_u32_async(const TensorView&, std::int32_t,
                         DriverStreamHandle) override { return Status::Ok(); }
};
class TxKernels final : public KernelLaunchDriver {
 public:
  Status launch(DriverFunctionHandle, const KernelLaunchGeometry&,
                DriverStreamHandle, void**) override { return Status::Ok(); }
};
class TxLinears final : public QwenBf16LinearExecutionDriver {
 public:
  Status execute(const QwenBf16LinearBinding&,
                 DriverStreamHandle) override { return Status::Ok(); }
};
class TxEvents final : public CompletionEventDriver {
 public:
  explicit TxEvents(PackedTrace& trace) : trace_(&trace) {}
  Status record(DriverEventHandle, DriverStreamHandle) override {
    trace_->events.push_back('E');
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    return query_result;
  }
  CudaEventQueryResult query_result = CudaEventQueryResult::kNotReady;
 private:
  PackedTrace* trace_;
};
class TxHealth final : public QwenBf16StepHealthProvider {
 public:
  Result<QwenBf16StepHealth> collect() override {
    ++calls;
    return QwenBf16StepHealth{true, false};
  }
  int calls = 0;
};

Result<QwenBf16PackedStepTransaction> make_transaction(
    TxCompute& compute, TxHealth& health, std::span<std::byte> result,
    std::uint32_t sample_count = 2, std::uint64_t upload_event = 23) {
  auto staging =
      QwenBf16PackedStepStagingLayout::CreateBounded(4, 2, 2).value();
  auto upload = QwenBf16PackedStepUpload::Create(
      staging,
      tx_endpoint(0x100000, staging.total_bytes(), 11,
                  CudaCopyMemoryType::kRegisteredPinnedHost),
      tx_endpoint(0x200000, staging.total_bytes(), 12,
                  CudaCopyMemoryType::kDevice),
      17, 19, upload_event, 100).value();
  auto layout = QwenBf16PackedResultLayout::Create(2).value();
  auto readback = QwenBf16PackedReadback::Create(
      layout, sample_count,
      tx_endpoint(0x300000, 8, 13, CudaCopyMemoryType::kDevice),
      tx_endpoint(0x400000, 4, 14, CudaCopyMemoryType::kDevice),
      tx_endpoint(reinterpret_cast<std::uintptr_t>(result.data()),
                  result.size(), 15,
                  CudaCopyMemoryType::kRegisteredPinnedHost),
      17, 19, 23, 200).value();
  auto slot = CompletionEventSlot::Create(31, 17).value();
  auto frontier = CudaCompletionFrontier::Create(
      {1, 0, 2, CudaCompletionPhase::kDecode, 3}, 23, 100, 200).value();
  return QwenBf16PackedStepTransaction::Create(
      std::move(upload), compute, std::move(readback), layout, sample_count,
      std::move(slot), std::move(frontier), result, 19, 23, health);
}

TEST(QwenBf16PackedStepTransactionTest, OrdersAndPublishesWholeBatch) {
  auto layout = QwenBf16PackedResultLayout::Create(2).value();
  std::vector<std::byte> result(layout.total_bytes());
  PackedTrace trace;
  TxCompute compute(trace);
  TxHealth health;
  auto transaction = make_transaction(compute, health, result);
  ASSERT_TRUE(transaction.ok()) << transaction.status().message();
  TxCopy copies(trace); TxClear clear; TxKernels kernels; TxLinears linears;
  TxEvents events(trace);
  ASSERT_TRUE(transaction->submit(copies, clear, kernels, linears, events).ok());
  EXPECT_EQ(trace.events.size(), 18U);
  EXPECT_EQ(trace.events[14], 'C');
  EXPECT_EQ(trace.events[15], 'D');
  EXPECT_EQ(trace.events[16], 'D');
  EXPECT_EQ(trace.events[17], 'E');
  EXPECT_EQ(transaction->poll(events).status().code(), StatusCode::kUnavailable);
  const std::uint32_t tokens[]{7, 9};
  const std::uint32_t error = 0;
  std::memcpy(result.data(), tokens, sizeof(tokens));
  std::memcpy(result.data() + layout.device_error().offset_bytes, &error, 4);
  events.query_result = CudaEventQueryResult::kSuccess;
  auto published = transaction->poll(events);
  ASSERT_TRUE(published.ok()) << published.status().message();
  EXPECT_EQ(*published, (std::vector<std::uint32_t>{7, 9}));
  EXPECT_EQ(health.calls, 1);
  EXPECT_TRUE(transaction->release_completion().ok());
}

TEST(QwenBf16PackedStepTransactionTest, ComputeFailurePoisonsBeforeReadback) {
  auto layout = QwenBf16PackedResultLayout::Create(2).value();
  std::vector<std::byte> result(layout.total_bytes());
  PackedTrace trace; TxCompute compute(trace); TxHealth health;
  compute.result = Status::Internal("injected compute failure");
  auto transaction = make_transaction(compute, health, result).value();
  TxCopy copies(trace); TxClear clear; TxKernels kernels; TxLinears linears;
  TxEvents events(trace);
  EXPECT_FALSE(transaction.submit(copies, clear, kernels, linears, events).ok());
  EXPECT_EQ(trace.events.size(), 15U);
  EXPECT_EQ(trace.events.back(), 'C');
  EXPECT_EQ(transaction.state(), QwenBf16PackedStepTransactionState::kPoisoned);
  EXPECT_FALSE(transaction.submit(copies, clear, kernels, linears, events).ok());
}

TEST(QwenBf16PackedStepTransactionTest,
     ZeroSamplePublishesEmptyBatchAfterErrorOnlyReadback) {
  auto layout = QwenBf16PackedResultLayout::Create(2).value();
  std::vector<std::byte> result(layout.total_bytes());
  PackedTrace trace; TxCompute compute(trace); TxHealth health;
  auto transaction = make_transaction(compute, health, result, 0).value();
  TxCopy copies(trace); TxClear clear; TxKernels kernels; TxLinears linears;
  TxEvents events(trace);
  ASSERT_TRUE(transaction.submit(copies, clear, kernels, linears, events).ok());
  EXPECT_EQ(trace.events.size(), 17U);
  EXPECT_EQ(std::count(trace.events.begin(), trace.events.end(), 'D'), 1);
  const std::uint32_t error = 0;
  std::memcpy(result.data() + layout.device_error().offset_bytes, &error, 4);
  events.query_result = CudaEventQueryResult::kSuccess;
  auto published = transaction.poll(events);
  ASSERT_TRUE(published.ok()) << published.status().message();
  EXPECT_TRUE(published->empty());
}

TEST(QwenBf16PackedStepTransactionTest, RejectsCrossGenerationComposition) {
  auto layout = QwenBf16PackedResultLayout::Create(2).value();
  std::vector<std::byte> result(layout.total_bytes());
  PackedTrace trace; TxCompute compute(trace); TxHealth health;
  EXPECT_FALSE(make_transaction(compute, health, result, 2, 22).ok());
}

}  // namespace
}  // namespace pih
