#include <array>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_teacher_forced_chunk_transaction.h"
#include "pih/model/qwen3_teacher_forced_pair_completion.h"
#include "pih/model/qwen3_teacher_forced_pair_chunk_executor.h"
#include "pih/model/qwen3_teacher_forced_pair_run_executor.h"

namespace pih {
namespace {

struct ChunkTrace { std::vector<char> values; };

CudaCopyEndpoint chunk_ep(std::uintptr_t address, std::uint64_t bytes,
                          std::uint64_t owner, CudaCopyMemoryType type) {
  return {address, bytes, 0, owner, 3, type, 0, 0};
}
TensorView chunk_view(std::uintptr_t address, DType dtype,
                      std::span<const std::int64_t> shape) {
  return TensorView::Create(reinterpret_cast<void*>(address), dtype, shape, {},
      Device::Create(DeviceType::kCuda, 0).value(), 3).value();
}
ResolvedKernelFunction chunk_metric_function() {
  const std::string digest(64, 'c');
  auto manifest = qwen_bf16_kernel_manifest(
      QwenBf16Primitive::kTeacherForcedMetric, digest).value();
  return {std::string(qwen_bf16_kernel_symbol(
              QwenBf16Primitive::kTeacherForcedMetric).value()),
          std::string(manifest.logical_id()), digest,
          std::string(manifest.parameter_abi_sha256()), 73};
}
ResolvedKernelFunction chunk_gather_function() {
  const std::string digest(64, 'd');
  auto manifest = qwen_bf16_packed_kernel_manifest(
      QwenBf16PackedPrimitive::kSampleHidden, digest).value();
  return {std::string(qwen_bf16_packed_kernel_symbol(
              QwenBf16PackedPrimitive::kSampleHidden).value()),
          std::string(manifest.logical_id()), digest,
          std::string(manifest.parameter_abi_sha256()), 81};
}

class ChunkCopies final : public TypedCopyDriver {
 public:
  explicit ChunkCopies(ChunkTrace& trace) : trace_(&trace) {}
  std::uintptr_t context_identity() const noexcept override { return 17; }
  Status copy(CudaCopyKind kind, std::uintptr_t, std::uintptr_t,
              std::uint64_t, DriverStreamHandle) override {
    trace_->values.push_back(kind == CudaCopyKind::kHostToDevice ? 'U' : 'D');
    return Status::Ok();
  }
 private: ChunkTrace* trace_;
};
class ChunkClear final : public QwenBf16DeviceErrorClearDriver {
 public:
  explicit ChunkClear(ChunkTrace& trace) : trace_(&trace) {}
  Status clear_u32_async(const TensorView&, std::int32_t,
                         DriverStreamHandle) override {
    trace_->values.push_back('C'); return Status::Ok();
  }
 private: ChunkTrace* trace_;
};
class ChunkKernels final : public KernelLaunchDriver {
 public:
  explicit ChunkKernels(ChunkTrace& trace) : trace_(&trace) {}
  Status launch(DriverFunctionHandle, const KernelLaunchGeometry&,
                DriverStreamHandle, void**) override {
    trace_->values.push_back('K'); return Status::Ok();
  }
 private: ChunkTrace* trace_;
};
class ChunkBf16Head final : public QwenBf16LinearExecutionDriver {
 public:
  explicit ChunkBf16Head(ChunkTrace& trace) : trace_(&trace) {}
  Status execute(const QwenBf16LinearBinding&, DriverStreamHandle) override {
    trace_->values.push_back('L'); return result;
  }
  Status result = Status::Ok();
 private: ChunkTrace* trace_;
};
class ChunkInt4Head final : public QwenInt4LmHeadExecutionDriver {
 public:
  explicit ChunkInt4Head(ChunkTrace& trace) : trace_(&trace) {}
  Status execute(const QwenInt4LmHeadBinding&, DriverStreamHandle) override {
    trace_->values.push_back('L'); return Status::Ok();
  }
 private: ChunkTrace* trace_;
};
class ChunkEvents final : public CompletionEventDriver {
 public:
  explicit ChunkEvents(ChunkTrace& trace) : trace_(&trace) {}
  Status record(DriverEventHandle, DriverStreamHandle) override {
    trace_->values.push_back('E'); return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    return ready ? CudaEventQueryResult::kSuccess
                 : CudaEventQueryResult::kNotReady;
  }
  bool ready = false;
 private: ChunkTrace* trace_;
};
class ChunkHealth final : public QwenBf16StepHealthProvider {
 public:
  Result<QwenBf16StepHealth> collect() override {
    return QwenBf16StepHealth{true, false};
  }
};

Result<QwenTeacherForcedChunkTransaction> make_chunk_transaction(
    std::span<std::byte> result, ChunkHealth& health,
    std::uintptr_t metric_logits_address = 0x30000000) {
  auto layout = QwenTeacherForcedMetricResultLayout::Create(1).value();
  auto transfer = QwenTeacherForcedMetricTransfer::Create(
      layout, 1,
      chunk_ep(0x080000, 4, 5, CudaCopyMemoryType::kRegisteredPinnedHost),
      chunk_ep(0x180000, 4, 6, CudaCopyMemoryType::kDevice),
      chunk_ep(0x100000, 4, 1, CudaCopyMemoryType::kRegisteredPinnedHost),
      chunk_ep(0x200000, 4, 2, CudaCopyMemoryType::kDevice),
      chunk_ep(0x400000, 1024, 3, CudaCopyMemoryType::kDevice),
      chunk_ep(reinterpret_cast<std::uintptr_t>(result.data()), 1024, 4,
               CudaCopyMemoryType::kRegisteredPinnedHost),
      17, 19, 23, 100, 101, 102).value();
  const std::array<std::int64_t, 2> normalized_shape{3, 1024};
  const std::array<std::int64_t, 1> one{1};
  const std::array<std::int64_t, 1> four{4};
  const std::array<std::int64_t, 2> gathered_shape{1, 1024};
  const std::array<std::int64_t, 2> weight_shape{151936, 1024};
  const std::array<std::int64_t, 2> logits_shape{1, 151936};
  const auto error_address = 0x400000 + layout.device_error().offset_bytes;
  auto logits = chunk_view(0x30000000, DType::kFloat32, logits_shape);
  auto logits_plan = QwenTeacherForcedLogitsPlan::Create(
      chunk_gather_function(),
      chunk_view(0x500000, DType::kBFloat16, normalized_shape),
      chunk_view(0x180000, DType::kUInt32, one),
      chunk_view(0x600000, DType::kBFloat16, gathered_shape),
      chunk_view(0x10000000, DType::kBFloat16, weight_shape), logits,
      chunk_view(error_address, DType::kUInt8, four), 0).value();
  auto error = chunk_view(error_address, DType::kUInt32, one);
  auto metric_plan = QwenTeacherForcedMetricPlan::Create(
      chunk_metric_function(),
      chunk_view(metric_logits_address, DType::kFloat32, logits_shape),
      chunk_view(0x200000, DType::kUInt32, one),
      chunk_view(0x400000, DType::kUInt32, one),
      chunk_view(0x400000 + layout.target_nll().offset_bytes,
                 DType::kFloat64, one),
      chunk_view(0x400000 + layout.nonfinite_rows().offset_bytes,
                 DType::kUInt32, one), error, 0).value();
  auto slot = CompletionEventSlot::Create(31, 17).value();
  auto frontier = CudaCompletionFrontier::Create(
      {1, 0, 2, CudaCompletionPhase::kPrefill, 3}, 23, 100, 200).value();
  return QwenTeacherForcedChunkTransaction::Create(
      std::move(transfer), std::move(logits_plan), std::move(metric_plan),
      error, layout, std::move(slot), std::move(frontier), result,
      19, 23, 0, health);
}

TEST(QwenTeacherForcedChunkTransactionTest,
     OrdersBf16LogitsAndMetricBeforePublication) {
  alignas(256) std::array<std::byte, 1024> result{};
  ChunkHealth health;
  auto transaction = make_chunk_transaction(result, health).value();
  ChunkTrace trace; ChunkCopies copies(trace); ChunkClear clear(trace);
  ChunkKernels kernels(trace); ChunkBf16Head head(trace); ChunkEvents events(trace);
  ASSERT_TRUE(transaction.submit_bf16(
      copies, clear, kernels, head, events).ok());
  EXPECT_EQ(transaction.submitted_role(),
            QwenTeacherForcedChunkRole::kBf16);
  EXPECT_EQ(trace.values,
            (std::vector<char>{'U','U','C','K','L','K','D','E'}));
}

TEST(QwenTeacherForcedChunkTransactionTest, SupportsIdenticalInt4HeadFlow) {
  alignas(256) std::array<std::byte, 1024> result{};
  ChunkHealth health;
  auto transaction = make_chunk_transaction(result, health).value();
  ChunkTrace trace; ChunkCopies copies(trace); ChunkClear clear(trace);
  ChunkKernels kernels(trace); ChunkInt4Head head(trace); ChunkEvents events(trace);
  ASSERT_TRUE(transaction.submit_int4(
      copies, clear, kernels, head, events).ok());
  EXPECT_EQ(transaction.submitted_role(),
            QwenTeacherForcedChunkRole::kInt4);
  EXPECT_EQ(trace.values,
            (std::vector<char>{'U','U','C','K','L','K','D','E'}));
}

TEST(QwenTeacherForcedChunkTransactionTest,
     HeadFailureSuppressesMetricAndPublication) {
  alignas(256) std::array<std::byte, 1024> result{};
  ChunkHealth health;
  auto transaction = make_chunk_transaction(result, health).value();
  ChunkTrace trace; ChunkCopies copies(trace); ChunkClear clear(trace);
  ChunkKernels kernels(trace); ChunkBf16Head head(trace); ChunkEvents events(trace);
  head.result = Status::Internal("injected head");
  EXPECT_FALSE(transaction.submit_bf16(
      copies, clear, kernels, head, events).ok());
  EXPECT_EQ(trace.values, (std::vector<char>{'U','U','C','K','L'}));
  EXPECT_EQ(transaction.state(),
            QwenTeacherForcedChunkTransactionState::kPoisoned);
  EXPECT_EQ(transaction.submitted_role(),
            QwenTeacherForcedChunkRole::kNone);
}

TEST(QwenTeacherForcedChunkTransactionTest, RejectsDifferentMetricLogits) {
  alignas(256) std::array<std::byte, 1024> result{};
  ChunkHealth health;
  EXPECT_FALSE(make_chunk_transaction(result, health, 0x31000000).ok());
}

TEST(QwenTeacherForcedChunkTransactionTest,
     PublishesOnlyAfterAuthorizedCompletion) {
  alignas(256) std::array<std::byte, 1024> result{};
  ChunkHealth health;
  auto transaction = make_chunk_transaction(result, health).value();
  ChunkTrace trace; ChunkCopies copies(trace); ChunkClear clear(trace);
  ChunkKernels kernels(trace); ChunkBf16Head head(trace); ChunkEvents events(trace);
  ASSERT_TRUE(transaction.submit_bf16(
      copies, clear, kernels, head, events).ok());
  EXPECT_EQ(transaction.poll(events).status().code(), StatusCode::kUnavailable);

  auto layout = QwenTeacherForcedMetricResultLayout::Create(1).value();
  const std::uint32_t zero = 0;
  const std::uint32_t token = 9;
  const double nll = 0.75;
  std::memcpy(result.data() + layout.device_error().offset_bytes, &zero, 4);
  std::memcpy(result.data() + layout.argmax_tokens().offset_bytes, &token, 4);
  std::memcpy(result.data() + layout.target_nll().offset_bytes, &nll, 8);
  std::memcpy(result.data() + layout.nonfinite_rows().offset_bytes, &zero, 4);
  events.ready = true;
  auto batch = transaction.poll(events);
  ASSERT_TRUE(batch.ok()) << batch.status().message();
  EXPECT_DOUBLE_EQ(batch->nll_sum, 0.75);
  EXPECT_EQ(batch->rows[0].argmax_token, 9U);
  EXPECT_TRUE(transaction.release_completion().ok());
}

std::array<QwenTeacherForcedChunk, 6> completion_chunks() {
  std::array<QwenTeacherForcedChunk, 6> chunks{};
  for (std::size_t index = 0; index < chunks.size(); ++index) {
    chunks[index] = {static_cast<QwenTeacherForcedCategory>(index), 0, 1,
        QwenTeacherForcedChunkPlan::kVocabularySize * sizeof(float)};
  }
  return chunks;
}

void publish_chunk_result(std::span<std::byte> result, std::uint32_t token,
                          double nll) {
  const auto layout = QwenTeacherForcedMetricResultLayout::Create(1).value();
  const std::uint32_t zero = 0;
  std::memcpy(result.data() + layout.device_error().offset_bytes, &zero, 4);
  std::memcpy(result.data() + layout.argmax_tokens().offset_bytes, &token, 4);
  std::memcpy(result.data() + layout.target_nll().offset_bytes, &nll, 8);
  std::memcpy(result.data() + layout.nonfinite_rows().offset_bytes, &zero, 4);
}

TEST(QwenTeacherForcedPairCompletionTest,
     PollsReleasesAndPairsNativeTransactionRoles) {
  const auto chunks = completion_chunks();
  auto completion = QwenTeacherForcedPairCompletion::Create(chunks).value();
  ChunkTrace trace; ChunkCopies copies(trace); ChunkClear clear(trace);
  ChunkKernels kernels(trace); ChunkEvents events(trace); ChunkHealth health;

  alignas(256) std::array<std::byte, 1024> bf16_result{};
  auto bf16 = make_chunk_transaction(bf16_result, health).value();
  ChunkBf16Head bf16_head(trace);
  ASSERT_TRUE(bf16.submit_bf16(
      copies, clear, kernels, bf16_head, events).ok());
  EXPECT_EQ(completion.collect(chunks[0], bf16, events).code(),
            StatusCode::kUnavailable);
  EXPECT_EQ(completion.expire(bf16, 150).code(), StatusCode::kUnavailable);
  EXPECT_FALSE(completion.poisoned());
  publish_chunk_result(bf16_result, 7, 0.25);
  events.ready = true;
  ASSERT_TRUE(completion.collect(chunks[0], bf16, events).ok());

  alignas(256) std::array<std::byte, 1024> int4_result{};
  auto int4 = make_chunk_transaction(int4_result, health).value();
  ChunkInt4Head int4_head(trace);
  ASSERT_TRUE(int4.submit_int4(
      copies, clear, kernels, int4_head, events).ok());
  publish_chunk_result(int4_result, 7, 0.5);
  ASSERT_TRUE(completion.collect(chunks[0], int4, events).ok());
  EXPECT_EQ(completion.completed_pairs(), 1U);
}

TEST(QwenTeacherForcedPairCompletionTest, RejectsInt4AsFirstRole) {
  const auto chunks = completion_chunks();
  auto completion = QwenTeacherForcedPairCompletion::Create(chunks).value();
  ChunkTrace trace; ChunkCopies copies(trace); ChunkClear clear(trace);
  ChunkKernels kernels(trace); ChunkEvents events(trace); ChunkHealth health;
  alignas(256) std::array<std::byte, 1024> result{};
  auto transaction = make_chunk_transaction(result, health).value();
  ChunkInt4Head head(trace);
  ASSERT_TRUE(transaction.submit_int4(
      copies, clear, kernels, head, events).ok());
  EXPECT_FALSE(completion.collect(chunks[0], transaction, events).ok());
  EXPECT_TRUE(completion.poisoned());
}

TEST(QwenTeacherForcedPairCompletionTest, DeadlinePoisonsPairedRun) {
  const auto chunks = completion_chunks();
  auto completion = QwenTeacherForcedPairCompletion::Create(chunks).value();
  ChunkTrace trace; ChunkCopies copies(trace); ChunkClear clear(trace);
  ChunkKernels kernels(trace); ChunkEvents events(trace); ChunkHealth health;
  alignas(256) std::array<std::byte, 1024> result{};
  auto transaction = make_chunk_transaction(result, health).value();
  ChunkBf16Head head(trace);
  ASSERT_TRUE(transaction.submit_bf16(
      copies, clear, kernels, head, events).ok());
  EXPECT_FALSE(completion.expire(transaction, 200).ok());
  EXPECT_TRUE(completion.poisoned());
}

TEST(QwenTeacherForcedPairChunkExecutorTest,
     OwnsSequentialBf16AndInt4TransactionLifecycle) {
  const auto chunks = completion_chunks();
  auto executor = QwenTeacherForcedPairChunkExecutor::Create(chunks).value();
  ChunkTrace trace; ChunkCopies copies(trace); ChunkClear clear(trace);
  ChunkKernels kernels(trace); ChunkEvents events(trace); ChunkHealth health;

  alignas(256) std::array<std::byte, 1024> bf16_result{};
  ChunkBf16Head bf16_head(trace);
  ASSERT_TRUE(executor.submit_bf16(
      chunks[0], make_chunk_transaction(bf16_result, health).value(),
      copies, clear, kernels, bf16_head, events).ok());
  EXPECT_EQ(executor.state(),
            QwenTeacherForcedPairChunkExecutorState::kBf16Pending);
  EXPECT_EQ(executor.poll(events).code(), StatusCode::kUnavailable);
  publish_chunk_result(bf16_result, 7, 0.25);
  events.ready = true;
  ASSERT_TRUE(executor.poll(events).ok());
  EXPECT_EQ(executor.state(),
            QwenTeacherForcedPairChunkExecutorState::kAwaitingInt4);

  alignas(256) std::array<std::byte, 1024> int4_result{};
  ChunkInt4Head int4_head(trace);
  ASSERT_TRUE(executor.submit_int4(
      chunks[0], make_chunk_transaction(int4_result, health).value(),
      copies, clear, kernels, int4_head, events).ok());
  publish_chunk_result(int4_result, 7, 0.5);
  ASSERT_TRUE(executor.poll(events).ok());
  EXPECT_EQ(executor.state(),
            QwenTeacherForcedPairChunkExecutorState::kAwaitingBf16);
  EXPECT_EQ(executor.completed_pairs(), 1U);
}

TEST(QwenTeacherForcedPairChunkExecutorTest, RejectsInt4BeforeBf16) {
  const auto chunks = completion_chunks();
  auto executor = QwenTeacherForcedPairChunkExecutor::Create(chunks).value();
  ChunkTrace trace; ChunkCopies copies(trace); ChunkClear clear(trace);
  ChunkKernels kernels(trace); ChunkEvents events(trace); ChunkHealth health;
  alignas(256) std::array<std::byte, 1024> result{};
  ChunkInt4Head head(trace);
  EXPECT_FALSE(executor.submit_int4(
      chunks[0], make_chunk_transaction(result, health).value(),
      copies, clear, kernels, head, events).ok());
  EXPECT_EQ(executor.state(),
            QwenTeacherForcedPairChunkExecutorState::kPoisoned);
}

TEST(QwenTeacherForcedPairRunExecutorTest,
     DrivesFrozenPlanToSixFinalizedOwnedAggregates) {
  const auto chunks = completion_chunks();
  auto runner = QwenTeacherForcedPairRunExecutor::Create(chunks).value();
  ChunkTrace trace; ChunkCopies copies(trace); ChunkClear clear(trace);
  ChunkKernels kernels(trace); ChunkEvents events(trace); ChunkHealth health;
  ChunkBf16Head bf16_head(trace); ChunkInt4Head int4_head(trace);

  for (std::size_t index = 0; index < chunks.size(); ++index) {
    ASSERT_EQ(runner.next_chunk()->category,
              static_cast<QwenTeacherForcedCategory>(index));
    alignas(256) std::array<std::byte, 1024> bf16_result{};
    ASSERT_TRUE(runner.submit_bf16(
        make_chunk_transaction(bf16_result, health).value(),
        copies, clear, kernels, bf16_head, events).ok());
    publish_chunk_result(bf16_result, 7, 0.25);
    events.ready = true;
    ASSERT_TRUE(runner.poll(events).ok());

    alignas(256) std::array<std::byte, 1024> int4_result{};
    ASSERT_TRUE(runner.submit_int4(
        make_chunk_transaction(int4_result, health).value(),
        copies, clear, kernels, int4_head, events).ok());
    publish_chunk_result(int4_result, 7, 0.5);
    ASSERT_TRUE(runner.poll(events).ok());
  }
  EXPECT_FALSE(runner.next_chunk().ok());
  auto aggregates = runner.finalize();
  ASSERT_TRUE(aggregates.ok()) << aggregates.status().message();
  EXPECT_EQ(aggregates->size(), 6U);
  for (const auto& aggregate : *aggregates) {
    EXPECT_EQ(aggregate.evaluated_positions, 1U);
    EXPECT_EQ(aggregate.paired_nll_deltas.values().size(), 1U);
  }
}

}  // namespace
}  // namespace pih
