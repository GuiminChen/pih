#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>

#include <gtest/gtest.h>

#include "pih/model/qwen3_teacher_forced_transaction_factory.h"
#include "pih/model/qwen3_teacher_forced_transaction_identity.h"
#include "pih/model/qwen3_teacher_forced_pair_run_driver.h"

namespace pih {
namespace {

class FactoryDeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++calls;
    return Allocation{reinterpret_cast<void*>(
        UINT64_C(0x100000000) + UINT64_C(0x100000000) * calls),
        bytes, alignment, static_cast<std::uint64_t>(calls),
        Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation) noexcept override {}
  int calls = 0;
};

class FactoryPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    void* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t(alignment), std::nothrow);
    if (data == nullptr) return Status::ResourceExhausted("test allocation");
    return Allocation{data, bytes, alignment, ++generation, Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override {
    ::operator delete(value.data, std::align_val_t(value.alignment));
  }
  std::uint64_t generation = 100;
};

class FactoryPlacement final : public PinnedPlacementVerifier {
 public:
  Status verify(const void*, std::uint64_t bytes, std::int32_t numa) override {
    return bytes != 0 && numa == 2 ? Status::Ok()
                                  : Status::InvalidArgument("placement");
  }
};

TensorView factory_view(std::uintptr_t address, DType dtype,
                        std::span<const std::int64_t> shape) {
  return TensorView::Create(reinterpret_cast<void*>(address), dtype, shape, {},
      Device::Create(DeviceType::kCuda, 0).value(), 3).value();
}

ResolvedKernelFunction factory_function(QwenBf16PackedPrimitive primitive,
                                        char digest_character,
                                        DriverFunctionHandle handle) {
  const std::string digest(64, digest_character);
  auto manifest = qwen_bf16_packed_kernel_manifest(primitive, digest).value();
  return {std::string(qwen_bf16_packed_kernel_symbol(primitive).value()),
          std::string(manifest.logical_id()), digest,
          std::string(manifest.parameter_abi_sha256()), handle};
}

ResolvedKernelFunction factory_metric_function() {
  const std::string digest(64, 'e');
  auto manifest = qwen_bf16_kernel_manifest(
      QwenBf16Primitive::kTeacherForcedMetric, digest).value();
  return {std::string(qwen_bf16_kernel_symbol(
              QwenBf16Primitive::kTeacherForcedMetric).value()),
          std::string(manifest.logical_id()), digest,
          std::string(manifest.parameter_abi_sha256()), 82};
}

class FactoryHealth final : public QwenBf16StepHealthProvider {
 public:
  Result<QwenBf16StepHealth> collect() override {
    return QwenBf16StepHealth{true, false};
  }
};

class FactoryCopies final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 17; }
  Status copy(CudaCopyKind, std::uintptr_t, std::uintptr_t, std::uint64_t,
              DriverStreamHandle) override { return Status::Ok(); }
};

class FactoryClear final : public QwenBf16DeviceErrorClearDriver {
 public:
  Status clear_u32_async(const TensorView&, std::int32_t,
                         DriverStreamHandle) override { return Status::Ok(); }
};

class FactoryKernels final : public KernelLaunchDriver {
 public:
  Status launch(DriverFunctionHandle, const KernelLaunchGeometry&,
                DriverStreamHandle, void**) override { return Status::Ok(); }
};

class FactoryBf16Head final : public QwenBf16LinearExecutionDriver {
 public:
  Status execute(const QwenBf16LinearBinding&,
                 DriverStreamHandle) override { return Status::Ok(); }
};

class FactoryInt4Head final : public QwenInt4LmHeadExecutionDriver {
 public:
  Status execute(const QwenInt4LmHeadBinding&,
                 DriverStreamHandle) override { return Status::Ok(); }
};

class FactoryEvents final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle, DriverStreamHandle) override {
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    return ready ? CudaEventQueryResult::kSuccess
                 : CudaEventQueryResult::kNotReady;
  }
  bool ready = false;
};

std::array<QwenTeacherForcedChunk, 6> factory_chunks() {
  std::array<QwenTeacherForcedChunk, 6> chunks{};
  for (std::size_t index = 0; index < chunks.size(); ++index) {
    chunks[index] = {
        static_cast<QwenTeacherForcedCategory>(index), 0, 1,
        QwenTeacherForcedChunkPlan::kVocabularySize * sizeof(float)};
  }
  return chunks;
}

TEST(QwenTeacherForcedTransactionFactoryTest,
     MaterializesBatchAndBuildsFrozenTransaction) {
  FactoryDeviceAllocator device;
  FactoryPinnedAllocator pinned;
  FactoryPlacement placement;
  auto arenas = QwenTeacherForcedMetricArenas::AllocateVerified(
      2, device, pinned, placement, 0, 2).value();
  const std::array<std::int64_t, 2> normalized_shape{5, 1024};
  const std::array<std::int64_t, 2> gathered_shape{2, 1024};
  const std::array<std::int64_t, 2> weight_shape{151936, 1024};
  const std::array<std::int64_t, 2> logits_shape{2, 151936};
  auto factory = QwenTeacherForcedTransactionFactory::Create(
      arenas,
      factory_function(QwenBf16PackedPrimitive::kSampleHidden, 'd', 81),
      factory_metric_function(),
      factory_view(0x50000000, DType::kBFloat16, normalized_shape),
      factory_view(0x60000000, DType::kBFloat16, gathered_shape),
      factory_view(0x70000000, DType::kBFloat16, weight_shape),
      factory_view(0x90000000, DType::kFloat32, logits_shape),
      17, 19, 0, {10, 11, 12, 13, 14, 15}).value();
  auto peer_factory = QwenTeacherForcedTransactionFactory::Create(
      arenas,
      factory_function(QwenBf16PackedPrimitive::kSampleHidden, 'd', 81),
      factory_metric_function(),
      factory_view(0x50000000, DType::kBFloat16, normalized_shape),
      factory_view(0x60000000, DType::kBFloat16, gathered_shape),
      factory_view(0x70000000, DType::kBFloat16, weight_shape),
      factory_view(0x90000000, DType::kFloat32, logits_shape),
      17, 19, 0, {20, 21, 22, 23, 24, 25}).value();
  EXPECT_EQ(factory.capability().arena_identity,
            peer_factory.capability().arena_identity);
  EXPECT_EQ(factory.capability().factory_generation, 1U);
  EXPECT_EQ(peer_factory.capability().factory_generation, 2U);
  const auto factory_copy = factory;
  EXPECT_EQ(factory_copy.capability(), factory.capability());
  const QwenTeacherForcedChunk chunk{
      QwenTeacherForcedCategory::kCode, 0, 2,
      2ULL * QwenTeacherForcedChunkPlan::kVocabularySize * sizeof(float)};
  const std::array<std::uint32_t, 1> sequence_tokens{5};
  const std::array<QwenTeacherForcedTarget, 2> targets{{
      {0, 1, 7}, {0, 4, 11},
  }};
  auto batch = QwenTeacherForcedBatchPlan::Create(
      chunk, sequence_tokens, targets).value();
  const auto identity =
      reserve_qwen_teacher_forced_transaction_identity(100, 23).value();
  auto completion = make_qwen_teacher_forced_transaction_completion(
      identity, 31, 17, 1, 0, 3, 100, 200).value();
  class Health final : public QwenBf16StepHealthProvider {
   public:
    Result<QwenBf16StepHealth> collect() override {
      return QwenBf16StepHealth{true, false};
    }
  } health;
  auto transaction = factory.build_reserved(
      batch, identity, std::move(completion), health);
  ASSERT_TRUE(transaction.ok()) << transaction.status().message();
  EXPECT_EQ(transaction->state(),
            QwenTeacherForcedChunkTransactionState::kPrepared);
  std::array<std::uint32_t, 2> rows{}, ids{};
  std::memcpy(rows.data(), arenas.pinned_sample_rows().data(), sizeof(rows));
  std::memcpy(ids.data(), arenas.pinned_targets().data(), sizeof(ids));
  EXPECT_EQ(rows, (std::array<std::uint32_t, 2>{1, 4}));
  EXPECT_EQ(ids, (std::array<std::uint32_t, 2>{7, 11}));

  std::ranges::fill(arenas.pinned_sample_rows().first(8), std::byte{0x5a});
  const auto overlapping_identity =
      reserve_qwen_teacher_forced_transaction_identity(300, 40).value();
  auto overlapping_completion =
      make_qwen_teacher_forced_transaction_completion(
          overlapping_identity, 31, 17, 1, 0, 5, 100, 200).value();
  EXPECT_FALSE(peer_factory.build_reserved(
      batch, overlapping_identity, std::move(overlapping_completion),
      health).ok());
  EXPECT_TRUE(std::ranges::all_of(
      arenas.pinned_sample_rows().first(8),
      [](std::byte value) { return value == std::byte{0x5a}; }));

  const auto wrong_rank_identity =
      reserve_qwen_teacher_forced_transaction_identity(200, 30).value();
  auto wrong_rank_completion =
      make_qwen_teacher_forced_transaction_completion(
          wrong_rank_identity, 31, 17, 1, 1, 4, 100, 200).value();
  EXPECT_FALSE(factory.build_reserved(
      batch, wrong_rank_identity, std::move(wrong_rank_completion),
      health).ok());

  std::ranges::fill(arenas.pinned_targets().first(8), std::byte{0x6b});
  auto wrong_slot = CompletionEventSlot::Create(32, 18).value();
  auto next_frontier = CudaCompletionFrontier::Create(
      {1, 0, 2, CudaCompletionPhase::kPrefill, 4}, 24, 100, 200).value();
  EXPECT_FALSE(factory.build(
      batch, 24, 103, 104, 105, std::move(wrong_slot),
      std::move(next_frontier), health).ok());
  EXPECT_TRUE(std::ranges::all_of(
      arenas.pinned_sample_rows().first(8),
      [](std::byte value) { return value == std::byte{0x5a}; }));
  EXPECT_TRUE(std::ranges::all_of(
      arenas.pinned_targets().first(8),
      [](std::byte value) { return value == std::byte{0x6b}; }));
}

TEST(QwenTeacherForcedPairRunDriverTest,
     CommitsReservedIdentitiesAndFreezesFactoryCapability) {
  FactoryDeviceAllocator device;
  FactoryPinnedAllocator pinned;
  FactoryPlacement placement;
  auto arenas = QwenTeacherForcedMetricArenas::AllocateVerified(
      1, device, pinned, placement, 0, 2).value();
  const std::array<std::int64_t, 2> normalized_shape{2, 1024};
  const std::array<std::int64_t, 2> gathered_shape{1, 1024};
  const std::array<std::int64_t, 2> weight_shape{151936, 1024};
  const std::array<std::int64_t, 2> logits_shape{1, 151936};
  auto factory = QwenTeacherForcedTransactionFactory::Create(
      arenas,
      factory_function(QwenBf16PackedPrimitive::kSampleHidden, 'd', 81),
      factory_metric_function(),
      factory_view(0x50000000, DType::kBFloat16, normalized_shape),
      factory_view(0x60000000, DType::kBFloat16, gathered_shape),
      factory_view(0x70000000, DType::kBFloat16, weight_shape),
      factory_view(0x90000000, DType::kFloat32, logits_shape),
      17, 19, 0, {10, 11, 12, 13, 14, 15}).value();
  auto other_factory = QwenTeacherForcedTransactionFactory::Create(
      arenas,
      factory_function(QwenBf16PackedPrimitive::kSampleHidden, 'd', 81),
      factory_metric_function(),
      factory_view(0x50000000, DType::kBFloat16, normalized_shape),
      factory_view(0x60000000, DType::kBFloat16, gathered_shape),
      factory_view(0x70000000, DType::kBFloat16, weight_shape),
      factory_view(0x90000000, DType::kFloat32, logits_shape),
      17, 19, 0, {10, 11, 12, 13, 14, 15}).value();
  const auto chunks = factory_chunks();
  const std::array<std::uint32_t, 1> sequence_tokens{2};
  const std::array<QwenTeacherForcedTarget, 1> targets{{{0, 1, 7}}};
  auto batch = QwenTeacherForcedBatchPlan::Create(
      chunks[0], sequence_tokens, targets).value();
  auto driver = QwenTeacherForcedPairRunDriver::Create(
      chunks, {1, 0, 31, 17, 100, 23, 3}).value();
  FactoryHealth health;
  FactoryCopies copies;
  FactoryClear clear;
  FactoryKernels kernels;
  FactoryBf16Head head;
  FactoryInt4Head int4_head;
  FactoryEvents events;

  ASSERT_TRUE(driver.submit_bf16(
      batch, factory, 100, 200, health, copies, clear, kernels, head,
      events).ok());
  EXPECT_EQ(driver.next_plan_id(), 104U);
  EXPECT_EQ(driver.next_event_generation(), 24U);
  EXPECT_EQ(driver.next_completion_frontier(), 4U);
  EXPECT_FALSE(driver.poisoned());

  std::ranges::fill(arenas.pinned_sample_rows().first(4), std::byte{0x5a});
  EXPECT_FALSE(driver.submit_int4(
      batch, factory, 101, 201, health, copies, clear, kernels,
      int4_head, events).ok());
  EXPECT_TRUE(std::ranges::all_of(
      arenas.pinned_sample_rows().first(4),
      [](std::byte value) { return value == std::byte{0x5a}; }));
  EXPECT_EQ(driver.next_plan_id(), 104U);
  EXPECT_FALSE(driver.poisoned());

  const auto layout = QwenTeacherForcedMetricResultLayout::Create(1).value();
  const std::uint32_t zero = 0;
  const std::uint32_t token = 7;
  const double nll = 0.25;
  auto result = arenas.pinned_result();
  std::memcpy(result.data() + layout.device_error().offset_bytes, &zero, 4);
  std::memcpy(result.data() + layout.argmax_tokens().offset_bytes, &token, 4);
  std::memcpy(result.data() + layout.target_nll().offset_bytes, &nll, 8);
  std::memcpy(result.data() + layout.nonfinite_rows().offset_bytes, &zero, 4);
  events.ready = true;
  ASSERT_TRUE(driver.poll(events).ok());

  EXPECT_FALSE(driver.submit_int4(
      batch, other_factory, 102, 202, health, copies, clear, kernels,
      int4_head, events).ok());
  EXPECT_EQ(driver.next_plan_id(), 104U);
  EXPECT_EQ(driver.next_event_generation(), 24U);
  EXPECT_EQ(driver.next_completion_frontier(), 4U);
  EXPECT_FALSE(driver.poisoned());

  ASSERT_TRUE(driver.submit_int4(
      batch, factory, 103, 203, health, copies, clear, kernels,
      int4_head, events).ok());
  EXPECT_EQ(driver.next_plan_id(), 108U);
  EXPECT_EQ(driver.next_event_generation(), 25U);
  EXPECT_EQ(driver.next_completion_frontier(), 5U);
  const double int4_nll = 0.5;
  result = arenas.pinned_result();
  std::memcpy(result.data() + layout.device_error().offset_bytes, &zero, 4);
  std::memcpy(result.data() + layout.argmax_tokens().offset_bytes, &token, 4);
  std::memcpy(result.data() + layout.target_nll().offset_bytes, &int4_nll, 8);
  std::memcpy(result.data() + layout.nonfinite_rows().offset_bytes, &zero, 4);
  ASSERT_TRUE(driver.poll(events).ok());
  EXPECT_EQ(driver.phase(),
            QwenTeacherForcedPairRunDriverPhase::kAwaitingBf16);

  for (std::size_t index = 1; index < chunks.size(); ++index) {
    auto next_batch = QwenTeacherForcedBatchPlan::Create(
        chunks[index], sequence_tokens, targets).value();
    ASSERT_TRUE(driver.submit_bf16(
        next_batch, factory, 200 + index * 10, 300 + index * 10,
        health, copies, clear, kernels, head, events).ok());
    result = arenas.pinned_result();
    std::memcpy(result.data() + layout.device_error().offset_bytes, &zero, 4);
    std::memcpy(result.data() + layout.argmax_tokens().offset_bytes, &token, 4);
    std::memcpy(result.data() + layout.target_nll().offset_bytes, &nll, 8);
    std::memcpy(result.data() + layout.nonfinite_rows().offset_bytes, &zero, 4);
    ASSERT_TRUE(driver.poll(events).ok());

    ASSERT_TRUE(driver.submit_int4(
        next_batch, factory, 201 + index * 10, 301 + index * 10,
        health, copies, clear, kernels, int4_head, events).ok());
    result = arenas.pinned_result();
    std::memcpy(result.data() + layout.device_error().offset_bytes, &zero, 4);
    std::memcpy(result.data() + layout.argmax_tokens().offset_bytes, &token, 4);
    std::memcpy(
        result.data() + layout.target_nll().offset_bytes, &int4_nll, 8);
    std::memcpy(result.data() + layout.nonfinite_rows().offset_bytes, &zero, 4);
    ASSERT_TRUE(driver.poll(events).ok());
  }
  EXPECT_EQ(driver.next_plan_id(), 148U);
  EXPECT_EQ(driver.next_event_generation(), 35U);
  EXPECT_EQ(driver.next_completion_frontier(), 15U);
  auto aggregates = driver.finalize();
  ASSERT_TRUE(aggregates.ok()) << aggregates.status().message();
  for (const auto& aggregate : *aggregates) {
    EXPECT_EQ(aggregate.evaluated_positions, 1U);
    EXPECT_EQ(aggregate.equal_argmax_positions, 1U);
    EXPECT_EQ(aggregate.paired_nll_deltas.values().size(), 1U);
  }
  EXPECT_EQ(driver.phase(),
            QwenTeacherForcedPairRunDriverPhase::kFinalized);
}

TEST(QwenTeacherForcedPairRunDriverTest,
     RejectsBatchDriftWithoutConsumingIdentities) {
  const auto chunks = factory_chunks();
  auto driver = QwenTeacherForcedPairRunDriver::Create(
      chunks, {1, 0, 31, 17, 100, 23, 3}).value();
  const std::array<std::uint32_t, 1> sequence_tokens{2};
  const std::array<QwenTeacherForcedTarget, 1> targets{{{0, 1, 7}}};
  auto wrong_chunk = chunks[0];
  wrong_chunk.category_row_begin = 1;
  auto batch = QwenTeacherForcedBatchPlan::Create(
      wrong_chunk, sequence_tokens, targets).value();
  FactoryDeviceAllocator device;
  FactoryPinnedAllocator pinned;
  FactoryPlacement placement;
  auto arenas = QwenTeacherForcedMetricArenas::AllocateVerified(
      1, device, pinned, placement, 0, 2).value();
  const std::array<std::int64_t, 2> normalized_shape{2, 1024};
  const std::array<std::int64_t, 2> gathered_shape{1, 1024};
  const std::array<std::int64_t, 2> weight_shape{151936, 1024};
  const std::array<std::int64_t, 2> logits_shape{1, 151936};
  auto factory = QwenTeacherForcedTransactionFactory::Create(
      arenas,
      factory_function(QwenBf16PackedPrimitive::kSampleHidden, 'd', 81),
      factory_metric_function(),
      factory_view(0x50000000, DType::kBFloat16, normalized_shape),
      factory_view(0x60000000, DType::kBFloat16, gathered_shape),
      factory_view(0x70000000, DType::kBFloat16, weight_shape),
      factory_view(0x90000000, DType::kFloat32, logits_shape),
      17, 19, 0, {10, 11, 12, 13, 14, 15}).value();
  FactoryHealth health;
  FactoryCopies copies;
  FactoryClear clear;
  FactoryKernels kernels;
  FactoryBf16Head head;
  FactoryEvents events;

  EXPECT_FALSE(driver.submit_bf16(
      batch, factory, 100, 200, health, copies, clear, kernels, head,
      events).ok());
  EXPECT_EQ(driver.next_plan_id(), 100U);
  EXPECT_EQ(driver.next_event_generation(), 23U);
  EXPECT_EQ(driver.next_completion_frontier(), 3U);
  EXPECT_FALSE(driver.poisoned());
}

TEST(QwenTeacherForcedPairRunDriverTest,
     RejectsIdentitySeedsThatCannotCoverTheCompleteRun) {
  const auto chunks = factory_chunks();
  const auto reservations = static_cast<std::uint64_t>(chunks.size()) * 2U;
  const auto maximum = std::numeric_limits<std::uint64_t>::max();

  EXPECT_TRUE(QwenTeacherForcedPairRunDriver::Create(
      chunks, {1, 0, 31, 17, maximum - reservations * 4U,
                    maximum - reservations, maximum - reservations}).ok());
  EXPECT_EQ(QwenTeacherForcedPairRunDriver::Create(
      chunks, {1, 0, 31, 17, maximum - reservations * 4U + 1U,
                    1, 1}).status().code(),
            StatusCode::kResourceExhausted);
  EXPECT_EQ(QwenTeacherForcedPairRunDriver::Create(
      chunks, {1, 0, 31, 17, 1, maximum - reservations + 1U,
                    1}).status().code(),
            StatusCode::kResourceExhausted);
  EXPECT_EQ(QwenTeacherForcedPairRunDriver::Create(
      chunks, {1, 0, 31, 17, 1, 1,
                    maximum - reservations + 1U}).status().code(),
            StatusCode::kResourceExhausted);
}

}  // namespace
}  // namespace pih
