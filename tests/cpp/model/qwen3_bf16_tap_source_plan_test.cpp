#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_bf16_step_resource_factory.h"
#include "pih/model/qwen3_bf16_tap_fixture_execution.h"
#include "pih/model/qwen3_bf16_tap_fixture_preparation.h"
#include "pih/model/qwen3_bf16_tap_source_plan.h"

namespace pih {
namespace {

Qwen3Config config() {
  return {1024, 3072, 28, 16, 8, 128, 151936, 40960,
          1'000'000.0, 0.000001, 151643, 151645};
}

struct Fixture final {
  QwenBf16CommandBuffer commands;
  QwenBf16ResourceSet resources;
  QwenKvBlockTable table;
  std::vector<QwenKvSlotState> states;
};

class TapDeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    return Allocation{reinterpret_cast<void*>(UINT64_C(0x400000000)), bytes,
                      alignment, 71,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation) noexcept override {}
};

class TapPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    void* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t(alignment), std::nothrow);
    if (data == nullptr) return Status::ResourceExhausted("test allocation");
    std::memset(data, 0, static_cast<std::size_t>(bytes));
    return Allocation{data, bytes, alignment, 72, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ::operator delete(allocation.data,
                      std::align_val_t(allocation.alignment));
  }
};

class TapPlacement final : public PinnedPlacementVerifier {
 public:
  Status verify(const void*, std::uint64_t, std::int32_t numa) override {
    return numa == 2 ? Status::Ok()
                     : Status::FailedPrecondition("wrong NUMA node");
  }
};

class TapCopyDriver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 99; }
  Status copy(CudaCopyKind kind, std::uintptr_t destination, std::uintptr_t,
              std::uint64_t bytes, DriverStreamHandle) override {
    kinds.push_back(kind);
    if (kind == CudaCopyKind::kDeviceToHost && bytes == sizeof(std::int64_t)) {
      std::memcpy(reinterpret_cast<void*>(destination), &sampled_token,
                  sizeof(sampled_token));
    }
    if (kind == CudaCopyKind::kDeviceToHost && bytes == sizeof(std::uint32_t)) {
      std::uint32_t error = 0;
      std::memcpy(reinterpret_cast<void*>(destination), &error, sizeof(error));
    }
    return Status::Ok();
  }
  std::vector<CudaCopyKind> kinds;
  std::int64_t sampled_token = 7;
};

class TapNoopClear final : public QwenBf16DeviceErrorClearDriver {
 public:
  Status clear_u32_async(const TensorView&, std::int32_t,
                         DriverStreamHandle) override { return Status::Ok(); }
};

class TapNoopKernel final : public KernelLaunchDriver {
 public:
  Status launch(DriverFunctionHandle, const KernelLaunchGeometry&,
                DriverStreamHandle, void**) override {
    return Status::Ok();
  }
};

class TapNoopLinear final : public QwenBf16LinearExecutionDriver {
 public:
  Status execute(const QwenBf16LinearBinding&,
                 DriverStreamHandle) override { return Status::Ok(); }
};

class TapEventDriver final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle event, DriverStreamHandle stream) override {
    events.push_back(event);
    streams.push_back(stream);
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    return CudaEventQueryResult::kSuccess;
  }
  std::vector<DriverEventHandle> events;
  std::vector<DriverStreamHandle> streams;
};

class TapEvidence final : public CompletionEvidenceProvider {
 public:
  Result<CompletionPublicationEvidence> collect() override {
    return CompletionPublicationEvidence{true, 0, false};
  }
};

class TapHealth final : public QwenBf16StepHealthProvider {
 public:
  Result<QwenBf16StepHealth> collect() override {
    return QwenBf16StepHealth{true, false};
  }
};

class TapClock final : public QwenBf16MonotonicClock {
 public:
  Result<std::uint64_t> now_ns() override { return ++now; }
  std::uint64_t now = 100;
};

class TapWaiter final : public QwenBf16PollWaiter {
 public:
  Status wait() override { return Status::Ok(); }
};

class TapCompute final : public QwenBf16InstrumentedStepCompute {
 public:
  Status submit_instrumented_and_record(
      QwenBf16DeviceErrorClearDriver&, KernelLaunchDriver&,
      QwenBf16LinearExecutionDriver&, QwenBf16TapSnapshotDriver& snapshots,
      const QwenBf16TapBindingPlan& bindings,
      QwenBf16TapSnapshotFrontierRecorder& recorder,
      CompletionEventDriver& events, DriverStreamHandle stream,
      std::uint64_t submit_ns, std::uint64_t deadline_ns) override {
    for (const auto& binding : bindings) {
      const Status status = snapshots.snapshot(binding, stream);
      if (!status.ok()) return status;
    }
    return recorder.record_snapshot(events, submit_ns, deadline_ns);
  }
};

Fixture fixture() {
  auto schedule = QwenBf16ExecutionSchedule::Create(config()).value();
  auto weights = QwenBf16WeightBindingPlan::Create(schedule).value();
  auto commands = QwenBf16CommandBuffer::Create(schedule, weights).value();
  const std::array<QwenKvBlockHandle, 2> handles{{{1, 7}, {3, 9}}};
  auto table = QwenKvBlockTable::Create(23, 4, 32, handles).value();
  auto append = table.prepare_append(18).value();
  const std::array<std::int64_t, 18> tokens{};
  auto input = QwenBf16StepInputPlan::Create(tokens, 0, table, append).value();
  auto staging = QwenBf16StepStagingLayout::Create(input).value();
  auto arena = QwenBf16ExecutionArenaLayout::Create(18, 1).value();
  const QwenBf16StepDeviceOwners owners{
      {0x2100000000, staging.total_bytes(), 11},
      {0x2200000000, arena.activation_arena_bytes(), 12},
      {0x2300000000, arena.mlp().arena_bytes(), 13},
      {0x2400000000, arena.rope_workspace_bytes(), 14},
      {0x2500000000, arena.logit_workspace_bytes(), 15},
      {0x2600000000, sizeof(std::int64_t), 16},
      {0x2700000000, sizeof(std::uint32_t), 11},
      {0x2800000000, 4 * QwenKvSlotPool::kSlotPayloadBytes, 18},
      {0x2900000000, 4 * sizeof(QwenKvSlotState), 11}};
  auto resources = QwenBf16StepResourceFactory::Create(
                       11, 0, 4, arena, staging, owners)
                       .value();
  std::vector<QwenKvSlotState> states(4);
  states[1] = {7, 23, 16, QwenKvSlotLifecycle::kOwned, 0, 0};
  states[3] = {9, 23, 2, QwenKvSlotLifecycle::kOwned, 0, 0};
  return {std::move(commands), std::move(resources), std::move(table),
          std::move(states)};
}

TEST(QwenBf16TapSourcePlanTest, MapsActivationRowsAndPhysicalKvPlanes) {
  auto data = fixture();
  const std::array requests{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLayerHidden, 0, 2, 2},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kKvKey, 5, 17, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kKvValue, 5, 17, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLogits, 28, 17, 1},
  };
  auto taps = QwenNumericalTapPlan::Create(requests, 1ULL << 20).value();
  auto bindings =
      QwenBf16TapBindingPlan::Create(taps, data.commands).value();

  auto sources = QwenBf16TapSourcePlan::Create(
      taps, bindings, data.resources, data.table, data.states, 0, 1000);

  ASSERT_TRUE(sources.ok()) << sources.status().message();
  ASSERT_EQ(sources->size(), 4);
  EXPECT_EQ((*sources)[0].offset, 2 * 1024 * 2);
  const std::uint64_t slot_base = 3 * QwenKvSlotPool::kSlotPayloadBytes;
  const std::uint64_t layer_base = 5 * QwenKvAddressMapper::kBytesPerLayer;
  const std::uint64_t token_base = QwenKvAddressMapper::kBytesPerToken;
  EXPECT_EQ((*sources)[1].offset, slot_base + layer_base + token_base);
  EXPECT_EQ((*sources)[2].offset,
            slot_base + layer_base + QwenKvAddressMapper::kBytesPerPlane +
                token_base);
  EXPECT_EQ((*sources)[3].offset, 0);
  EXPECT_EQ((*sources)[1].generation, 18);
  EXPECT_EQ((*sources)[0].rank, 0);
}

TEST(QwenBf16TapSourcePlanTest, RejectsStaleKvAndOutOfStepActivation) {
  auto data = fixture();
  const std::array kv_request{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kKvKey, 5, 17, 1}};
  auto kv_taps =
      QwenNumericalTapPlan::Create(kv_request, 1ULL << 20).value();
  auto kv_bindings =
      QwenBf16TapBindingPlan::Create(kv_taps, data.commands).value();
  data.states[3].generation = 8;
  EXPECT_FALSE(QwenBf16TapSourcePlan::Create(
                   kv_taps, kv_bindings, data.resources, data.table,
                   data.states, 0, 1000)
                   .ok());

  const std::array activation_request{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLayerHidden, 0, 18, 1}};
  auto activation_taps =
      QwenNumericalTapPlan::Create(activation_request, 1ULL << 20).value();
  auto activation_bindings = QwenBf16TapBindingPlan::Create(
                                 activation_taps, data.commands)
                                 .value();
  EXPECT_FALSE(QwenBf16TapSourcePlan::Create(
                   activation_taps, activation_bindings, data.resources,
                   data.table, data.states, 0, 1000)
                   .ok());
}

TEST(QwenBf16TapFixturePreparationTest,
     AtomicallyBindsStepResourcesToTransferPipeline) {
  auto data = fixture();
  const std::array requests{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLayerHidden, 0, 2, 2},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kKvKey, 5, 17, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kKvValue, 5, 17, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLogits, 28, 17, 1},
  };
  auto taps = QwenNumericalTapPlan::Create(requests, 1ULL << 20).value();
  TapDeviceAllocator device;
  TapPinnedAllocator pinned;
  TapPlacement placement;
  const QwenBf16TapTransferIdentity identity{
      31, 11, 101, 102, 201, 202, 99, 301, 302, 401, 402, 0};

  auto prepared = QwenBf16TapFixturePreparation::Create(
      taps, data.commands, data.resources, data.table, data.states, 0, 500,
      device, pinned, placement, 2, identity, 601, 1);

  ASSERT_TRUE(prepared.ok()) << prepared.status().message();
  EXPECT_EQ(prepared->bindings().size(), requests.size());
  EXPECT_EQ(prepared->arenas().arena_bytes(), taps.arena_bytes());
  EXPECT_EQ(prepared->pipeline().state(),
            QwenBf16TapEventPipelineState::kCapturing);
  auto drift = identity;
  drift.producer_plan_generation = 12;
  EXPECT_FALSE(QwenBf16TapFixturePreparation::Create(
      taps, data.commands, data.resources, data.table, data.states, 0, 500,
      device, pinned, placement, 2, drift, 601, 1).ok());
}

TEST(QwenBf16TapFixtureExecutionTest, RunsUploadSnapshotAndHostSealOnce) {
  auto data = fixture();
  const std::array requests{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLayerHidden, 0, 2, 2},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kKvKey, 5, 17, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kKvValue, 5, 17, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLogits, 28, 17, 1},
  };
  auto taps = QwenNumericalTapPlan::Create(requests, 1ULL << 20).value();
  TapDeviceAllocator device;
  TapPinnedAllocator pinned;
  TapPlacement placement;
  const QwenBf16TapTransferIdentity transfer{
      31, 11, 101, 102, 201, 202, 99, 301, 302, 401, 402, 0};
  auto preparation = QwenBf16TapFixturePreparation::Create(
      taps, data.commands, data.resources, data.table, data.states, 0, 500,
      device, pinned, placement, 2, transfer, 601, 1).value();
  auto append = data.table.prepare_append(18).value();
  const std::array<std::int64_t, 18> tokens{};
  auto input = QwenBf16StepInputPlan::Create(
      tokens, 0, data.table, append).value();
  auto staging = QwenBf16StepStagingLayout::Create(input).value();
  const CudaCopyEndpoint host{0x100000000, staging.total_bytes(), 0, 1, 1,
                              CudaCopyMemoryType::kRegisteredPinnedHost, 0, 2};
  const CudaCopyEndpoint gpu{0x2100000000, staging.total_bytes(), 0, 2, 1,
                             CudaCopyMemoryType::kDevice, 0, 0};
  auto upload = QwenBf16StepUpload::Create(
      staging, host, gpu, 99, 301, 201, 701).value();
  alignas(256) std::array<std::byte, QwenBf16StepResultLayout::kTotalBytes>
      result_backing{};
  const CudaCopyEndpoint sampled{0x2600000000, sizeof(std::int64_t), 0, 3, 1,
                                 CudaCopyMemoryType::kDevice, 0, 0};
  const CudaCopyEndpoint device_error{0x2700000000, sizeof(std::uint32_t), 0,
                                      4, 11, CudaCopyMemoryType::kDevice, 0,
                                      0};
  const CudaCopyEndpoint result{
      reinterpret_cast<std::uintptr_t>(result_backing.data()),
      result_backing.size(), 0, 5, 1,
      CudaCopyMemoryType::kRegisteredPinnedHost, 0, 2};
  auto readback = QwenBf16StepReadback::Create(
      sampled, device_error, result, 99, 302, 202, 801).value();
  TapCompute compute;
  TapCopyDriver copies;
  TapNoopClear clear;
  TapNoopKernel kernels;
  TapNoopLinear linears;
  TapEventDriver events;
  TapEvidence evidence;
  TapHealth health;
  TapClock clock;
  TapWaiter waiter;
  auto execution = QwenBf16TapFixtureExecution::Create(
      std::move(upload), std::move(readback), result_backing, compute,
      std::move(preparation), {301, 601, 1000},
      {&copies, &clear, &kernels, &linears, &events, &evidence, &health,
       &clock, &waiter}).value();

  auto receipt = execution.run();

  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(receipt->tap_run.capture_count, 4);
  EXPECT_EQ(receipt->completion.handle, 601);
  EXPECT_EQ(receipt->completion.generation, 202);
  EXPECT_EQ(receipt->sampled_token, 7);
  EXPECT_EQ(execution.state(), QwenBf16TapFixtureExecutionState::kSealed);
  EXPECT_EQ(copies.kinds.size(), 15);
  EXPECT_EQ(events.streams,
            (std::vector<DriverStreamHandle>{301, 302}));
  EXPECT_FALSE(execution.run().ok());
}

}  // namespace
}  // namespace pih
