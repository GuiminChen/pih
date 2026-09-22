#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#include <gtest/gtest.h>

#include "pih/model/qwen3_bf16_tap_evidence_run.h"
#include "pih/model/qwen3_bf16_tap_event_pipeline.h"

namespace pih {
namespace {

class DeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    return Allocation{reinterpret_cast<void*>(UINT64_C(0x400000000)), bytes,
                      alignment, 41,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation) noexcept override {}
};

class PinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    void* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t(alignment), std::nothrow);
    if (data == nullptr) return Status::ResourceExhausted("test allocation");
    std::memset(data, 0, static_cast<std::size_t>(bytes));
    return Allocation{data, bytes, alignment, 73, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ::operator delete(allocation.data,
                      std::align_val_t(allocation.alignment));
  }
};

class Placement final : public PinnedPlacementVerifier {
 public:
  Status verify(const void*, std::uint64_t, std::int32_t numa) override {
    return numa == 2 ? Status::Ok()
                     : Status::FailedPrecondition("wrong NUMA node");
  }
};

class CopyDriver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 99; }
  Status copy(CudaCopyKind kind, std::uintptr_t, std::uintptr_t,
              std::uint64_t, DriverStreamHandle) override {
    kinds.push_back(kind);
    return Status::Ok();
  }
  std::vector<CudaCopyKind> kinds;
};

class EventDriver final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle event,
                DriverStreamHandle stream) override {
    events.push_back(event);
    streams.push_back(stream);
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    ++queries;
    return queries == 1 ? CudaEventQueryResult::kNotReady
                        : CudaEventQueryResult::kSuccess;
  }
  std::vector<DriverEventHandle> events;
  std::vector<DriverStreamHandle> streams;
  int queries = 0;
};

class Evidence final : public CompletionEvidenceProvider {
 public:
  Result<CompletionPublicationEvidence> collect() override {
    return CompletionPublicationEvidence{true, 0, false};
  }
};

CudaCompletionFrontier frontier(std::uint64_t plan, std::uint64_t event) {
  auto result = CudaCompletionFrontier::Create(
                    {1, 0, plan, CudaCompletionPhase::kCopy, plan}, event, 1, 2)
                    .value();
  EXPECT_TRUE(result.observe(event, CudaEventQueryResult::kSuccess, true, 0,
                             false)
                  .ok());
  return result;
}

struct RunFixture final {
  QwenNumericalTapPlan taps;
  QwenNumericalTapArenas arenas;
  QwenBf16TapEvidenceRun run;
};

RunFixture fixture(DeviceAllocator& device, PinnedAllocator& pinned,
                   Placement& placement) {
  const std::array requests{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLayerHidden, 0, 0, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLogits, 28, 0, 1},
  };
  auto taps = QwenNumericalTapPlan::Create(requests, 1ULL << 20).value();
  auto arenas = QwenNumericalTapArenas::AllocateVerified(
                    taps, device, pinned, placement, 0, 2)
                    .value();
  const std::array<CudaCopyEndpoint, 2> endpoints{{
      {0x500000000, 4096, 0, 10, 11, CudaCopyMemoryType::kDevice, 0, 0},
      {0x600000000, 151936 * 4, 0, 12, 13,
       CudaCopyMemoryType::kDevice, 0, 0},
  }};
  const QwenBf16TapTransferIdentity identity{
      7, 90, 91, 92, 501, 502, 99, 123, 124, 200, 300, 0};
  auto transfers = QwenBf16TapTransferPlan::Create(
                       taps, endpoints, arenas, identity)
                       .value();
  auto run = QwenBf16TapEvidenceRun::Create(taps, std::move(transfers)).value();
  return {std::move(taps), std::move(arenas), std::move(run)};
}

QwenBf16TapBinding binding(std::size_t index,
                           QwenNumericalTapRequest request) {
  return {index, request, index, {QwenBf16ExecutionOp::kEmbedding, 0}, 0,
          index == 0 ? QwenBf16ActivationSlot::kHidden
                     : QwenBf16ActivationSlot::kLogits,
          QwenBf16TapKvComponent::kNotApplicable};
}

TEST(QwenBf16TapEvidenceRunTest, DuplicateInlineDriverPoisonsRun) {
  DeviceAllocator device;
  PinnedAllocator pinned;
  Placement placement;
  auto data = fixture(device, pinned, placement);
  CopyDriver copies;
  auto inline_driver = data.run.inline_snapshot_driver(copies).value();
  (void)inline_driver;
  EXPECT_FALSE(data.run.inline_snapshot_driver(copies).ok());
  EXPECT_EQ(data.run.state(), QwenBf16TapEvidenceRunState::kPoisoned);
}

TEST(QwenBf16TapEvidenceRunTest, SealsCompleteOneShotCapture) {
  DeviceAllocator device;
  PinnedAllocator pinned;
  Placement placement;
  auto data = fixture(device, pinned, placement);
  CopyDriver copies;
  auto inline_driver = data.run.inline_snapshot_driver(copies).value();
  ASSERT_TRUE(inline_driver.snapshot(
                  binding(0, data.taps.captures()[0].request), 123)
                  .ok());
  ASSERT_TRUE(inline_driver.snapshot(
                  binding(1, data.taps.captures()[1].request), 123)
                  .ok());
  ASSERT_TRUE(data.run.complete_snapshot_batch(frontier(91, 501)).ok());
  ASSERT_TRUE(data.run.submit_host_batch(copies).ok());
  ASSERT_TRUE(data.run.complete_host_batch(frontier(92, 502)).ok());

  auto receipt = data.run.seal(data.arenas);

  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(receipt->capture_count, 2);
  EXPECT_EQ(receipt->run_generation, 7);
  EXPECT_EQ(data.run.state(), QwenBf16TapEvidenceRunState::kSealed);
  EXPECT_EQ(copies.kinds,
            (std::vector<CudaCopyKind>{CudaCopyKind::kDeviceToDevice,
                                       CudaCopyKind::kDeviceToDevice,
                                       CudaCopyKind::kDeviceToHost,
                                       CudaCopyKind::kDeviceToHost}));
}

TEST(QwenBf16TapEvidenceRunTest, MissingSnapshotPoisonsWholeBatch) {
  DeviceAllocator device;
  PinnedAllocator pinned;
  Placement placement;
  auto data = fixture(device, pinned, placement);
  CopyDriver copies;
  auto inline_driver = data.run.inline_snapshot_driver(copies).value();
  ASSERT_TRUE(inline_driver.snapshot(
                  binding(0, data.taps.captures()[0].request), 123)
                  .ok());

  EXPECT_FALSE(data.run.complete_snapshot_batch(frontier(91, 501)).ok());
  EXPECT_EQ(data.run.state(), QwenBf16TapEvidenceRunState::kPoisoned);
  EXPECT_FALSE(data.run.submit_host_batch(copies).ok());
}

TEST(QwenBf16TapEventPipelineTest,
     ReusesEventOnlyAfterSnapshotGenerationCompletes) {
  DeviceAllocator device;
  PinnedAllocator pinned;
  Placement placement;
  auto data = fixture(device, pinned, placement);
  auto pipeline = QwenBf16TapEventPipeline::Create(
                      std::move(data.run), 700, 3)
                      .value();
  CopyDriver copies;
  EventDriver events;
  Evidence evidence;
  auto inline_driver = pipeline.inline_snapshot_driver(copies).value();
  ASSERT_TRUE(inline_driver.snapshot(
                  binding(0, data.taps.captures()[0].request), 123)
                  .ok());
  ASSERT_TRUE(inline_driver.snapshot(
                  binding(1, data.taps.captures()[1].request), 123)
                  .ok());
  ASSERT_TRUE(pipeline.record_snapshot(events, 10, 20).ok());
  EXPECT_EQ(pipeline.poll_snapshot(events, evidence).code(),
            StatusCode::kUnavailable);
  ASSERT_TRUE(pipeline.poll_snapshot(events, evidence).ok());
  ASSERT_TRUE(
      pipeline.submit_host_and_record(copies, events, 21, 30).ok());
  ASSERT_TRUE(pipeline.poll_host(events, evidence).ok());

  auto receipt = pipeline.seal(data.arenas);

  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(pipeline.state(), QwenBf16TapEventPipelineState::kSealed);
  EXPECT_EQ(events.events,
            std::vector<DriverEventHandle>({700, 700}));
  EXPECT_EQ(events.streams,
            std::vector<DriverStreamHandle>({123, 124}));
  EXPECT_EQ(events.queries, 3);
}

TEST(QwenBf16TapEventPipelineTest, DeadlineEqualityPoisonsPendingSnapshot) {
  DeviceAllocator device;
  PinnedAllocator pinned;
  Placement placement;
  auto data = fixture(device, pinned, placement);
  auto pipeline = QwenBf16TapEventPipeline::Create(
                      std::move(data.run), 700, 3)
                      .value();
  CopyDriver copies;
  EventDriver events;
  auto inline_driver = pipeline.inline_snapshot_driver(copies).value();
  ASSERT_TRUE(inline_driver.snapshot(
                  binding(0, data.taps.captures()[0].request), 123)
                  .ok());
  ASSERT_TRUE(inline_driver.snapshot(
                  binding(1, data.taps.captures()[1].request), 123)
                  .ok());
  ASSERT_TRUE(pipeline.record_snapshot(events, 10, 20).ok());

  EXPECT_FALSE(pipeline.expire(20).ok());
  EXPECT_EQ(pipeline.state(), QwenBf16TapEventPipelineState::kPoisoned);
}

}  // namespace
}  // namespace pih
