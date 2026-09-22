#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_semantic_observation_finalizer.h"

namespace pih {
namespace {

class PinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    void* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t(alignment), std::nothrow);
    if (data == nullptr) return Status::ResourceExhausted("test allocation");
    ++allocations;
    return Allocation{data, bytes, alignment,
                      static_cast<std::uint64_t>(allocations), Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ++releases;
    ::operator delete(allocation.data,
                      std::align_val_t(allocation.alignment));
  }
  int allocations = 0;
  int releases = 0;
};

class Placement final : public PinnedPlacementVerifier {
 public:
  Status verify(const void*, std::uint64_t bytes,
                std::int32_t numa_node) override {
    return bytes != 0 && numa_node == 2
               ? Status::Ok()
               : Status::FailedPrecondition("bad placement");
  }
};

class CopyDriver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 41; }
  Status copy(CudaCopyKind, std::uintptr_t source,
              std::uintptr_t destination, std::uint64_t bytes,
              DriverStreamHandle) override {
    std::memcpy(reinterpret_cast<void*>(destination),
                reinterpret_cast<const void*>(source),
                static_cast<std::size_t>(bytes));
    ++calls;
    return Status::Ok();
  }
  std::size_t calls = 0;
};

class Events final : public CompletionEventDriver {
 public:
  Status record(DriverEventHandle, DriverStreamHandle) override {
    return Status::Ok();
  }
  Result<CudaEventQueryResult> query(DriverEventHandle) override {
    return CudaEventQueryResult::kSuccess;
  }
};

class Evidence final : public CompletionEvidenceProvider {
 public:
  Result<CompletionPublicationEvidence> collect() override {
    return value;
  }
  CompletionPublicationEvidence value{true, 0, false};
};

class Clock final : public QwenBf16MonotonicClock {
 public:
  Result<std::uint64_t> now_ns() override { return ++now; }
  std::uint64_t now = 100;
};

class Waiter final : public QwenBf16PollWaiter {
 public:
  Status wait() override { return Status::Ok(); }
};

QwenKvSemanticObservationPlan kv_plan() {
  const std::array<QwenKvBlockHandle, 1> handles{{{2, 11}}};
  auto table = QwenKvBlockTable::Create(4, 7, 1, handles).value();
  EXPECT_TRUE(table.commit_append(table.prepare_append(1).value()).ok());
  std::vector<QwenKvSlotState> states(
      4, {1, QwenKvSlotPool::kNoOwner, 0,
          QwenKvSlotLifecycle::kFreeClean, 0, 0});
  states[2] = {11, 4, 1, QwenKvSlotLifecycle::kOwned, 0, 0};
  return QwenKvSemanticObservationPlan::Create(
             table, states, 4 * QwenKvSlotPool::kSlotPayloadBytes)
      .value();
}

CudaCopyEndpoint endpoint(std::vector<std::byte>& bytes, std::uint64_t owner,
                          CudaCopyMemoryType type) {
  return {reinterpret_cast<std::uintptr_t>(bytes.data()), bytes.size(), 0,
          owner, 1, type, 0, 0};
}

TEST(QwenSemanticObservationFinalizerTest,
     AtomicallyPublishesFinalObservationAndCapacity) {
  const auto plan = kv_plan();
  std::vector<std::byte> logits(
      QwenSemanticObservationTransfer::kFinalLogitsBytes, std::byte{3});
  std::vector<std::byte> kv(4 * QwenKvSlotPool::kSlotPayloadBytes,
                            std::byte{7});
  PinnedAllocator allocator;
  Placement placement;
  CopyDriver copy;
  Events events;
  Evidence evidence;
  Clock clock;
  Waiter waiter;
  QwenSemanticOutcomeRecorder recorder;
  const std::array seed{std::byte{1}, std::byte{2}};
  ASSERT_TRUE(recorder.record_dispatch(seed).ok());
  ASSERT_TRUE(recorder.record_greedy_trajectory(seed).ok());
  ASSERT_TRUE(recorder.record_accepted_token_ledger(seed).ok());
  auto outcome = QwenSemanticObservationFinalizer::Run(
      plan,
      {endpoint(logits, 1, CudaCopyMemoryType::kDevice),
       endpoint(kv, 2, CudaCopyMemoryType::kDevice)},
      {1, 100, 200, 300, 1000, 41, 43, 47, 0, 2, 4, 5},
      {&allocator, &placement, &copy, &events, &evidence, &clock, &waiter},
      900, 800, std::move(recorder));
  ASSERT_TRUE(outcome.ok()) << outcome.status().message();
  EXPECT_EQ(outcome->device_peak_bytes, 900);
  EXPECT_EQ(outcome->pinned_peak_bytes, 800);
  EXPECT_EQ(copy.calls, plan.slices().size() + 1);
  EXPECT_EQ(allocator.releases, 2);
  EXPECT_NE(outcome->final_logits_root, Sha256Digest{});
  EXPECT_NE(outcome->kv_state_root, Sha256Digest{});
}

TEST(QwenSemanticObservationFinalizerTest,
     RejectsDirtyEvidenceAndReleasesBothPinnedOwners) {
  const auto plan = kv_plan();
  std::vector<std::byte> logits(
      QwenSemanticObservationTransfer::kFinalLogitsBytes, std::byte{3});
  std::vector<std::byte> kv(4 * QwenKvSlotPool::kSlotPayloadBytes,
                            std::byte{7});
  PinnedAllocator allocator;
  Placement placement;
  CopyDriver copy;
  Events events;
  Evidence evidence;
  evidence.value.device_error_code = 9;
  Clock clock;
  Waiter waiter;
  QwenSemanticOutcomeRecorder recorder;
  const std::array seed{std::byte{1}};
  ASSERT_TRUE(recorder.record_dispatch(seed).ok());
  ASSERT_TRUE(recorder.record_greedy_trajectory(seed).ok());
  ASSERT_TRUE(recorder.record_accepted_token_ledger(seed).ok());
  const auto outcome = QwenSemanticObservationFinalizer::Run(
      plan,
      {endpoint(logits, 1, CudaCopyMemoryType::kDevice),
       endpoint(kv, 2, CudaCopyMemoryType::kDevice)},
      {1, 100, 200, 300, 1000, 41, 43, 47, 0, 2, 4, 5},
      {&allocator, &placement, &copy, &events, &evidence, &clock, &waiter},
      900, 800, std::move(recorder));
  EXPECT_FALSE(outcome.ok());
  EXPECT_EQ(allocator.allocations, 2);
  EXPECT_EQ(allocator.releases, 2);
}

}  // namespace
}  // namespace pih
