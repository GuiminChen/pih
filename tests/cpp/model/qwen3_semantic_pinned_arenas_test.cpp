#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_semantic_pinned_arenas.h"

namespace pih {
namespace {

class PinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++allocations;
    if (allocations == fail_at) {
      return Status::ResourceExhausted("injected pinned failure");
    }
    void* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t(alignment), std::nothrow);
    if (data == nullptr) return Status::ResourceExhausted("test allocation");
    return Allocation{data, bytes, alignment,
                      static_cast<std::uint64_t>(70 + allocations),
                      Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ++releases;
    ::operator delete(allocation.data,
                      std::align_val_t(allocation.alignment));
  }
  int allocations = 0;
  int releases = 0;
  int fail_at = 99;
};

class Placement final : public PinnedPlacementVerifier {
 public:
  Status verify(const void* data, std::uint64_t bytes,
                std::int32_t numa_node) override {
    ++calls;
    if (data == nullptr || bytes == 0 || numa_node != 2) {
      return Status::FailedPrecondition("invalid placement request");
    }
    if (calls == fail_at) {
      return Status::FailedPrecondition("injected placement failure");
    }
    return Status::Ok();
  }
  int calls = 0;
  int fail_at = 99;
};

QwenKvSemanticObservationPlan kv_plan() {
  const std::array<QwenKvBlockHandle, 1> handles{{{2, 11}}};
  auto table = QwenKvBlockTable::Create(4, 7, 1, handles).value();
  const auto append = table.prepare_append(1).value();
  EXPECT_TRUE(table.commit_append(append).ok());
  std::vector<QwenKvSlotState> states(
      4, {1, QwenKvSlotPool::kNoOwner, 0,
          QwenKvSlotLifecycle::kFreeClean, 0, 0});
  states[2] = {11, 4, 1, QwenKvSlotLifecycle::kOwned, 0, 0};
  return QwenKvSemanticObservationPlan::Create(
             table, states, 4 * QwenKvSlotPool::kSlotPayloadBytes)
      .value();
}

TEST(QwenSemanticPinnedArenasTest, OwnsVerifiedTypedRanges) {
  const auto plan = kv_plan();
  PinnedAllocator allocator;
  Placement placement;
  {
    auto arenas = QwenSemanticPinnedArenas::AllocateVerified(
        plan, allocator, placement, 0, 2);
    ASSERT_TRUE(arenas.ok()) << arenas.status().message();
    EXPECT_EQ(arenas->logits().size(),
              QwenSemanticObservationTransfer::kFinalLogitsBytes);
    EXPECT_EQ(arenas->kv().size(), plan.payload_bytes());
    EXPECT_EQ(arenas->total_bytes(),
              QwenSemanticObservationTransfer::kFinalLogitsBytes +
                  plan.payload_bytes());
    auto logits = arenas->logits_endpoint(101, 0).value();
    auto kv = arenas->kv_endpoint(102, 0).value();
    EXPECT_EQ(logits.generation, 71);
    EXPECT_EQ(kv.generation, 72);
    EXPECT_EQ(logits.device_or_numa, 2);
    EXPECT_EQ(kv.memory_type,
              CudaCopyMemoryType::kRegisteredPinnedHost);
    EXPECT_FALSE(arenas->kv_endpoint(0, 0).ok());
    EXPECT_FALSE(arenas->logits_endpoint(1, 1).ok());
  }
  EXPECT_EQ(allocator.releases, 2);
  EXPECT_EQ(placement.calls, 2);
}

TEST(QwenSemanticPinnedArenasTest, AllocationAndPlacementFailureRollBack) {
  const auto plan = kv_plan();
  PinnedAllocator allocation_failed;
  allocation_failed.fail_at = 2;
  Placement placement;
  EXPECT_FALSE(QwenSemanticPinnedArenas::AllocateVerified(
                   plan, allocation_failed, placement, 0, 2)
                   .ok());
  EXPECT_EQ(allocation_failed.releases, 1);

  PinnedAllocator placement_failed;
  Placement bad_placement;
  bad_placement.fail_at = 2;
  EXPECT_FALSE(QwenSemanticPinnedArenas::AllocateVerified(
                   plan, placement_failed, bad_placement, 0, 2)
                   .ok());
  EXPECT_EQ(placement_failed.releases, 2);
}

}  // namespace
}  // namespace pih
