#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_semantic_observation_transfer.h"

namespace pih {
namespace {

class CopyDriver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 41; }
  Status copy(CudaCopyKind kind, std::uintptr_t destination,
              std::uintptr_t source, std::uint64_t bytes,
              DriverStreamHandle stream) override {
    calls.push_back({kind, destination, source, bytes, stream});
    if (calls.size() == fail_at) return Status::Internal("injected copy failure");
    return Status::Ok();
  }
  struct Call {
    CudaCopyKind kind;
    std::uintptr_t destination;
    std::uintptr_t source;
    std::uint64_t bytes;
    DriverStreamHandle stream;
  };
  std::vector<Call> calls;
  std::size_t fail_at = 0;
};

QwenKvSemanticObservationPlan kv_plan() {
  const std::array<QwenKvBlockHandle, 2> handles{{{2, 11}, {7, 13}}};
  auto table = QwenKvBlockTable::Create(4, 7, 17, handles).value();
  const auto append = table.prepare_append(17).value();
  EXPECT_TRUE(table.commit_append(append).ok());
  std::vector<QwenKvSlotState> states(
      10, {1, QwenKvSlotPool::kNoOwner, 0,
           QwenKvSlotLifecycle::kFreeClean, 0, 0});
  states[2] = {11, 4, 16, QwenKvSlotLifecycle::kOwned, 0, 0};
  states[7] = {13, 4, 1, QwenKvSlotLifecycle::kOwned, 0, 0};
  return QwenKvSemanticObservationPlan::Create(
             table, states, 10 * QwenKvSlotPool::kSlotPayloadBytes)
      .value();
}

CudaCopyEndpoint endpoint(std::uintptr_t base, std::uint64_t bytes,
                          std::uint64_t owner, CudaCopyMemoryType type,
                          std::int32_t locality) {
  return {base, bytes, 0, owner, 1, type, 0, locality};
}

QwenSemanticObservationTransfer transfer() {
  const auto kv = kv_plan();
  return QwenSemanticObservationTransfer::Create(
             endpoint(0x20000000, 1 << 20, 1, CudaCopyMemoryType::kDevice, 0),
             endpoint(0x30000000, 1 << 20, 2,
                      CudaCopyMemoryType::kRegisteredPinnedHost, 1),
             endpoint(0x40000000, 10 * QwenKvSlotPool::kSlotPayloadBytes, 3,
                      CudaCopyMemoryType::kDevice, 0),
             endpoint(0x50000000, kv.payload_bytes(), 4,
                      CudaCopyMemoryType::kRegisteredPinnedHost, 1),
             kv, {100, 41, 43, 47})
      .value();
}

TEST(QwenSemanticObservationTransferTest, BuildsAndSubmitsTypedBatchOnce) {
  auto value = transfer();
  ASSERT_EQ(value.plans().size(), 113);
  EXPECT_EQ(value.plans().front().plan_id(), 100);
  EXPECT_EQ(value.plans().back().plan_id(), 212);
  EXPECT_EQ(value.plans().front().bytes(),
            QwenSemanticObservationTransfer::kFinalLogitsBytes);
  EXPECT_EQ(value.plans()[1].source_address(),
            0x40000000 + 2 * QwenKvSlotPool::kSlotPayloadBytes);
  EXPECT_EQ(value.plans()[1].destination_address(), 0x50000000);
  EXPECT_EQ(value.plans()[2].bytes(),
            QwenKvAddressMapper::kBytesPerToken);

  CopyDriver driver;
  ASSERT_TRUE(value.submit(driver).ok());
  EXPECT_EQ(driver.calls.size(), 113);
  EXPECT_EQ(driver.calls.front().stream, 43);
  EXPECT_EQ(value.state(), QwenSemanticObservationTransferState::kSubmitted);
  EXPECT_FALSE(value.submit(driver).ok());
  EXPECT_EQ(value.state(), QwenSemanticObservationTransferState::kPoisoned);
}

TEST(QwenSemanticObservationTransferTest, CopyFailurePoisonsPartialBatch) {
  auto value = transfer();
  CopyDriver driver;
  driver.fail_at = 3;
  EXPECT_FALSE(value.submit(driver).ok());
  EXPECT_EQ(driver.calls.size(), 3);
  EXPECT_EQ(value.state(), QwenSemanticObservationTransferState::kPoisoned);
}

}  // namespace
}  // namespace pih
