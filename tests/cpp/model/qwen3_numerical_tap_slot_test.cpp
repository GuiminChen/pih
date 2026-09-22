#include <array>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include "pih/model/qwen3_numerical_tap_slot.h"

namespace pih {
namespace {

class TapCopyDriver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 99; }
  Status copy(CudaCopyKind, std::uintptr_t, std::uintptr_t,
              std::uint64_t, DriverStreamHandle) override {
    ++copies;
    return result;
  }
  Status result = Status::Ok();
  std::size_t copies = 0;
};

CudaCompletionFrontier completed_frontier(std::uint64_t plan,
                                            std::uint64_t event) {
  auto frontier = CudaCompletionFrontier::Create(
                      {7, 0, plan, CudaCompletionPhase::kCopy, plan}, event,
                      10, 20)
                      .value();
  EXPECT_TRUE(frontier.observe(event, CudaEventQueryResult::kSuccess, true, 0,
                               false)
                  .ok());
  return frontier;
}

CudaTypedCopyPlan copy_plan(std::uint64_t plan, std::uint64_t generation,
                            std::uint64_t source_generation = 11) {
  return CudaTypedCopyPlan::Create(
             plan, CudaCopyPurpose::kDiagnostic,
             CudaCopyKind::kDeviceToHost,
             {0x1000, 4096, 0, 41, source_generation,
              CudaCopyMemoryType::kDevice, 0, 0},
             {0x3000, 4096, 0, 77, 101,
              CudaCopyMemoryType::kRegisteredPinnedHost, 0, 0},
             16, 8, 99, 123, generation)
      .value();
}

TEST(QwenNumericalTapSlotTest, SealsOnlyAfterExactProducerAndCopyFrontiers) {
  auto slot = QwenNumericalTapSlot::Create(77, 101, 4096);
  ASSERT_TRUE(slot.ok());
  ASSERT_TRUE(slot->begin(5, 3, 41, 11, 16, 90).ok());
  auto producer = completed_frontier(90, 4);
  ASSERT_TRUE(slot->complete_producer(producer).ok());

  auto copy = copy_plan(91, 5);
  TapCopyDriver driver;
  ASSERT_TRUE(slot->submit_copy(copy, driver).ok());
  EXPECT_EQ(driver.copies, 1);
  auto copied = completed_frontier(91, 5);
  ASSERT_TRUE(slot->complete_copy(copied).ok());

  const std::array<std::byte, 16> observed{std::byte{1}, std::byte{2}};
  auto receipt = slot->seal(5, observed);
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(receipt->capture_index, 3);
  EXPECT_EQ(receipt->capture_generation, 5);
  EXPECT_EQ(receipt->bytes, 16);
  EXPECT_FALSE(receipt->digest.hex().empty());
  EXPECT_TRUE(slot->release(5).ok());
  EXPECT_EQ(slot->state(), QwenNumericalTapSlotState::kFree);
}

TEST(QwenNumericalTapSlotTest, GenerationOrFrontierDriftPoisonsSlot) {
  auto slot = QwenNumericalTapSlot::Create(77, 101, 4096).value();
  ASSERT_TRUE(slot.begin(5, 3, 41, 11, 16, 90).ok());
  auto wrong = completed_frontier(89, 4);
  EXPECT_FALSE(slot.complete_producer(wrong).ok());
  EXPECT_EQ(slot.state(), QwenNumericalTapSlotState::kPoisoned);
  EXPECT_FALSE(slot.release(5).ok());
}

TEST(QwenNumericalTapSlotTest, RejectsCopyEndpointDriftAndPrematureReuse) {
  auto slot = QwenNumericalTapSlot::Create(77, 101, 4096).value();
  ASSERT_TRUE(slot.begin(5, 3, 41, 11, 16, 90).ok());
  EXPECT_FALSE(slot.begin(6, 4, 41, 12, 16, 92).ok());
  ASSERT_TRUE(slot.complete_producer(completed_frontier(90, 4)).ok());
  auto wrong_source = copy_plan(91, 5, 12);
  TapCopyDriver driver;
  EXPECT_FALSE(slot.submit_copy(wrong_source, driver).ok());
  EXPECT_EQ(slot.state(), QwenNumericalTapSlotState::kPoisoned);
  EXPECT_EQ(driver.copies, 0);
}

TEST(QwenNumericalTapSlotTest, WriterCannotSealDroppedOrWrongSizedBytes) {
  auto slot = QwenNumericalTapSlot::Create(77, 101, 4096).value();
  ASSERT_TRUE(slot.begin(5, 3, 41, 11, 16, 90).ok());
  ASSERT_TRUE(slot.complete_producer(completed_frontier(90, 4)).ok());
  auto copy = copy_plan(91, 5);
  TapCopyDriver driver;
  ASSERT_TRUE(slot.submit_copy(copy, driver).ok());
  ASSERT_TRUE(slot.complete_copy(completed_frontier(91, 5)).ok());
  const std::array<std::byte, 15> truncated{};
  EXPECT_FALSE(slot.seal(5, truncated).ok());
  EXPECT_EQ(slot.state(), QwenNumericalTapSlotState::kPoisoned);
}

}  // namespace
}  // namespace pih
