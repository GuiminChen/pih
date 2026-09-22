#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_numerical_tap_transfer.h"

namespace pih {
namespace {

class TransferDriver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return 99; }
  Status copy(CudaCopyKind kind, std::uintptr_t, std::uintptr_t,
              std::uint64_t, DriverStreamHandle) override {
    kinds.push_back(kind);
    return result;
  }
  std::vector<CudaCopyKind> kinds;
  Status result = Status::Ok();
};

CudaCompletionFrontier complete(std::uint64_t plan, std::uint64_t event,
                                CudaCompletionPhase phase) {
  auto frontier = CudaCompletionFrontier::Create(
                      {1, 0, plan, phase, plan}, event, 1, 2)
                      .value();
  EXPECT_TRUE(frontier.observe(event, CudaEventQueryResult::kSuccess, true, 0,
                               false)
                  .ok());
  return frontier;
}

CudaTypedCopyPlan snapshot_plan(std::uint64_t destination_owner = 200) {
  return CudaTypedCopyPlan::Create(
             91, CudaCopyPurpose::kSameRankMove,
             CudaCopyKind::kDeviceToDevice,
             {0x1000, 4096, 0, 100, 11, CudaCopyMemoryType::kDevice, 0, 0},
             {0x3000, 4096, 256, destination_owner, 41,
              CudaCopyMemoryType::kDevice, 0, 0},
             16, 8, 99, 123, 501)
      .value();
}

CudaTypedCopyPlan host_plan(std::uint64_t source_owner = 200) {
  return CudaTypedCopyPlan::Create(
             92, CudaCopyPurpose::kDiagnostic,
             CudaCopyKind::kDeviceToHost,
             {0x3000, 4096, 256, source_owner, 41,
              CudaCopyMemoryType::kDevice, 0, 0},
             {0x5000, 4096, 256, 300, 73,
              CudaCopyMemoryType::kRegisteredPinnedHost, 0, 2},
             16, 8, 99, 124, 502)
      .value();
}

TEST(QwenNumericalTapTransferTest, ExecutesD2dThenD2hAndSealsBytes) {
  auto transfer = QwenNumericalTapTransfer::Create(
      3, 7, 16, 90, snapshot_plan(), host_plan());
  ASSERT_TRUE(transfer.ok()) << transfer.status().message();
  ASSERT_TRUE(transfer->complete_producer(
                           complete(90, 500, CudaCompletionPhase::kAttention))
                  .ok());
  TransferDriver driver;
  ASSERT_TRUE(transfer->submit_snapshot(driver).ok());
  ASSERT_EQ(driver.kinds,
            std::vector<CudaCopyKind>{CudaCopyKind::kDeviceToDevice});
  ASSERT_TRUE(transfer->complete_snapshot(
                           complete(91, 501, CudaCompletionPhase::kCopy))
                  .ok());
  ASSERT_TRUE(transfer->submit_host(driver).ok());
  ASSERT_EQ(driver.kinds,
            (std::vector<CudaCopyKind>{CudaCopyKind::kDeviceToDevice,
                                       CudaCopyKind::kDeviceToHost}));
  ASSERT_TRUE(transfer->complete_host(
                           complete(92, 502, CudaCompletionPhase::kCopy))
                  .ok());
  const std::array<std::byte, 16> bytes{std::byte{1}};
  auto receipt = transfer->seal(bytes);
  ASSERT_TRUE(receipt.ok());
  EXPECT_EQ(receipt->capture_index, 3);
  EXPECT_EQ(receipt->capture_generation, 7);
  EXPECT_EQ(transfer->state(), QwenNumericalTapTransferState::kSealed);
}

TEST(QwenNumericalTapTransferTest, RejectsEndpointChainDriftAtCreation) {
  EXPECT_FALSE(QwenNumericalTapTransfer::Create(
                   3, 7, 16, 90, snapshot_plan(), host_plan(201))
                   .ok());
  EXPECT_FALSE(QwenNumericalTapTransfer::Create(
                   3, 7, 17, 90, snapshot_plan(), host_plan())
                   .ok());
}

TEST(QwenNumericalTapTransferTest, OutOfOrderOrWrongFrontierPoisonsTransfer) {
  auto transfer = QwenNumericalTapTransfer::Create(
                      3, 7, 16, 90, snapshot_plan(), host_plan())
                      .value();
  TransferDriver driver;
  EXPECT_FALSE(transfer.submit_snapshot(driver).ok());
  EXPECT_EQ(transfer.state(), QwenNumericalTapTransferState::kPoisoned);

  transfer = QwenNumericalTapTransfer::Create(
                 3, 7, 16, 90, snapshot_plan(), host_plan())
                 .value();
  EXPECT_FALSE(transfer.complete_producer(
                           complete(89, 500, CudaCompletionPhase::kAttention))
                   .ok());
  EXPECT_EQ(transfer.state(), QwenNumericalTapTransferState::kPoisoned);
}

TEST(QwenNumericalTapTransferTest, TruncatedWriterInvalidatesTransfer) {
  auto transfer = QwenNumericalTapTransfer::Create(
                      3, 7, 16, 90, snapshot_plan(), host_plan())
                      .value();
  ASSERT_TRUE(transfer.complete_producer(
                          complete(90, 500, CudaCompletionPhase::kAttention))
                  .ok());
  TransferDriver driver;
  ASSERT_TRUE(transfer.submit_snapshot(driver).ok());
  ASSERT_TRUE(transfer.complete_snapshot(
                          complete(91, 501, CudaCompletionPhase::kCopy))
                  .ok());
  ASSERT_TRUE(transfer.submit_host(driver).ok());
  ASSERT_TRUE(transfer.complete_host(
                          complete(92, 502, CudaCompletionPhase::kCopy))
                  .ok());
  const std::array<std::byte, 15> bytes{};
  EXPECT_FALSE(transfer.seal(bytes).ok());
  EXPECT_EQ(transfer.state(), QwenNumericalTapTransferState::kPoisoned);
}

TEST(QwenNumericalTapTransferTest,
     InlineDriverSubmitsOnlyExactCaptureAtProducerFrontier) {
  std::vector<QwenNumericalTapTransfer> transfers;
  transfers.push_back(QwenNumericalTapTransfer::Create(
                          0, 7, 16, 90, snapshot_plan(), host_plan())
                          .value());
  TransferDriver copy_driver;
  auto inline_driver = QwenNumericalTapInlineSnapshotDriver::Create(
      transfers, copy_driver, 90);
  ASSERT_TRUE(inline_driver.ok()) << inline_driver.status().message();
  const QwenBf16TapBinding binding{
      0,
      {QwenNumericalTapPoint::kQueryAfterRope, 0, 0, 1},
      7,
      {QwenBf16ExecutionOp::kRope, 0},
      0,
      QwenBf16ActivationSlot::kQuery,
      QwenBf16TapKvComponent::kNotApplicable};

  ASSERT_TRUE(inline_driver->snapshot(binding, 123).ok());
  EXPECT_EQ(copy_driver.kinds,
            std::vector<CudaCopyKind>{CudaCopyKind::kDeviceToDevice});
  EXPECT_EQ(transfers[0].state(),
            QwenNumericalTapTransferState::kSnapshotInFlight);
  EXPECT_FALSE(inline_driver->snapshot(binding, 123).ok());
  EXPECT_EQ(transfers[0].state(), QwenNumericalTapTransferState::kPoisoned);
}

TEST(QwenNumericalTapTransferTest, InlineDriverRejectsCrossStreamSnapshot) {
  std::vector<QwenNumericalTapTransfer> transfers;
  transfers.push_back(QwenNumericalTapTransfer::Create(
                          0, 7, 16, 90, snapshot_plan(), host_plan())
                          .value());
  TransferDriver copy_driver;
  auto inline_driver = QwenNumericalTapInlineSnapshotDriver::Create(
                           transfers, copy_driver, 90)
                           .value();
  const QwenBf16TapBinding binding{
      0,
      {QwenNumericalTapPoint::kQueryAfterRope, 0, 0, 1},
      7,
      {QwenBf16ExecutionOp::kRope, 0},
      0,
      QwenBf16ActivationSlot::kQuery,
      QwenBf16TapKvComponent::kNotApplicable};

  EXPECT_FALSE(inline_driver.snapshot(binding, 124).ok());
  EXPECT_TRUE(copy_driver.kinds.empty());
  EXPECT_EQ(transfers[0].state(),
            QwenNumericalTapTransferState::kProducerPending);
}

TEST(QwenNumericalTapTransferTest,
     InlineDriverRejectsIncompleteOrCrossGenerationTransferSet) {
  std::vector<QwenNumericalTapTransfer> wrong_index;
  wrong_index.push_back(QwenNumericalTapTransfer::Create(
                            1, 7, 16, 90, snapshot_plan(), host_plan())
                            .value());
  TransferDriver copy_driver;
  EXPECT_FALSE(QwenNumericalTapInlineSnapshotDriver::Create(
                   wrong_index, copy_driver, 90)
                   .ok());

  std::vector<QwenNumericalTapTransfer> wrong_generation;
  wrong_generation.push_back(QwenNumericalTapTransfer::Create(
                                 0, 7, 16, 91, snapshot_plan(), host_plan())
                                 .value());
  EXPECT_FALSE(QwenNumericalTapInlineSnapshotDriver::Create(
                   wrong_generation, copy_driver, 90)
                   .ok());
}

}  // namespace
}  // namespace pih
