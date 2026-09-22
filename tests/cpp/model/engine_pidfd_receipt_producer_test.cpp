#include "pih/model/engine_pidfd_receipt_producer.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

EnginePidfdReapTarget direct_target() {
  return {{EngineSupervisedProcessRole::kRank, 0, 101, 201}, 100, 0};
}
EnginePidfdReapTarget adopted_target() {
  return {{EngineSupervisedProcessRole::kRank, 0, 101, 201}, 90, 100};
}

class Operations final : public EnginePidfdReapOperations {
 public:
  Result<std::optional<EnginePidfdKernelReapObservation>> poll_and_reap(
      const EnginePidfdReapTarget& target) override {
    ++polls;
    if (pending) return std::optional<EnginePidfdKernelReapObservation>{};
    return std::optional<EnginePidfdKernelReapObservation>{{
        target.process.process_identity, target.process.pidfd_identity,
        exited, reaped}};
  }
  bool pending = false;
  bool exited = true;
  bool reaped = true;
  int polls = 0;
};

TEST(EnginePidfdReceiptProducerTest, ProducesDirectParentReceipt) {
  Operations operations;
  const auto target = direct_target();
  auto producer = EnginePidfdReceiptProducer::Create(
      {7, 100, EnginePidfdReaperMode::kDirectParent, false},
      {&target, 1}, operations).value();
  auto result = producer.poll(target, 1);
  ASSERT_TRUE(result.ok());
  ASSERT_TRUE(result->has_value());
  EXPECT_EQ((*result)->engine_generation, 7U);
  EXPECT_EQ((*result)->event_identity, 1U);
  EXPECT_TRUE((*result)->exited);
  EXPECT_TRUE((*result)->reaped);
}

TEST(EnginePidfdReceiptProducerTest, AllowsOnlyExplicitKernelSubreaperAdoption) {
  Operations operations;
  const auto target = adopted_target();
  auto producer = EnginePidfdReceiptProducer::Create(
      {7, 100, EnginePidfdReaperMode::kDesignatedSubreaper, true},
      {&target, 1}, operations).value();
  EXPECT_TRUE(producer.poll(target, 1).ok());
  for (int mutation = 0; mutation < 3; ++mutation) {
    auto target = adopted_target();
    if (mutation == 0) target.adopted_reaper_identity = 0;
    if (mutation == 1) target.adopted_reaper_identity = 99;
    if (mutation == 2) target.creator_process_identity = 100;
    auto value = EnginePidfdReceiptProducer::Create(
        {7, 100, EnginePidfdReaperMode::kDesignatedSubreaper, true},
        {&target, 1}, operations);
    EXPECT_FALSE(value.ok()) << mutation;
  }
  EXPECT_FALSE(EnginePidfdReceiptProducer::Create(
      {7, 100, EnginePidfdReaperMode::kDesignatedSubreaper, false},
      {&target, 1}, operations).ok());
}

TEST(EnginePidfdReceiptProducerTest, PendingDoesNotProduceOrPoison) {
  Operations operations; operations.pending = true;
  const auto target = direct_target();
  auto producer = EnginePidfdReceiptProducer::Create(
      {7, 100, EnginePidfdReaperMode::kDirectParent, false},
      {&target, 1}, operations).value();
  auto result = producer.poll(target, 1);
  ASSERT_TRUE(result.ok());
  EXPECT_FALSE(result->has_value());
  operations.pending = false;
  EXPECT_TRUE(producer.poll(target, 1).ok());
}

TEST(EnginePidfdReceiptProducerTest, KernelIdentityOrReapDriftPoisons) {
  for (int mutation = 0; mutation < 3; ++mutation) {
    Operations operations;
    const auto authorized = direct_target();
    auto producer = EnginePidfdReceiptProducer::Create(
        {7, 100, EnginePidfdReaperMode::kDirectParent, false},
        {&authorized, 1}, operations).value();
    if (mutation == 0) operations.exited = false;
    if (mutation == 1) operations.reaped = false;
    auto target = direct_target();
    if (mutation == 2) ++target.process.pidfd_identity;
    EXPECT_FALSE(producer.poll(target, 1).ok()) << mutation;
    EXPECT_FALSE(producer.poll(authorized, 1).ok()) << mutation;
  }
}

}  // namespace
}  // namespace pih
