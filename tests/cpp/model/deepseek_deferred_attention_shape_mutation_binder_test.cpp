#include "pih/model/deepseek_deferred_attention_shape_mutation_binder.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class FixedOps final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

struct Fixture final {
  DeepSeekFixedStateLayout layout =
      DeepSeekFixedStateLayout::Build(std::vector<std::uint32_t>{2, 3}, false)
          .value();
  FixedOps operations;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x100000, layout.total_bytes()}, {0x200000, layout.total_bytes()}, 8,
      operations).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(4).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(4).value();
};

DeepSeekCompressedLayerUpdateSubmission update(std::uint32_t layer,
                                                std::uint32_t ratio,
                                                std::uint32_t position) {
  DeepSeekCompressedLayerUpdateSubmission value;
  value.ratio = ratio;
  value.main_state = {layer, false, 11, 12, 13, 14, 15, 7, 1, position};
  if (ratio == 4) {
    value.index_state = {layer, true, 21, 22, 23, 24, 15, 7, 1, position};
  }
  return value;
}

DeepSeekAttentionLayerPlanShapeSeed shape(std::uint32_t layer,
                                          std::uint32_t ratio,
                                          std::uint32_t position) {
  DeepSeekAttentionLayerPlanShapeSeed value;
  value.layer = layer;
  value.ratio = ratio;
  value.positions = {position};
  value.updates = {update(layer, ratio, position)};
  value.bind_compressed_page_mutations = true;
  value.main_rms_weight_bf16 = 31;
  value.index_rms_weight_bf16 = ratio == 4 ? 32 : 0;
  value.main_cos_sin_cache_f32 = 41;
  value.index_cos_sin_cache_f32 = ratio == 4 ? 42 : 0;
  value.rms_epsilon = 1.0e-6F;
  return value;
}

TEST(DeepSeekDeferredAttentionShapeMutationBinderTest,
     ResolvesPublishedRatio4TailInsidePreparingTransaction) {
  Fixture fixture;
  auto committed = fixture.ratio4.reserve(9, 2, 0).value();
  ASSERT_TRUE(fixture.ratio4.publish(committed).ok());
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 1, 0, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(7).ok());
  std::vector<DeepSeekAttentionLayerPlanShapeSeed> shapes{
      shape(2, 4, 7)};
  ASSERT_TRUE(DeepSeekDeferredAttentionShapeMutationBinder::Bind(
      9, shapes, transaction, fixture.ratio4, fixture.ratio128).ok());
  EXPECT_TRUE(shapes[0].updates[0].ratio4_slot.tail_cow);
  EXPECT_EQ(shapes[0].updates[0].ratio4_slot.committed.main, committed.main);
}

TEST(DeepSeekDeferredAttentionShapeMutationBinderTest,
     ReservesRatio128AppendWithoutSchedulerPhysicalHandle) {
  Fixture fixture;
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 0, 1, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(7).ok());
  std::vector<DeepSeekAttentionLayerPlanShapeSeed> shapes{
      shape(3, 128, 127)};
  ASSERT_TRUE(DeepSeekDeferredAttentionShapeMutationBinder::Bind(
      9, shapes, transaction, fixture.ratio4, fixture.ratio128).ok());
  EXPECT_TRUE(shapes[0].updates[0].has_completed_slot);
  EXPECT_EQ(shapes[0].updates[0].ratio128_slot.target.logical_page, 0U);
}

TEST(DeepSeekDeferredAttentionShapeMutationBinderTest,
     RejectsMissingPublishedTailBeforeAnyReservation) {
  Fixture fixture;
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      9, 1, 0, fixture.banks, fixture.ratio4, fixture.ratio128).value();
  ASSERT_TRUE(transaction.begin(7).ok());
  std::vector<DeepSeekAttentionLayerPlanShapeSeed> shapes{
      shape(2, 4, 7)};
  EXPECT_FALSE(DeepSeekDeferredAttentionShapeMutationBinder::Bind(
      9, shapes, transaction, fixture.ratio4, fixture.ratio128).ok());
  EXPECT_EQ(fixture.ratio4.reserved_pairs(), 0U);
}

} }  // namespace pih
