#include "pih/model/deepseek_bound_dspark_mtp_stage_work_provider.h"

#include <gtest/gtest.h>

#include <array>

namespace pih {
namespace {

class FixedOperations final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override {
    return Status::Ok();
  }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

template <typename T>
T* fake(std::uintptr_t address) {
  return reinterpret_cast<T*>(address);
}

struct Fixture final {
  Fixture()
      : layout(DeepSeekFixedStateLayout::Build({}, true).value()),
        banks(DeepSeekFixedStateBanks::Create(
                  {0x100000U, layout.total_bytes()},
                  {0x200000U, layout.total_bytes()}, 77, operations)
                  .value()),
        transaction(DeepSeekAttentionSequenceTransaction::Create(
                        1, 0, 0, banks, ratio4, ratio128)
                        .value()) {
    for (std::size_t index = 0; index < work.size(); ++index) {
      const auto stage = static_cast<DeepSeekDsparkStageId>(index);
      work[index].attention =
          fake<DeepSeekDsparkMtpStageOperation>(0xA000U + index * 0x100U);
      work[index].prefill_resources = {
          stage, {0x101U + index, 0x201U + index, 0x301U + index, 29},
          &layout, &transaction, 0, {}, 0x400U, 0x500U, 0x600U,
          0x700U, 0x800U, 0x900U, 0xA00U, 11, 12, 5, 4096, 29};
    }
  }

  FixedOperations operations;
  DeepSeekFixedStateLayout layout;
  DeepSeekFixedStateBanks banks;
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction;
  std::array<DeepSeekBoundDsparkMtpStageWork, 3> work;
};

constexpr DeepSeekPipelinePlanDescriptor plan() {
  return {9, 17, DeepSeekPlanPhase::kPrefill, 5, 1};
}

TEST(DeepSeekBoundDsparkMtpStageWorkProviderTest,
     ResolvesTentativeStageStateOnlyAfterTransactionBegin) {
  Fixture fixture;
  auto provider =
      DeepSeekBoundDsparkMtpStageWorkProvider::Create(true).value();
  ASSERT_TRUE(provider.bind(plan(), fixture.work).ok());
  EXPECT_EQ(provider.resolve(DeepSeekDsparkStageId::kMtp0, plan())
                .status().code(),
            StatusCode::kFailedPrecondition);

  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  const auto bank = fixture.transaction.tentative_fixed_state().value();
  for (std::size_t index = 0; index < fixture.work.size(); ++index) {
    const auto stage = static_cast<DeepSeekDsparkStageId>(index);
    auto resolved = provider.resolve(stage, plan());
    ASSERT_TRUE(resolved.ok()) << resolved.status().message();
    ASSERT_NE(*resolved, nullptr);
    const auto expected = fixture.layout.ResolveDspark(stage, bank).value();
    EXPECT_EQ((*resolved)->prefill_resources.prepare_epoch,
              fixture.transaction.prepare_epoch());
    EXPECT_EQ((*resolved)->prefill_resources.recent_state.recent_bf16.address,
              expected.recent_bf16.address);
    EXPECT_EQ(fixture.work[index].prefill_resources.prepare_epoch, 0U);
    EXPECT_EQ(fixture.work[index].prefill_resources.recent_state
                  .recent_bf16.address,
              0U);
  }
}

TEST(DeepSeekBoundDsparkMtpStageWorkProviderTest,
     RejectsPartialCrossStageAndMaterializedTemplates) {
  Fixture fixture;
  auto provider =
      DeepSeekBoundDsparkMtpStageWorkProvider::Create(true).value();
  EXPECT_EQ(provider.bind(plan(), std::span(fixture.work).first(2))
                .code(),
            StatusCode::kInvalidArgument);
  fixture.work[1].prefill_resources.stage = DeepSeekDsparkStageId::kMtp0;
  EXPECT_EQ(provider.bind(plan(), fixture.work).code(),
            StatusCode::kInvalidArgument);
  fixture.work[1].prefill_resources.stage = DeepSeekDsparkStageId::kMtp1;
  fixture.work[1].prefill_resources.prepare_epoch = 1;
  EXPECT_EQ(provider.bind(plan(), fixture.work).code(),
            StatusCode::kInvalidArgument);
  EXPECT_FALSE(DeepSeekBoundDsparkMtpStageWorkProvider::Create(false).ok());
}

TEST(DeepSeekBoundDsparkMtpStageWorkProviderTest,
     RejectsForeignDescriptorAndClearedBinding) {
  Fixture fixture;
  auto provider =
      DeepSeekBoundDsparkMtpStageWorkProvider::Create(true).value();
  ASSERT_TRUE(provider.bind(plan(), fixture.work).ok());
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  auto foreign = plan();
  ++foreign.plan_sequence;
  EXPECT_EQ(provider.resolve(DeepSeekDsparkStageId::kMtp0, foreign)
                .status().code(),
            StatusCode::kFailedPrecondition);
  provider.clear();
  EXPECT_EQ(provider.resolve(DeepSeekDsparkStageId::kMtp0, plan())
                .status().code(),
            StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace pih
