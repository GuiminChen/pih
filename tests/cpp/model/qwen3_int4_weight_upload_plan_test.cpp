#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_int4_weight_upload_plan.h"

namespace pih {
namespace {

CudaCopyEndpoint host(std::uintptr_t base) {
  return {base,QwenInt4ArtifactLayout::kOfficialFileBytes,0,7,11,
          CudaCopyMemoryType::kRegisteredPinnedHost,3,0};
}
CudaCopyEndpoint device(std::uintptr_t base) {
  return {base,QwenInt4ArtifactLayout::kOfficialLogicalPayloadBytes,0,8,13,
          CudaCopyMemoryType::kDevice,3,0};
}

class UploadDriver final : public TypedCopyDriver {
 public:
  std::uintptr_t context_identity() const noexcept override { return context; }
  Status copy(CudaCopyKind kind,std::uintptr_t destination,
              std::uintptr_t source,std::uint64_t bytes,
              DriverStreamHandle stream) override {
    ++calls; kinds.push_back(kind); destinations.push_back(destination);
    sources.push_back(source); sizes.push_back(bytes); streams.push_back(stream);
    if (calls==fail_on) return Status::Unavailable("injected upload failure");
    return Status::Ok();
  }
  std::uintptr_t context=0x55; int calls=0; int fail_on=0;
  std::vector<CudaCopyKind> kinds; std::vector<std::uintptr_t> destinations;
  std::vector<std::uintptr_t> sources; std::vector<std::uint64_t> sizes;
  std::vector<DriverStreamHandle> streams;
};

TEST(QwenInt4WeightUploadPlanTest, CompactsAllCanonicalPayloadExtents) {
  const auto layout=QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  auto upload=QwenInt4WeightUploadPlan::Create(
      layout,host(0x100000000ULL),device(0x200000000ULL),0x55,19,23,101);
  ASSERT_TRUE(upload.ok()) << upload.status().message();
  EXPECT_EQ(upload->copy_count(),506U);
  EXPECT_EQ(upload->logical_bytes(),538'378'240U);
  EXPECT_EQ(upload->copy(0).source_address(),
            0x100000000ULL+QwenInt4ArtifactLayout::kMetadataBytes);
  EXPECT_EQ(upload->copy(0).destination_address(),0x200000000ULL);
  EXPECT_EQ(upload->copy(0).bytes(),311'164'928U);
  EXPECT_EQ(upload->copy(505).destination_address()+upload->copy(505).bytes(),
            0x200000000ULL+538'378'240ULL);
}

TEST(QwenInt4WeightUploadPlanTest, SubmitsOnceAndPoisonsPartialFailure) {
  const auto layout=QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  auto upload=QwenInt4WeightUploadPlan::Create(
      layout,host(0x100000000ULL),device(0x200000000ULL),0x55,19,23,101)
      .value();
  UploadDriver driver; driver.fail_on=2;
  EXPECT_FALSE(upload.submit(driver).ok());
  EXPECT_EQ(upload.state(),QwenInt4WeightUploadState::kPoisoned);
  EXPECT_EQ(upload.submitted_copies(),1U); EXPECT_EQ(driver.calls,2);
  EXPECT_FALSE(upload.submit(driver).ok()); EXPECT_EQ(driver.calls,2);
}

TEST(QwenInt4WeightUploadPlanTest, RejectsOwnerRangeAndContextDrift) {
  const auto layout=QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  auto short_destination=device(0x200000000ULL);
  --short_destination.allocation_bytes;
  EXPECT_FALSE(QwenInt4WeightUploadPlan::Create(
      layout,host(0x100000000ULL),short_destination,0x55,19,23,101).ok());
  auto same_owner=device(0x200000000ULL); same_owner.owner_id=7;
  EXPECT_FALSE(QwenInt4WeightUploadPlan::Create(
      layout,host(0x100000000ULL),same_owner,0x55,19,23,101).ok());
  EXPECT_FALSE(QwenInt4WeightUploadPlan::Create(
      layout,host(0x100000000ULL),device(0x200000000ULL),0,19,23,101).ok());
}

}  // namespace
}  // namespace pih
