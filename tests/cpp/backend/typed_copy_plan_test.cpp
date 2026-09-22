#include "pih/backend/cuda/typed_copy_plan.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

CudaCopyEndpoint host(std::uintptr_t base, std::uint64_t bytes,
                      std::uint64_t owner = 1) {
  return {base, bytes, 0, owner, 3,
          CudaCopyMemoryType::kRegisteredPinnedHost, 0, 0};
}

CudaCopyEndpoint device(std::uintptr_t base, std::uint64_t bytes,
                        std::uint64_t owner = 2, std::int32_t ordinal = 0) {
  return {base, bytes, 0, owner, 5, CudaCopyMemoryType::kDevice, 0, ordinal};
}

class FakeCopyDriver final : public TypedCopyDriver {
 public:
  [[nodiscard]] std::uintptr_t context_identity() const noexcept override {
    return context;
  }
  Status copy(CudaCopyKind kind, std::uintptr_t destination,
              std::uintptr_t source, std::uint64_t bytes,
              DriverStreamHandle stream) override {
    ++calls;
    last_kind = kind;
    last_destination = destination;
    last_source = source;
    last_bytes = bytes;
    last_stream = stream;
    return result;
  }
  std::uintptr_t context = 17;
  Status result = Status::Ok();
  int calls = 0;
  CudaCopyKind last_kind = CudaCopyKind::kDeviceToDevice;
  std::uintptr_t last_destination = 0;
  std::uintptr_t last_source = 0;
  std::uint64_t last_bytes = 0;
  DriverStreamHandle last_stream = 0;
};

TEST(CudaTypedCopyPlanTest, SubmitsExactPinnedH2DOnce) {
  auto source = host(0x1000, 4096);
  auto destination = device(0x4000, 4096);
  source.offset = 128;
  destination.offset = 256;
  auto plan = CudaTypedCopyPlan::Create(
      7, CudaCopyPurpose::kInput, CudaCopyKind::kHostToDevice, source,
      destination, 1024, 128, 17, 19, 23);
  ASSERT_TRUE(plan.ok()) << plan.status().message();
  EXPECT_EQ(plan->source_address(), 0x1080);
  EXPECT_EQ(plan->destination_address(), 0x4100);
  FakeCopyDriver driver;
  ASSERT_TRUE(plan->submit(driver).ok());
  EXPECT_EQ(driver.calls, 1);
  EXPECT_EQ(driver.last_kind, CudaCopyKind::kHostToDevice);
  EXPECT_EQ(driver.last_bytes, 1024);
  EXPECT_EQ(driver.last_stream, 19);
  EXPECT_FALSE(plan->submit(driver).ok());
  EXPECT_EQ(driver.calls, 1);
}

TEST(CudaTypedCopyPlanTest, ZeroBytesIsManifestNoOp) {
  auto plan = CudaTypedCopyPlan::Create(
      7, CudaCopyPurpose::kInput, CudaCopyKind::kHostToDevice,
      host(0x1000, 64), device(0x2000, 64), 0, 1, 0, 0, 0);
  ASSERT_TRUE(plan.ok());
  EXPECT_TRUE(plan->no_op());
  FakeCopyDriver driver;
  EXPECT_TRUE(plan->submit(driver).ok());
  EXPECT_EQ(driver.calls, 0);
}

TEST(CudaTypedCopyPlanTest, RejectsDirectionPurposeRangeAndAlignmentDrift) {
  EXPECT_FALSE(CudaTypedCopyPlan::Create(
                   7, CudaCopyPurpose::kResult,
                   CudaCopyKind::kHostToDevice, host(0x1000, 64),
                   device(0x2000, 64), 32, 1, 17, 19, 23)
                   .ok());
  EXPECT_FALSE(CudaTypedCopyPlan::Create(
                   7, CudaCopyPurpose::kInput,
                   CudaCopyKind::kHostToDevice, device(0x1000, 64, 1),
                   device(0x2000, 64), 32, 1, 17, 19, 23)
                   .ok());
  auto short_source = host(0x1000, 64);
  short_source.offset = 48;
  EXPECT_FALSE(CudaTypedCopyPlan::Create(
                   7, CudaCopyPurpose::kInput,
                   CudaCopyKind::kHostToDevice, short_source,
                   device(0x2000, 64), 32, 1, 17, 19, 23)
                   .ok());
  EXPECT_FALSE(CudaTypedCopyPlan::Create(
                   7, CudaCopyPurpose::kInput,
                   CudaCopyKind::kHostToDevice, host(0x1001, 64),
                   device(0x2000, 64), 32, 16, 17, 19, 23)
                   .ok());
}

TEST(CudaTypedCopyPlanTest, RejectsOverlappingAndCrossDeviceD2D) {
  EXPECT_FALSE(CudaTypedCopyPlan::Create(
                   7, CudaCopyPurpose::kSameRankMove,
                   CudaCopyKind::kDeviceToDevice,
                   device(0x1000, 1024, 1), device(0x1200, 1024, 2), 768, 1,
                   17, 19, 23)
                   .ok());
  EXPECT_FALSE(CudaTypedCopyPlan::Create(
                   7, CudaCopyPurpose::kSameRankMove,
                   CudaCopyKind::kDeviceToDevice,
                   device(0x1000, 1024, 1, 0),
                   device(0x2000, 1024, 2, 1), 512, 1, 17, 19, 23)
                   .ok());
}

TEST(CudaTypedCopyPlanTest, ContextMismatchFailsWithoutDriverCopy) {
  auto plan = CudaTypedCopyPlan::Create(
                  7, CudaCopyPurpose::kDiagnostic,
                  CudaCopyKind::kDeviceToHost, device(0x2000, 64),
                  host(0x1000, 64), 32, 1, 17, 19, 23)
                  .value();
  FakeCopyDriver driver;
  driver.context = 18;
  EXPECT_FALSE(plan.submit(driver).ok());
  EXPECT_EQ(driver.calls, 0);
  driver.context = 17;
  EXPECT_FALSE(plan.submit(driver).ok());
}

}  // namespace
}  // namespace pih
