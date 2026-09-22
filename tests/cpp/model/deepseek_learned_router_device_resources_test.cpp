#include "pih/model/deepseek_learned_router_device_resources.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class RouterDeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("router device");
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, 1).value()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

TEST(DeepSeekLearnedRouterDeviceResourcesTest,
     ReusesExpertErrorFlagAndOwnsFullFp32Scores) {
  RouterDeviceAllocator allocator;
  DeepSeekExpertComputeArena arena;
  arena.error_flag_u32 = {0x30000, 256};
  auto resources = DeepSeekLearnedRouterDeviceResources::Allocate(
      8, arena, allocator, 17, 1);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_EQ(resources->view().error_flag_u32, 0x30000U);
  EXPECT_NE(resources->view().scores_f32, 0U);
  EXPECT_EQ(resources->scores_bytes(), 8U * 256U * sizeof(float));
}

TEST(DeepSeekLearnedRouterDeviceResourcesTest,
     RejectsMissingSharedErrorFlag) {
  RouterDeviceAllocator allocator;
  DeepSeekExpertComputeArena arena;
  arena.error_flag_u32 = {0, 256};
  EXPECT_FALSE(DeepSeekLearnedRouterDeviceResources::Allocate(
      1, arena, allocator, 17, 1).ok());
}

}  // namespace
}  // namespace pih
