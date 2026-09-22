#include "pih/model/deepseek_rope_table_device_resources.h"

#include <gtest/gtest.h>
#include <cstdlib>
#if !defined(_WIN32)
static void* _aligned_malloc(std::size_t bytes, std::size_t alignment) {
  void* result = nullptr;
  return posix_memalign(&result, alignment, bytes) == 0 ? result : nullptr;
}
static void _aligned_free(void* pointer) { std::free(pointer); }
#endif

#include <vector>

namespace pih { namespace {

class RopeAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    bytes_ = bytes;
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    return Allocation{data, bytes, alignment, 7,
                      Device::Create(DeviceType::kCuda, 2).value()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }
  std::uint64_t bytes_ = 0;
};

class RopeOperations final : public DeepSeekRopeTableOperations {
 public:
  bool completion_proven = false;
  std::vector<std::uintptr_t> synchronized;
  Status synchronize(std::uintptr_t stream) override {
    synchronized.push_back(stream);
    return completion_proven ? Status::Ok() : Status::Unavailable("test completion pending");
  }
  Status initialize(DeepSeekRopeTableLaunch launch) override {
    launches.push_back(launch);
    return validate_deepseek_rope_table_launch(launch);
  }
  std::vector<DeepSeekRopeTableLaunch> launches;
};

TEST(DeepSeekRopeTableDeviceResourcesTest,
     OwnsAndInitializesOnlyTablesRequiredByStage) {
  RopeAllocator allocator;
  RopeOperations operations;
  operations.completion_proven = true;
  DeepSeekStagePlan first{0, {0, 10}, true, false, false};
  auto resources = DeepSeekRopeTableDeviceResources::AllocateDeferred(
      allocator, first, 8, 0x55, operations, 91, 2);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_TRUE(operations.launches.empty());
  ASSERT_TRUE(resources->ensure_initialized().ok());
  ASSERT_EQ(operations.synchronized.size(), 2U);
  ASSERT_EQ(operations.launches.size(), 2U);
  EXPECT_FALSE(operations.launches[0].yarn);
  EXPECT_TRUE(operations.launches[1].yarn);
  EXPECT_EQ(operations.launches[0].scaling_factor, 1.0);
  EXPECT_EQ(operations.launches[1].scaling_factor, 16.0);
  EXPECT_NE(resources->view().base_f32, 0U);
  EXPECT_NE(resources->view().yarn_f32, 0U);
  EXPECT_NE(resources->view().base_f32, resources->view().yarn_f32);
  EXPECT_EQ(resources->view().error_flag_u32 % 256U, 0U);
  EXPECT_EQ(resources->backing_bytes(), allocator.bytes_);

  RopeOperations middle_operations;
  middle_operations.completion_proven = true;
  DeepSeekStagePlan middle{1, {11, 20}, false, false, false};
  auto middle_resources =
      DeepSeekRopeTableDeviceResources::AllocateDeferred(
          allocator, middle, 8, 0x55, middle_operations, 91, 2);
  ASSERT_TRUE(middle_resources.ok());
  ASSERT_TRUE(middle_resources->ensure_initialized().ok());
  ASSERT_EQ(middle_operations.launches.size(), 1U);
  EXPECT_TRUE(middle_operations.launches[0].yarn);
  EXPECT_EQ(middle_resources->view().base_f32, 0U);
  EXPECT_NE(middle_resources->view().yarn_f32, 0U);
}

TEST(DeepSeekRopeTableDeviceResourcesTest,
     FinalDsparkStageOwnsBaseAndYarnTables) {
  RopeAllocator allocator;
  RopeOperations operations;
  operations.completion_proven = true;
  DeepSeekStagePlan last{3, {33, 42}, false, true, true};
  auto resources = DeepSeekRopeTableDeviceResources::AllocateDeferred(
      allocator, last, 104, 0x55, operations, 91, 2);
  ASSERT_TRUE(resources.ok());
  EXPECT_TRUE(operations.launches.empty());
  ASSERT_TRUE(resources->ensure_initialized().ok());
  ASSERT_EQ(operations.synchronized.size(), 2U);
  ASSERT_EQ(operations.launches.size(), 2U);
  EXPECT_FALSE(operations.launches[0].yarn);
  EXPECT_TRUE(operations.launches[1].yarn);
}

TEST(DeepSeekRopeTableDeviceResourcesTest, RejectsInvalidProfileInputs) {
  RopeAllocator allocator;
  RopeOperations operations;
  operations.completion_proven = true;
  DeepSeekStagePlan stage{0, {0, 10}, true, false, false};
  EXPECT_FALSE(DeepSeekRopeTableDeviceResources::AllocateDeferred(
      allocator, stage, 0, 0x55, operations, 91, 2).ok());
  EXPECT_FALSE(DeepSeekRopeTableDeviceResources::AllocateDeferred(
      allocator, stage, 8, 0, operations, 91, 2).ok());
  auto invalid = DeepSeekRopeTableLaunch{
      1, 2, 3, 8, 64, 10000.0, 16.0, 65536, 32, 1, false};
  EXPECT_FALSE(validate_deepseek_rope_table_launch(invalid).ok());
}

} }  // namespace pih::<anonymous>
