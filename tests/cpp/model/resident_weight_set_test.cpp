#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/resident_weight_set.h"

namespace pih {
namespace {

class FailingAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes, std::uint64_t alignment) override {
    ++calls;
    if (calls == fail_on_call) return Status::ResourceExhausted("injected failure");
    void* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t(alignment), std::nothrow);
    if (data == nullptr) return Status::ResourceExhausted("test allocation failed");
    return Allocation{data, bytes, alignment, static_cast<std::uint64_t>(calls),
                      Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ++releases;
    ::operator delete(allocation.data, std::align_val_t(allocation.alignment));
  }
  int calls = 0;
  int releases = 0;
  int fail_on_call = 0;
};

class FailingCopier final : public MemoryCopier {
 public:
  Status copy(void* destination, Device, const void* source, Device,
              std::uint64_t bytes) override {
    ++calls;
    if (calls == fail_on_call) return Status::Unavailable("injected copy failure");
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
    return Status::Ok();
  }
  int calls = 0;
  int fail_on_call = 0;
};

class ResidentWeightSetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = std::filesystem::temp_directory_path() /
            ("pih-weights-" + std::to_string(counter_++) + ".bin");
    const std::string json =
        R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[2],"data_offsets":[4,6]}})";
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    const auto size = static_cast<std::uint64_t>(json.size());
    for (int index = 0; index < 8; ++index) output.put((size >> (index * 8)) & 0xff);
    output << json << "ABCDEF";
  }
  void TearDown() override { std::filesystem::remove(path_); }

  std::filesystem::path path_;
  static inline std::uint64_t counter_ = 1;
};

std::vector<WeightRequirement> requirements() {
  return {{"a", DType::kUInt8, {4}}, {"b", DType::kUInt8, {2}}};
}

TEST_F(ResidentWeightSetTest, CopiesAllWeightsAndPublishesImmutableCatalog) {
  auto source = SafetensorsFile::Open(path_, 1024);
  ASSERT_TRUE(source.ok());
  CpuAllocator allocator;
  CpuMemoryCopier copier;
  const auto required = requirements();
  auto loaded = ResidentWeightSet::Load(source.value(), required, allocator, copier, 64);
  ASSERT_TRUE(loaded.ok()) << loaded.status().message();
  EXPECT_EQ(loaded->size(), 2);
  EXPECT_EQ(loaded->total_bytes(), 6);
  auto a = loaded->tensor("a");
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(loaded->generation("a").ok());
  EXPECT_EQ(a->generation(), loaded->generation("a").value());
  EXPECT_EQ(static_cast<const char*>(a->data())[2], 'C');
  EXPECT_FALSE(loaded->tensor("missing").ok());
}

TEST_F(ResidentWeightSetTest, AllocationFailureRollsBackWithoutCatalog) {
  auto source = SafetensorsFile::Open(path_, 1024);
  ASSERT_TRUE(source.ok());
  FailingAllocator allocator;
  allocator.fail_on_call = 2;
  CpuMemoryCopier copier;
  const auto required = requirements();
  auto loaded = ResidentWeightSet::Load(source.value(), required, allocator, copier, 64);
  EXPECT_FALSE(loaded.ok());
  EXPECT_EQ(allocator.calls, 2);
  EXPECT_EQ(allocator.releases, 1);
}

TEST_F(ResidentWeightSetTest, CopyFailureRollsBackCurrentAndPriorAllocations) {
  auto source = SafetensorsFile::Open(path_, 1024);
  ASSERT_TRUE(source.ok());
  FailingAllocator allocator;
  FailingCopier copier;
  copier.fail_on_call = 2;
  const auto required = requirements();
  auto loaded = ResidentWeightSet::Load(source.value(), required, allocator, copier, 64);
  EXPECT_FALSE(loaded.ok());
  EXPECT_EQ(copier.calls, 2);
  EXPECT_EQ(allocator.calls, 2);
  EXPECT_EQ(allocator.releases, 2);
}

TEST_F(ResidentWeightSetTest, RejectsRequirementMismatchBeforeAllocating) {
  auto source = SafetensorsFile::Open(path_, 1024);
  ASSERT_TRUE(source.ok());
  FailingAllocator allocator;
  CpuMemoryCopier copier;
  auto wrong = requirements();
  wrong[0].shape = {5};
  EXPECT_FALSE(ResidentWeightSet::Load(source.value(), wrong, allocator, copier, 64).ok());
  EXPECT_EQ(allocator.calls, 0);
}

TEST_F(ResidentWeightSetTest, RejectsDuplicateRequirementBeforeAllocating) {
  auto source = SafetensorsFile::Open(path_, 1024);
  ASSERT_TRUE(source.ok());
  FailingAllocator allocator;
  CpuMemoryCopier copier;
  auto duplicate = requirements();
  duplicate[1] = duplicate[0];
  EXPECT_FALSE(
      ResidentWeightSet::Load(source.value(), duplicate, allocator, copier, 64).ok());
  EXPECT_EQ(allocator.calls, 0);
}

}  // namespace
}  // namespace pih
