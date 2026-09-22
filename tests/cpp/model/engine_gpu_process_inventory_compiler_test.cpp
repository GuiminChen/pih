#include "pih/model/engine_gpu_process_inventory_compiler.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest compiler_lease() {
  Sha256Digest value{}; value.bytes.fill(std::byte{6}); return value;
}

class Provider final : public EngineGpuProcessInventoryProvider {
 public:
  Result<std::vector<EngineGpuProcessInventoryEntry>> collect(
      std::span<const std::uint64_t> devices) override {
    ++calls;
    if (!status.ok()) return status;
    if (!entries.empty()) return entries;
    std::vector<EngineGpuProcessInventoryEntry> result;
    for (const auto device : devices) result.push_back({device, {}});
    return result;
  }
  Status status = Status::Ok();
  std::vector<EngineGpuProcessInventoryEntry> entries;
  int calls = 0;
};

TEST(EngineGpuProcessInventoryCompilerTest, CompilesConsecutiveVisibleReceipts) {
  Provider provider;
  const std::uint64_t devices[]{10, 20};
  auto compiler = EngineGpuProcessInventoryCompiler::Create(
      7, compiler_lease(), devices, provider).value();
  auto first = compiler.capture(100, 101);
  ASSERT_TRUE(first.ok());
  EXPECT_TRUE(first->visibility_complete);
  EXPECT_EQ(first->sample_identity, 1U);
  auto second = compiler.capture(101, 102);
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(second->sample_identity, 2U);
  EXPECT_EQ(provider.calls, 2);
}

TEST(EngineGpuProcessInventoryCompilerTest, UnavailableBecomesExplicitUnknown) {
  Provider provider;
  provider.status = Status::Unavailable("NVML visibility unavailable");
  const std::uint64_t devices[]{10, 20};
  auto compiler = EngineGpuProcessInventoryCompiler::Create(
      7, compiler_lease(), devices, provider).value();
  auto receipt = compiler.capture(100, 101);
  ASSERT_TRUE(receipt.ok());
  EXPECT_FALSE(receipt->visibility_complete);
  ASSERT_EQ(receipt->devices.size(), 2U);
  EXPECT_TRUE(receipt->devices[0].sorted_process_identities.empty());
  EXPECT_TRUE(receipt->devices[1].sorted_process_identities.empty());
}

TEST(EngineGpuProcessInventoryCompilerTest, MalformedProviderPoisonsCompiler) {
  Provider provider;
  provider.entries = {{20, {}}, {10, {}}};
  const std::uint64_t devices[]{10, 20};
  auto compiler = EngineGpuProcessInventoryCompiler::Create(
      7, compiler_lease(), devices, provider).value();
  EXPECT_FALSE(compiler.capture(100, 101).ok());
  provider.entries = {{10, {}}, {20, {}}};
  EXPECT_FALSE(compiler.capture(101, 102).ok());
}

TEST(EngineGpuProcessInventoryCompilerTest, RejectsTimeDriftAndHardFailure) {
  Provider provider;
  const std::uint64_t devices[]{10};
  auto compiler = EngineGpuProcessInventoryCompiler::Create(
      7, compiler_lease(), devices, provider).value();
  EXPECT_FALSE(compiler.capture(101, 100).ok());
  auto second = EngineGpuProcessInventoryCompiler::Create(
      7, compiler_lease(), devices, provider).value();
  provider.status = Status::Internal("provider corrupted");
  EXPECT_FALSE(second.capture(100, 101).ok());
}

}  // namespace
}  // namespace pih
