#include "pih/model/qwen3_bf16_kernel_bundle.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "pih/core/sha256.h"

namespace pih {
namespace {

class BundleDriver final : public KernelModuleDriver {
 public:
  Result<DriverModuleHandle> load_module(std::span<const std::byte>) override {
    ++loads;
    return UINT64_C(41);
  }
  Status unload_module(DriverModuleHandle) override {
    ++unloads;
    return Status::Ok();
  }
  Result<DriverFunctionHandle> get_function(
      DriverModuleHandle, std::string_view symbol) override {
    if (fail_symbol == symbol) return Status::InvalidArgument("missing symbol");
    const auto handle = next_handle++;
    functions.emplace(handle, std::string(symbol));
    return handle;
  }
  Result<std::uint32_t> get_parameter_count(
      DriverFunctionHandle function) override {
    const auto& symbol = functions.at(function);
    if (symbol.find("sample_f32_packed") != std::string::npos)
      return UINT32_C(14);
    if (symbol.find("sample_hidden_bf16_packed") != std::string::npos)
      return UINT32_C(7);
    if (symbol.find("greedy_argmax_f32_packed") != std::string::npos)
      return UINT32_C(5);
    if (symbol.find("kv_append_bf16_packed") != std::string::npos)
      return UINT32_C(13);
    if (symbol.find("paged_gqa_bf16_packed") != std::string::npos)
      return UINT32_C(16);
    if (symbol.find("embedding") != std::string::npos) return UINT32_C(6);
    if (symbol.find("rms_norm") != std::string::npos) return UINT32_C(6);
    if (symbol.find("rope_angles") != std::string::npos) return UINT32_C(5);
    if (symbol.find("rope") != std::string::npos) return UINT32_C(7);
    if (symbol.find("kv_append") != std::string::npos) return UINT32_C(11);
    if (symbol.find("paged_gqa") != std::string::npos) return UINT32_C(14);
    if (symbol.find("teacher_forced_metric") != std::string::npos)
      return UINT32_C(8);
    return UINT32_C(4);
  }
  Result<DriverParameterInfo> get_parameter_info(
      DriverFunctionHandle function, std::uint32_t ordinal) override {
    const auto count = get_parameter_count(function).value();
    if (ordinal >= count) return Status::InvalidArgument("bad ordinal");
    const auto& symbol = functions.at(function);
    if (symbol.find("sample_f32_packed") != std::string::npos) {
      constexpr std::array<std::uint32_t, 14> offsets{
          0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 92, 96};
      constexpr std::array<std::uint32_t, 14> sizes{
          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4};
      return DriverParameterInfo{offsets[ordinal], sizes[ordinal]};
    }
    if (symbol.find("sample_hidden_bf16_packed") != std::string::npos) {
      constexpr std::array<std::uint32_t, 7> offsets{0, 8, 16, 24, 32, 36, 40};
      constexpr std::array<std::uint32_t, 7> sizes{8, 8, 8, 8, 4, 4, 4};
      return DriverParameterInfo{offsets[ordinal], sizes[ordinal]};
    }
    if (symbol.find("greedy_argmax_f32_packed") != std::string::npos) {
      constexpr std::array<std::uint32_t, 5> offsets{0, 8, 16, 24, 28};
      constexpr std::array<std::uint32_t, 5> sizes{8, 8, 8, 4, 4};
      return DriverParameterInfo{offsets[ordinal], sizes[ordinal]};
    }
    if (symbol.find("kv_append_bf16_packed") != std::string::npos) {
      constexpr std::array<std::uint32_t, 13> offsets{
          0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 92};
      constexpr std::array<std::uint32_t, 13> sizes{
          8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 4, 4};
      return DriverParameterInfo{offsets[ordinal], sizes[ordinal]};
    }
    if (symbol.find("paged_gqa_bf16_packed") != std::string::npos) {
      constexpr std::array<std::uint32_t, 16> offsets{
          0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 92, 96, 100, 104};
      constexpr std::array<std::uint32_t, 16> sizes{
          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4};
      return DriverParameterInfo{offsets[ordinal], sizes[ordinal]};
    }
    const bool rms = symbol.find("rms_norm") != std::string::npos;
    if (rms && ordinal == 5) return DriverParameterInfo{40, 4};
    if (symbol.find("kv_append") != std::string::npos) {
      constexpr std::array<std::uint32_t, 11> offsets{
          0, 8, 16, 24, 32, 40, 48, 56, 60, 64, 72};
      constexpr std::array<std::uint32_t, 11> sizes{
          8, 8, 8, 8, 8, 8, 8, 4, 4, 8, 4};
      return DriverParameterInfo{offsets[ordinal], sizes[ordinal]};
    }
    if (symbol.find("paged_gqa") != std::string::npos) {
      constexpr std::array<std::uint32_t, 14> offsets{
          0, 8, 16, 24, 32, 40, 48, 52, 56, 64, 68, 72, 76, 80};
      constexpr std::array<std::uint32_t, 14> sizes{
          8, 8, 8, 8, 8, 8, 4, 4, 8, 4, 4, 4, 4, 4};
      return DriverParameterInfo{offsets[ordinal], sizes[ordinal]};
    }
    if (symbol.find("teacher_forced_metric") != std::string::npos) {
      constexpr std::array<std::uint32_t, 8> offsets{
          0, 8, 16, 24, 32, 40, 48, 52};
      constexpr std::array<std::uint32_t, 8> sizes{
          8, 8, 8, 8, 8, 8, 4, 4};
      return DriverParameterInfo{offsets[ordinal], sizes[ordinal]};
    }
    return DriverParameterInfo{ordinal * 8, 8};
  }

  std::string fail_symbol;
  int loads = 0;
  int unloads = 0;

 private:
  DriverFunctionHandle next_handle = 100;
  std::unordered_map<DriverFunctionHandle, std::string> functions;
};

class BundleFiles final {
 public:
  BundleFiles() {
    static std::uint64_t counter = 0;
    const auto base = std::filesystem::temp_directory_path() /
                      ("pih-qwen-bundle-" + std::to_string(counter++));
    cubin_path = base.string() + ".cubin";
    manifest_path = base.string() + ".json";
    std::vector<std::byte> bytes(64);
    bytes[0] = std::byte{0x7f};
    bytes[1] = std::byte{'E'};
    bytes[2] = std::byte{'L'};
    bytes[3] = std::byte{'F'};
    bytes[4] = std::byte{2};
    bytes[5] = std::byte{1};
    bytes[6] = std::byte{1};
    bytes[18] = std::byte{0xbe};
    bytes[20] = std::byte{1};
    const auto digest = sha256(bytes).value().hex();
    std::ofstream cubin(cubin_path, std::ios::binary | std::ios::trunc);
    cubin.write(reinterpret_cast<const char*>(bytes.data()), 64);
    std::ofstream manifest(manifest_path, std::ios::binary | std::ios::trunc);
    manifest << "{\"schema\":\"pih.cubin_artifact.v1\","
                "\"target_sm\":\"sm_89\",\"producer_toolkit\":\"13.2\","
                "\"producer_flags_sha256\":\"" << std::string(64, 'a')
             << "\",\"cubin_sha256\":\"" << digest
             << "\",\"cubin_bytes\":64}";
  }
  ~BundleFiles() {
    std::filesystem::remove(cubin_path);
    std::filesystem::remove(manifest_path);
  }
  std::filesystem::path cubin_path;
  std::filesystem::path manifest_path;
};

TEST(QwenBf16KernelBundleTest, ResolvesLegacyAndPackedFunctionsTransactionally) {
  BundleFiles files;
  BundleDriver driver;
  {
    auto bundle = QwenBf16KernelBundle::Load(
        driver, files.manifest_path, files.cubin_path, 8, 9, 1024);
    ASSERT_TRUE(bundle.ok()) << bundle.status().message();
    EXPECT_EQ(bundle->target_sm(), 89);
    auto embedding = bundle->function(QwenBf16Primitive::kEmbedding);
    auto rope = bundle->function(QwenBf16Primitive::kRope);
    auto paged_gqa = bundle->function(QwenBf16Primitive::kPagedGqa);
    auto rope_angles = bundle->function(QwenBf16Primitive::kRopeAngles);
    auto argmax = bundle->function(QwenBf16Primitive::kGreedyArgmax);
    auto metric =
        bundle->function(QwenBf16Primitive::kTeacherForcedMetric);
    ASSERT_TRUE(embedding.ok());
    ASSERT_TRUE(rope.ok());
    ASSERT_TRUE(paged_gqa.ok());
    ASSERT_TRUE(rope_angles.ok());
    ASSERT_TRUE(argmax.ok());
    ASSERT_TRUE(metric.ok());
    EXPECT_EQ(embedding.value()->handle, 100);
    EXPECT_EQ(rope.value()->handle, 104);
    EXPECT_EQ(paged_gqa.value()->handle, 106);
    EXPECT_EQ(rope_angles.value()->handle, 107);
    EXPECT_EQ(argmax.value()->handle, 108);
    EXPECT_EQ(metric.value()->handle, 109);
    auto packed_embedding =
        bundle->packed_function(QwenBf16PackedPrimitive::kEmbedding);
    auto packed_kv =
        bundle->packed_function(QwenBf16PackedPrimitive::kKvAppend);
    auto packed_gqa =
        bundle->packed_function(QwenBf16PackedPrimitive::kPagedGqa);
    auto sample_hidden =
        bundle->packed_function(QwenBf16PackedPrimitive::kSampleHidden);
    auto packed_argmax =
        bundle->packed_function(QwenBf16PackedPrimitive::kGreedyArgmax);
    auto packed_sampler =
        bundle->packed_function(QwenBf16PackedPrimitive::kSampler);
    ASSERT_TRUE(packed_embedding.ok());
    ASSERT_TRUE(packed_kv.ok());
    ASSERT_TRUE(packed_gqa.ok());
    ASSERT_TRUE(sample_hidden.ok());
    ASSERT_TRUE(packed_argmax.ok());
    ASSERT_TRUE(packed_sampler.ok());
    EXPECT_EQ(packed_embedding.value()->handle, 110);
    EXPECT_EQ(packed_kv.value()->handle, 112);
    EXPECT_EQ(packed_gqa.value()->handle, 113);
    EXPECT_EQ(sample_hidden.value()->handle, 114);
    EXPECT_EQ(packed_argmax.value()->handle, 115);
    EXPECT_EQ(packed_sampler.value()->handle, 116);
    EXPECT_EQ(bundle->functions().size(), 17U);
    ASSERT_EQ(bundle->legacy_functions().size(), 9U);
    ASSERT_EQ(bundle->packed_functions().size(), 7U);
    EXPECT_EQ(bundle->legacy_functions().front().handle, 100U);
    EXPECT_EQ(bundle->legacy_functions().back().handle, 108U);
    EXPECT_EQ(bundle->packed_functions().front().handle, 110U);
    EXPECT_EQ(bundle->packed_functions().back().handle, 116U);
    EXPECT_FALSE(bundle->function(static_cast<QwenBf16Primitive>(255)).ok());
    EXPECT_EQ(driver.loads, 1);
    EXPECT_EQ(driver.unloads, 0);
  }
  EXPECT_EQ(driver.unloads, 1);
}

TEST(QwenBf16KernelBundleTest, RollsBackWhenPackedEntrypointIsMissing) {
  BundleFiles files;
  BundleDriver driver;
  driver.fail_symbol = "pih_qwen_paged_gqa_bf16_packed_v3";
  EXPECT_FALSE(QwenBf16KernelBundle::Load(
                   driver, files.manifest_path, files.cubin_path, 8, 9, 1024)
                   .ok());
  EXPECT_EQ(driver.loads, 1);
  EXPECT_EQ(driver.unloads, 1);
}

TEST(QwenBf16KernelBundleTest, RejectsDeviceMismatchBeforeDriverLoad) {
  BundleFiles files;
  BundleDriver driver;
  EXPECT_FALSE(QwenBf16KernelBundle::Load(
                   driver, files.manifest_path, files.cubin_path, 9, 0, 1024)
                   .ok());
  EXPECT_FALSE(QwenBf16KernelBundle::Load(
                   driver, files.manifest_path, files.cubin_path, 8, 0, 1024)
                   .ok());
  EXPECT_EQ(driver.loads, 0);
}

TEST(QwenBf16KernelBundleTest, RollsBackWhenAnyEntrypointIsMissing) {
  BundleFiles files;
  BundleDriver driver;
  driver.fail_symbol = "pih_qwen_rms_norm_bf16_v1";
  EXPECT_FALSE(QwenBf16KernelBundle::Load(
                   driver, files.manifest_path, files.cubin_path, 8, 9, 1024)
                   .ok());
  EXPECT_EQ(driver.loads, 1);
  EXPECT_EQ(driver.unloads, 1);
}

}  // namespace
}  // namespace pih
