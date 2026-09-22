#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "pih/backend/cuda/verified_kernel_module.h"

namespace pih {
namespace {

class FakeKernelModuleDriver final : public KernelModuleDriver {
 public:
  Result<DriverModuleHandle> load_module(std::span<const std::byte> cubin) override {
    ++load_calls;
    if (fail_load) return Status::Unavailable("injected module load failure");
    if (cubin.empty()) return Status::InvalidArgument("empty fake cubin");
    return module_handle;
  }

  Status unload_module(DriverModuleHandle module) override {
    ++unload_calls;
    last_unloaded = module;
    return Status::Ok();
  }

  Result<DriverFunctionHandle> get_function(
      DriverModuleHandle module, std::string_view symbol) override {
    if (module != module_handle) return Status::InvalidArgument("wrong module");
    const auto found = functions.find(std::string(symbol));
    if (found == functions.end()) return Status::Unavailable("missing function");
    return found->second;
  }

  Result<std::uint32_t> get_parameter_count(
      DriverFunctionHandle function) override {
    const auto found = parameters.find(function);
    if (found == parameters.end()) return Status::Unavailable("missing parameters");
    return static_cast<std::uint32_t>(found->second.size());
  }

  Result<DriverParameterInfo> get_parameter_info(
      DriverFunctionHandle function, std::uint32_t ordinal) override {
    const auto found = parameters.find(function);
    if (found == parameters.end() || ordinal >= found->second.size()) {
      return Status::InvalidArgument("parameter ordinal is absent");
    }
    return found->second[ordinal];
  }

  bool fail_load = false;
  int load_calls = 0;
  int unload_calls = 0;
  DriverModuleHandle last_unloaded = 0;
  DriverModuleHandle module_handle = 17;
  std::unordered_map<std::string, DriverFunctionHandle> functions;
  std::unordered_map<DriverFunctionHandle, std::vector<DriverParameterInfo>>
      parameters;
};

class VerifiedKernelModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = std::filesystem::temp_directory_path() /
            ("pih-module-" + std::to_string(counter_++) + ".cubin");
    cubin_bytes_.resize(64);
    cubin_bytes_[0] = std::byte{0x7f};
    cubin_bytes_[1] = std::byte{'E'};
    cubin_bytes_[2] = std::byte{'L'};
    cubin_bytes_[3] = std::byte{'F'};
    cubin_bytes_[4] = std::byte{2};
    cubin_bytes_[5] = std::byte{1};
    cubin_bytes_[6] = std::byte{1};
    cubin_bytes_[18] = std::byte{0xbe};
    cubin_bytes_[20] = std::byte{1};
    std::ofstream stream(path_, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(cubin_bytes_.data()),
                 static_cast<std::streamsize>(cubin_bytes_.size()));
  }
  void TearDown() override { std::filesystem::remove(path_); }

  Result<VerifiedCubin> cubin() {
    auto digest = sha256(cubin_bytes_);
    if (!digest.ok()) return digest.status();
    return VerifiedCubin::Load(path_, cubin_bytes_.size(), digest.value());
  }

  Result<KernelSignatureManifest> manifest(std::string cubin_digest) {
    const std::vector<KernelParameterSpec> parameters{
        {"input", KernelWireType::kDevicePointerU64, 0, "read_bf16"},
        {"count", KernelWireType::kU32, 8, "positive_count"}};
    return KernelSignatureManifest::Create(
        "qwen.residual.bf16", std::move(cubin_digest), std::string(64, 'b'),
        parameters, 12);
  }

  std::filesystem::path path_;
  std::vector<std::byte> cubin_bytes_;
  static inline std::uint64_t counter_ = 1;
};

TEST_F(VerifiedKernelModuleTest, PublishesOnlyResolvedVerifiedFunctions) {
  auto artifact = cubin();
  ASSERT_TRUE(artifact.ok());
  auto signature = manifest(artifact->digest().hex());
  ASSERT_TRUE(signature.ok());
  FakeKernelModuleDriver driver;
  driver.functions["residual_bf16"] = 23;
  driver.parameters[23] = {{0, 8}, {8, 4}};
  const KernelEntrypointRequest request{"residual_bf16", &signature.value()};

  {
    auto module = VerifiedKernelModule::Load(driver, artifact.value(), {&request, 1});
    ASSERT_TRUE(module.ok()) << module.status().message();
    EXPECT_EQ(module->module_handle(), 17);
    ASSERT_EQ(module->functions().size(), 1);
    EXPECT_EQ(module->functions()[0].handle, 23);
    EXPECT_EQ(module->functions()[0].logical_id, "qwen.residual.bf16");
    EXPECT_EQ(driver.unload_calls, 0);
  }
  EXPECT_EQ(driver.unload_calls, 1);
  EXPECT_EQ(driver.last_unloaded, 17);
}

TEST_F(VerifiedKernelModuleTest, RollsBackMissingFunctionAndSignatureDrift) {
  auto artifact = cubin();
  ASSERT_TRUE(artifact.ok());
  auto signature = manifest(artifact->digest().hex());
  ASSERT_TRUE(signature.ok());
  const KernelEntrypointRequest request{"residual_bf16", &signature.value()};
  FakeKernelModuleDriver driver;

  EXPECT_FALSE(
      VerifiedKernelModule::Load(driver, artifact.value(), {&request, 1}).ok());
  EXPECT_EQ(driver.unload_calls, 1);

  driver.functions["residual_bf16"] = 23;
  driver.parameters[23] = {{0, 8}, {16, 4}};
  EXPECT_FALSE(
      VerifiedKernelModule::Load(driver, artifact.value(), {&request, 1}).ok());
  EXPECT_EQ(driver.unload_calls, 2);
}

TEST_F(VerifiedKernelModuleTest, RejectsRequestDriftBeforeDriverLoad) {
  auto artifact = cubin();
  ASSERT_TRUE(artifact.ok());
  auto wrong = manifest(std::string(64, 'a'));
  ASSERT_TRUE(wrong.ok());
  FakeKernelModuleDriver driver;
  const KernelEntrypointRequest wrong_digest{"residual_bf16", &wrong.value()};
  EXPECT_FALSE(VerifiedKernelModule::Load(driver, artifact.value(),
                                          {&wrong_digest, 1})
                   .ok());
  EXPECT_EQ(driver.load_calls, 0);

  auto correct = manifest(artifact->digest().hex());
  ASSERT_TRUE(correct.ok());
  const std::array<KernelEntrypointRequest, 2> duplicate_logical_id{
      KernelEntrypointRequest{"residual_bf16", &correct.value()},
      KernelEntrypointRequest{"residual_bf16_alt", &correct.value()}};
  EXPECT_FALSE(VerifiedKernelModule::Load(driver, artifact.value(),
                                          duplicate_logical_id)
                   .ok());
  EXPECT_EQ(driver.load_calls, 0);

  const KernelEntrypointRequest null_manifest{"residual_bf16", nullptr};
  EXPECT_FALSE(VerifiedKernelModule::Load(driver, artifact.value(),
                                          {&null_manifest, 1})
                   .ok());
  EXPECT_EQ(driver.load_calls, 0);
}

}  // namespace
}  // namespace pih
