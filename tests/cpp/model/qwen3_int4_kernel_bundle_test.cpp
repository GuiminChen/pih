#include "pih/model/qwen3_int4_kernel_bundle.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

namespace pih { namespace {

class Int4BundleDriver final : public KernelModuleDriver {
 public:
  Result<DriverModuleHandle> load_module(std::span<const std::byte>) override { ++loads; return 11; }
  Status unload_module(DriverModuleHandle) override { ++unloads; return Status::Ok(); }
  Result<DriverFunctionHandle> get_function(DriverModuleHandle,std::string_view symbol) override {
    if (missing == symbol) return Status::InvalidArgument("missing");
    const auto handle = next_handle++;
    functions.emplace(handle, std::string(symbol));
    return handle;
  }
  Result<std::uint32_t> get_parameter_count(DriverFunctionHandle function) override {
    const auto& symbol = functions.at(function);
    if (symbol.find("embedding") != std::string::npos) return UINT32_C(6);
    if (symbol.find("rms_norm") != std::string::npos) return UINT32_C(6);
    if (symbol.find("rope_angles") != std::string::npos) return UINT32_C(5);
    if (symbol.find("rope_bf16") != std::string::npos) return UINT32_C(7);
    if (symbol.find("kv_append") != std::string::npos) return UINT32_C(11);
    if (symbol.find("paged_gqa") != std::string::npos) return UINT32_C(14);
    if (symbol.find("w4a16") != std::string::npos) return UINT32_C(8);
    return UINT32_C(4);
  }
  Result<DriverParameterInfo> get_parameter_info(DriverFunctionHandle function,std::uint32_t ordinal) override {
    const auto count = get_parameter_count(function).value();
    if(ordinal>=count)return Status::InvalidArgument("ordinal");
    const auto& symbol = functions.at(function);
    if (symbol.find("rms_norm") != std::string::npos && ordinal == 5)
      return DriverParameterInfo{40, 4};
    if (symbol.find("kv_append") != std::string::npos) {
      constexpr std::array<std::uint32_t,11> offsets{0,8,16,24,32,40,48,56,60,64,72};
      constexpr std::array<std::uint32_t,11> sizes{8,8,8,8,8,8,8,4,4,8,4};
      return DriverParameterInfo{offsets[ordinal],sizes[ordinal]};
    }
    if (symbol.find("paged_gqa") != std::string::npos) {
      constexpr std::array<std::uint32_t,14> offsets{0,8,16,24,32,40,48,52,56,64,68,72,76,80};
      constexpr std::array<std::uint32_t,14> sizes{8,8,8,8,8,8,4,4,8,4,4,4,4,4};
      return DriverParameterInfo{offsets[ordinal],sizes[ordinal]};
    }
    return DriverParameterInfo{ordinal*8,8};
  }
  std::string missing; int loads=0,unloads=0;
 private:
  DriverFunctionHandle next_handle=22;
  std::unordered_map<DriverFunctionHandle,std::string> functions;
};

class Int4BundleFiles final {
 public:
  Int4BundleFiles(){static std::uint64_t counter=0;auto base=std::filesystem::temp_directory_path()/("pih-int4-bundle-"+std::to_string(counter++));cubin=base.string()+".cubin";manifest=base.string()+".json";std::vector<std::byte> bytes(64);bytes[0]=std::byte{0x7f};bytes[1]=std::byte{'E'};bytes[2]=std::byte{'L'};bytes[3]=std::byte{'F'};bytes[4]=std::byte{2};bytes[5]=std::byte{1};bytes[6]=std::byte{1};bytes[18]=std::byte{0xbe};bytes[20]=std::byte{1};auto digest=sha256(bytes).value().hex();std::ofstream c(cubin,std::ios::binary);c.write(reinterpret_cast<const char*>(bytes.data()),64);std::ofstream m(manifest);m<<"{\"schema\":\"pih.cubin_artifact.v1\",\"target_sm\":\"sm_89\",\"producer_toolkit\":\"13.2\",\"producer_flags_sha256\":\""<<std::string(64,'a')<<"\",\"cubin_sha256\":\""<<digest<<"\",\"cubin_bytes\":64}";}
  ~Int4BundleFiles(){std::filesystem::remove(cubin);std::filesystem::remove(manifest);} std::filesystem::path cubin,manifest;
};

TEST(QwenInt4KernelBundleTest, ResolvesAllServiceFunctionsInOneModule){Int4BundleFiles files;Int4BundleDriver driver;{auto bundle=QwenInt4KernelBundle::Load(driver,files.manifest,files.cubin,8,9,1024);ASSERT_TRUE(bundle.ok())<<bundle.status().message();EXPECT_EQ(bundle->target_sm(),89);EXPECT_EQ(bundle->functions().size(),10U);EXPECT_EQ(bundle->compatibility_function().handle,31U);EXPECT_EQ(bundle->compatibility_manifest().parameter_count(),8U);auto paged=bundle->bf16_function(QwenBf16Primitive::kPagedGqa);ASSERT_TRUE(paged.ok());EXPECT_EQ(paged.value()->handle,28U);EXPECT_EQ(driver.loads,1);EXPECT_EQ(driver.unloads,0);}EXPECT_EQ(driver.unloads,1);}
TEST(QwenInt4KernelBundleTest, RejectsWrongDeviceBeforeLoad){Int4BundleFiles files;Int4BundleDriver driver;EXPECT_FALSE(QwenInt4KernelBundle::Load(driver,files.manifest,files.cubin,9,0,1024).ok());EXPECT_EQ(driver.loads,0);}
TEST(QwenInt4KernelBundleTest, RollsBackMissingSymbol){Int4BundleFiles files;Int4BundleDriver driver;driver.missing="pih_qwen_w4a16_gemm_compat_v1";EXPECT_FALSE(QwenInt4KernelBundle::Load(driver,files.manifest,files.cubin,8,9,1024).ok());EXPECT_EQ(driver.loads,1);EXPECT_EQ(driver.unloads,1);}

} }  // namespace pih
