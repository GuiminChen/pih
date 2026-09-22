#include "pih/model/nvidia_qwen3_int4_kernel_assets.h"
#include <gtest/gtest.h>
namespace pih{namespace{
class Driver final:public KernelModuleDriver{public:
 Result<DriverModuleHandle>load_module(std::span<const std::byte>)override{return 1;}
 Status unload_module(DriverModuleHandle)override{return Status::Ok();}
 Result<DriverFunctionHandle>get_function(DriverModuleHandle,std::string_view)override{return 1;}
 Result<std::uint32_t>get_parameter_count(DriverFunctionHandle)override{return 0;}
 Result<DriverParameterInfo>get_parameter_info(DriverFunctionHandle,std::uint32_t)override{return DriverParameterInfo{0,0};}
};
TEST(NvidiaQwenInt4KernelAssetsTest,RejectsIdentityBeforeCudaAccess){Driver d;
 EXPECT_FALSE(NvidiaQwenInt4KernelAssets::Load(d,{},0,1).ok());
 EXPECT_FALSE(NvidiaQwenInt4KernelAssets::Load(d,"kernels",-1,1).ok());
 EXPECT_FALSE(NvidiaQwenInt4KernelAssets::Load(d,"kernels",0,0).ok());}
}}
