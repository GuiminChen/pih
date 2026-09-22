#pragma once
#include <filesystem>
#include "pih/model/qwen3_int4_kernel_bundle.h"
#include "pih/model/qwen3_bf16_kernel_bundle.h"
namespace pih{
class NvidiaQwenInt4KernelAssets final{public:
 static Result<QwenInt4KernelBundle>Load(KernelModuleDriver& driver,
  const std::filesystem::path& cubin_root,std::int32_t device_ordinal,
  std::uint64_t maximum_cubin_bytes);
 static Result<QwenBf16KernelBundle>LoadPacked(KernelModuleDriver& driver,
  const std::filesystem::path& cubin_root,std::int32_t device_ordinal,
  std::uint64_t maximum_cubin_bytes);
};}
