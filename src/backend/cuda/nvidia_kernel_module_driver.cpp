#include "pih/backend/cuda/nvidia_kernel_module_driver.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <limits>
#include <string>

#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/backend/cuda/cuda_status.h"

#if CUDA_VERSION < 13020
#error "PIH verified kernel ABI requires CUDA Toolkit 13.2 or newer"
#endif

namespace pih {
namespace {

CUmodule as_module(DriverModuleHandle handle) noexcept {
  return reinterpret_cast<CUmodule>(handle);
}

CUfunction as_function(DriverFunctionHandle handle) noexcept {
  return reinterpret_cast<CUfunction>(handle);
}

}  // namespace

Result<NvidiaKernelModuleDriver> NvidiaKernelModuleDriver::Create() {
  CUcontext context = nullptr;
  CUresult result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuCtxGetCurrent");
  }
  if (context == nullptr) {
    return Status::FailedPrecondition(
        "NVIDIA kernel module driver requires a current CUDA context");
  }
  CUmoduleLoadingMode loading_mode{};
  result = cuModuleGetLoadingMode(&loading_mode);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuModuleGetLoadingMode");
  }
  if (loading_mode != CU_MODULE_EAGER_LOADING) {
    return Status::FailedPrecondition(
        "PIH requires CUDA_MODULE_LOADING=EAGER");
  }
  return NvidiaKernelModuleDriver(reinterpret_cast<std::uintptr_t>(context));
}

Status NvidiaKernelModuleDriver::require_current_context() const {
  CUcontext current = nullptr;
  const CUresult result = cuCtxGetCurrent(&current);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuCtxGetCurrent");
  }
  if (reinterpret_cast<std::uintptr_t>(current) != context_) {
    return Status::FailedPrecondition("current CUDA context differs from module owner");
  }
  return Status::Ok();
}

Result<DriverModuleHandle> NvidiaKernelModuleDriver::load_module(
    std::span<const std::byte> cubin) {
  if (cubin.empty()) return Status::InvalidArgument("cannot load an empty cubin");
  const Status context = require_current_context();
  if (!context.ok()) return context;
  CUmodule module = nullptr;
  const CUresult result = cuModuleLoadData(&module, cubin.data());
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuModuleLoadData");
  }
  return reinterpret_cast<DriverModuleHandle>(module);
}

Status NvidiaKernelModuleDriver::unload_module(
    DriverModuleHandle module) {
  if (module == 0) return Status::InvalidArgument("cannot unload a null module");
  const Status context = require_current_context();
  if (!context.ok()) return context;
  return cuda_driver_status(cuModuleUnload(as_module(module)), "cuModuleUnload");
}

Result<DriverFunctionHandle> NvidiaKernelModuleDriver::get_function(
    DriverModuleHandle module, std::string_view symbol) {
  if (module == 0 || symbol.empty()) {
    return Status::InvalidArgument("module and kernel symbol are required");
  }
  const Status context = require_current_context();
  if (!context.ok()) return context;
  const std::string terminated_symbol(symbol);
  CUfunction function = nullptr;
  const CUresult result =
      cuModuleGetFunction(&function, as_module(module), terminated_symbol.c_str());
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuModuleGetFunction");
  }
  return reinterpret_cast<DriverFunctionHandle>(function);
}

Result<std::uint32_t> NvidiaKernelModuleDriver::get_parameter_count(
    DriverFunctionHandle function) {
  if (function == 0) return Status::InvalidArgument("function handle is null");
  const Status context = require_current_context();
  if (!context.ok()) return context;
  std::size_t count = 0;
  const CUresult result = cuFuncGetParamCount(as_function(function), &count);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuFuncGetParamCount");
  }
  if (count > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted("kernel parameter count exceeds u32");
  }
  return static_cast<std::uint32_t>(count);
}

Result<DriverParameterInfo> NvidiaKernelModuleDriver::get_parameter_info(
    DriverFunctionHandle function, std::uint32_t ordinal) {
  if (function == 0) return Status::InvalidArgument("function handle is null");
  const Status context = require_current_context();
  if (!context.ok()) return context;
  std::size_t offset = 0;
  std::size_t size = 0;
  const CUresult result =
      cuFuncGetParamInfo(as_function(function), ordinal, &offset, &size);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuFuncGetParamInfo");
  }
  if (offset > std::numeric_limits<std::uint32_t>::max() ||
      size > std::numeric_limits<std::uint32_t>::max()) {
    return Status::ResourceExhausted("kernel parameter layout exceeds u32");
  }
  return DriverParameterInfo{static_cast<std::uint32_t>(offset),
                             static_cast<std::uint32_t>(size)};
}

Status NvidiaKernelModuleDriver::launch(
    DriverFunctionHandle function, const KernelLaunchGeometry& geometry,
    DriverStreamHandle stream, void** kernel_params) {
  if (function == 0 || stream == 0 || kernel_params == nullptr) {
    return Status::InvalidArgument(
        "kernel launch requires function, non-default stream, and parameters");
  }
  const Status context = require_current_context();
  if (!context.ok()) return context;
  const Status clean_before =
      cuda_status(cudaPeekAtLastError(), "cudaPeekAtLastError before launch");
  if (!clean_before.ok()) return clean_before;
  const auto cuda_stream = reinterpret_cast<CUstream>(stream);
  const CUresult result = cuLaunchKernel(
      as_function(function), geometry.grid_x(), geometry.grid_y(),
      geometry.grid_z(), geometry.block_x(), geometry.block_y(),
      geometry.block_z(), geometry.dynamic_shared_bytes(), cuda_stream,
      kernel_params, nullptr);
  const Status clean_after =
      cuda_status(cudaPeekAtLastError(), "cudaPeekAtLastError after launch");
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(result, "cuLaunchKernel");
  }
  return clean_after;
}

}  // namespace pih
