#pragma once

#include "pih/backend/cuda/verified_kernel_module.h"
#include "pih/backend/cuda/verified_kernel_launcher.h"

namespace pih {

class NvidiaKernelModuleDriver final : public KernelModuleDriver,
                                       public KernelLaunchDriver {
 public:
  static Result<NvidiaKernelModuleDriver> Create();

  Result<DriverModuleHandle> load_module(
      std::span<const std::byte> cubin) override;
  Status unload_module(DriverModuleHandle module) override;
  Result<DriverFunctionHandle> get_function(
      DriverModuleHandle module, std::string_view symbol) override;
  Result<std::uint32_t> get_parameter_count(
      DriverFunctionHandle function) override;
  Result<DriverParameterInfo> get_parameter_info(
      DriverFunctionHandle function, std::uint32_t ordinal) override;
  Status launch(DriverFunctionHandle function,
                const KernelLaunchGeometry& geometry,
                DriverStreamHandle stream, void** kernel_params) override;

  [[nodiscard]] std::uintptr_t context_handle() const noexcept {
    return context_;
  }

 private:
  explicit NvidiaKernelModuleDriver(std::uintptr_t context)
      : context_(context) {}
  Status require_current_context() const;

  std::uintptr_t context_;
};

}  // namespace pih
