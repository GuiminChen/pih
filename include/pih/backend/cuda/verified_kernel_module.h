#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pih/backend/cuda/kernel_signature_manifest.h"
#include "pih/backend/cuda/verified_cubin.h"
#include "pih/core/result.h"

namespace pih {

using DriverModuleHandle = std::uintptr_t;
using DriverFunctionHandle = std::uintptr_t;

struct DriverParameterInfo final {
  std::uint32_t offset;
  std::uint32_t size;
};

class KernelModuleDriver {
 public:
  virtual ~KernelModuleDriver() = default;
  virtual Result<DriverModuleHandle> load_module(
      std::span<const std::byte> cubin) = 0;
  virtual Status unload_module(DriverModuleHandle module) = 0;
  virtual Result<DriverFunctionHandle> get_function(
      DriverModuleHandle module, std::string_view symbol) = 0;
  virtual Result<std::uint32_t> get_parameter_count(
      DriverFunctionHandle function) = 0;
  virtual Result<DriverParameterInfo> get_parameter_info(
      DriverFunctionHandle function, std::uint32_t ordinal) = 0;
};

struct KernelEntrypointRequest final {
  std::string_view symbol;
  const KernelSignatureManifest* manifest;
};

struct ResolvedKernelFunction final {
  std::string symbol;
  std::string logical_id;
  std::string selected_cubin_sha256;
  std::string parameter_abi_sha256;
  DriverFunctionHandle handle;
};

class VerifiedKernelModule final {
 public:
  static constexpr std::size_t kMaximumEntrypoints = 1024;
  static constexpr std::size_t kMaximumSymbolBytes = 256;

  static Result<VerifiedKernelModule> Load(
      KernelModuleDriver& driver, const VerifiedCubin& cubin,
      std::span<const KernelEntrypointRequest> entrypoints);
  ~VerifiedKernelModule();

  VerifiedKernelModule(const VerifiedKernelModule&) = delete;
  VerifiedKernelModule& operator=(const VerifiedKernelModule&) = delete;
  VerifiedKernelModule(VerifiedKernelModule&& other) noexcept;
  VerifiedKernelModule& operator=(VerifiedKernelModule&& other) noexcept;

  [[nodiscard]] DriverModuleHandle module_handle() const noexcept {
    return module_;
  }
  [[nodiscard]] std::span<const ResolvedKernelFunction> functions() const noexcept {
    return functions_;
  }

 private:
  VerifiedKernelModule(KernelModuleDriver* driver, DriverModuleHandle module,
                       std::vector<ResolvedKernelFunction> functions)
      : driver_(driver), module_(module), functions_(std::move(functions)) {}
  void reset() noexcept;

  KernelModuleDriver* driver_ = nullptr;
  DriverModuleHandle module_ = 0;
  std::vector<ResolvedKernelFunction> functions_;
};

}  // namespace pih
