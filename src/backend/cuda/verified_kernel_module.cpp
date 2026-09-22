#include "pih/backend/cuda/verified_kernel_module.h"

#include <unordered_set>
#include <utility>

namespace pih {
namespace {

Status validate_entrypoint_request(const KernelEntrypointRequest& request) {
  if (request.symbol.empty() ||
      request.symbol.size() > VerifiedKernelModule::kMaximumSymbolBytes) {
    return Status::InvalidArgument("kernel symbol is empty or oversized");
  }
  if (request.symbol.find('\0') != std::string_view::npos) {
    return Status::InvalidArgument("kernel symbol contains embedded NUL");
  }
  if (request.manifest == nullptr) {
    return Status::InvalidArgument("kernel entrypoint has no signature manifest");
  }
  return Status::Ok();
}

Status verify_signature(KernelModuleDriver& driver,
                        DriverFunctionHandle function,
                        const KernelSignatureManifest& manifest) {
  auto count = driver.get_parameter_count(function);
  if (!count.ok()) return count.status();
  if (count.value() != manifest.parameter_count()) {
    return Status::InvalidArgument("loaded kernel parameter count differs from manifest");
  }
  for (std::uint32_t ordinal = 0; ordinal < count.value(); ++ordinal) {
    auto actual = driver.get_parameter_info(function, ordinal);
    if (!actual.ok()) return actual.status();
    auto expected_size = kernel_wire_size(manifest.parameter(ordinal).wire_type);
    if (!expected_size.ok()) return expected_size.status();
    if (actual->offset != manifest.parameter(ordinal).device_layout_offset ||
        actual->size != expected_size.value()) {
      return Status::InvalidArgument(
          "loaded kernel parameter layout differs from manifest");
    }
  }
  return Status::Ok();
}

}  // namespace

Result<VerifiedKernelModule> VerifiedKernelModule::Load(
    KernelModuleDriver& driver, const VerifiedCubin& cubin,
    std::span<const KernelEntrypointRequest> entrypoints) {
  if (entrypoints.empty() || entrypoints.size() > kMaximumEntrypoints) {
    return Status::InvalidArgument("kernel entrypoint count is outside bounds");
  }
  std::unordered_set<std::string_view> symbols;
  std::unordered_set<std::string_view> logical_ids;
  symbols.reserve(entrypoints.size());
  logical_ids.reserve(entrypoints.size());
  for (const auto& entrypoint : entrypoints) {
    const Status valid = validate_entrypoint_request(entrypoint);
    if (!valid.ok()) return valid;
    if (!symbols.insert(entrypoint.symbol).second) {
      return Status::InvalidArgument("kernel entrypoint symbol is duplicated");
    }
    if (!logical_ids.insert(entrypoint.manifest->logical_id()).second) {
      return Status::InvalidArgument("kernel logical id is duplicated");
    }
    if (entrypoint.manifest->selected_cubin_sha256() != cubin.digest().hex()) {
      return Status::InvalidArgument("kernel manifest names a different cubin");
    }
  }

  auto module = driver.load_module(cubin.bytes());
  if (!module.ok()) return module.status();
  if (module.value() == 0) {
    return Status::Internal("driver returned a null module handle");
  }

  std::vector<ResolvedKernelFunction> resolved;
  resolved.reserve(entrypoints.size());
  for (const auto& entrypoint : entrypoints) {
    auto function = driver.get_function(module.value(), entrypoint.symbol);
    if (!function.ok() || function.value() == 0) {
      driver.unload_module(module.value());
      if (!function.ok()) return function.status();
      return Status::Internal("driver returned a null function handle");
    }
    const Status signature =
        verify_signature(driver, function.value(), *entrypoint.manifest);
    if (!signature.ok()) {
      driver.unload_module(module.value());
      return signature;
    }
    resolved.push_back({std::string(entrypoint.symbol),
                        std::string(entrypoint.manifest->logical_id()),
                        std::string(entrypoint.manifest->selected_cubin_sha256()),
                        std::string(entrypoint.manifest->parameter_abi_sha256()),
                        function.value()});
  }
  return VerifiedKernelModule(&driver, module.value(), std::move(resolved));
}

VerifiedKernelModule::~VerifiedKernelModule() { reset(); }

VerifiedKernelModule::VerifiedKernelModule(VerifiedKernelModule&& other) noexcept
    : driver_(std::exchange(other.driver_, nullptr)),
      module_(std::exchange(other.module_, 0)),
      functions_(std::move(other.functions_)) {}

VerifiedKernelModule& VerifiedKernelModule::operator=(
    VerifiedKernelModule&& other) noexcept {
  if (this != &other) {
    reset();
    driver_ = std::exchange(other.driver_, nullptr);
    module_ = std::exchange(other.module_, 0);
    functions_ = std::move(other.functions_);
  }
  return *this;
}

void VerifiedKernelModule::reset() noexcept {
  if (driver_ != nullptr && module_ != 0) {
    try {
      driver_->unload_module(module_);
    } catch (...) {
      // Destruction cannot report cleanup failure. Bootstrap rollback paths call
      // unload directly before returning their primary verification failure.
    }
  }
  driver_ = nullptr;
  module_ = 0;
  functions_.clear();
}

}  // namespace pih
