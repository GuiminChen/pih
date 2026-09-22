#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "pih/backend/cuda/kernel_signature_manifest.h"
#include "pih/backend/cuda/verified_kernel_module.h"
#include "pih/core/result.h"
#include "pih/model/qwen3_bf16_kernel_manifest.h"
#include "pih/model/qwen3_bf16_packed_kernel_manifest.h"

namespace pih {

class QwenBf16KernelBundle final {
 public:
  static constexpr std::size_t kPrimitiveCount = 10;
  static constexpr std::size_t kExecutionPrimitiveCount = 9;
  static constexpr std::size_t kPackedPrimitiveCount = 7;
  static constexpr std::size_t kFunctionCount =
      kPrimitiveCount + kPackedPrimitiveCount;

  static Result<QwenBf16KernelBundle> Load(
      KernelModuleDriver& driver, const std::filesystem::path& manifest_path,
      const std::filesystem::path& cubin_path,
      std::uint32_t compute_major, std::uint32_t compute_minor,
      std::uint64_t maximum_cubin_bytes);

  QwenBf16KernelBundle(const QwenBf16KernelBundle&) = delete;
  QwenBf16KernelBundle& operator=(const QwenBf16KernelBundle&) = delete;
  QwenBf16KernelBundle(QwenBf16KernelBundle&&) noexcept = default;
  QwenBf16KernelBundle& operator=(QwenBf16KernelBundle&&) noexcept = default;

  [[nodiscard]] Result<const ResolvedKernelFunction*> function(
      QwenBf16Primitive primitive) const;
  [[nodiscard]] Result<const ResolvedKernelFunction*> packed_function(
      QwenBf16PackedPrimitive primitive) const;
  [[nodiscard]] std::span<const ResolvedKernelFunction> functions()
      const noexcept {
    return module_.functions();
  }
  [[nodiscard]] std::span<const ResolvedKernelFunction> legacy_functions()
      const noexcept {
    return module_.functions().first(kExecutionPrimitiveCount);
  }
  [[nodiscard]] std::span<const ResolvedKernelFunction> packed_functions()
      const noexcept {
    return module_.functions().subspan(kPrimitiveCount,
                                       kPackedPrimitiveCount);
  }
  [[nodiscard]] std::uint32_t target_sm() const noexcept { return target_sm_; }

 private:
  QwenBf16KernelBundle(
      std::uint32_t target_sm,
      std::vector<KernelSignatureManifest> manifests,
      VerifiedKernelModule module)
      : target_sm_(target_sm),
        manifests_(std::move(manifests)),
        module_(std::move(module)) {}

  std::uint32_t target_sm_ = 0;
  std::vector<KernelSignatureManifest> manifests_;
  VerifiedKernelModule module_;
};

}  // namespace pih
