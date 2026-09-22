#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "pih/backend/cuda/verified_kernel_module.h"
#include "pih/model/qwen3_bf16_kernel_manifest.h"
#include "pih/model/qwen3_int4_gemm_plan.h"

namespace pih {

class QwenInt4KernelBundle final {
 public:
  static constexpr std::size_t kBf16PrimitiveCount = 9;
  static constexpr std::size_t kFunctionCount = 10;
  static Result<QwenInt4KernelBundle> Load(
      KernelModuleDriver& driver, const std::filesystem::path& manifest_path,
      const std::filesystem::path& cubin_path,
      std::uint32_t compute_major, std::uint32_t compute_minor,
      std::uint64_t maximum_cubin_bytes);

  QwenInt4KernelBundle(const QwenInt4KernelBundle&) = delete;
  QwenInt4KernelBundle& operator=(const QwenInt4KernelBundle&) = delete;
  QwenInt4KernelBundle(QwenInt4KernelBundle&&) noexcept = default;
  QwenInt4KernelBundle& operator=(QwenInt4KernelBundle&&) noexcept = default;

  [[nodiscard]] const ResolvedKernelFunction& compatibility_function()
      const noexcept { return module_.functions().back(); }
  [[nodiscard]] const KernelSignatureManifest& compatibility_manifest()
      const noexcept { return manifest_; }
  [[nodiscard]] Result<const ResolvedKernelFunction*> bf16_function(
      QwenBf16Primitive primitive) const;
  [[nodiscard]] std::span<const ResolvedKernelFunction> functions()
      const noexcept { return module_.functions(); }
  [[nodiscard]] std::uint32_t target_sm() const noexcept { return target_sm_; }

 private:
  QwenInt4KernelBundle(std::uint32_t target_sm,
                       std::vector<KernelSignatureManifest> bf16_manifests,
                       KernelSignatureManifest manifest,
                       VerifiedKernelModule module)
      : target_sm_(target_sm),
        bf16_manifests_(std::move(bf16_manifests)),
        manifest_(std::move(manifest)),
        module_(std::move(module)) {}
  std::uint32_t target_sm_;
  std::vector<KernelSignatureManifest> bf16_manifests_;
  KernelSignatureManifest manifest_;
  VerifiedKernelModule module_;
};

}  // namespace pih
