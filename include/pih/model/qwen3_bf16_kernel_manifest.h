#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "pih/backend/cuda/kernel_signature_manifest.h"

namespace pih {

enum class QwenBf16Primitive : std::uint8_t {
  kEmbedding = 0,
  kResidualAdd,
  kSiluMul,
  kRmsNorm,
  kRope,
  kKvAppend,
  kPagedGqa,
  kRopeAngles,
  kGreedyArgmax,
  kTeacherForcedMetric,
};

Result<std::string_view> qwen_bf16_kernel_symbol(QwenBf16Primitive primitive);
Result<std::string_view> qwen_bf16_parameter_abi_descriptor(
    QwenBf16Primitive primitive);
Result<KernelSignatureManifest> qwen_bf16_kernel_manifest(
    QwenBf16Primitive primitive, std::string selected_cubin_sha256);

}  // namespace pih
