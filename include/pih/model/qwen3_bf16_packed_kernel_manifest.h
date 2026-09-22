#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "pih/backend/cuda/kernel_signature_manifest.h"

namespace pih {

enum class QwenBf16PackedPrimitive : std::uint8_t {
  kEmbedding = 1,
  kRopeAngles = 2,
  kKvAppend = 3,
  kPagedGqa = 4,
  kSampleHidden = 5,
  kGreedyArgmax = 6,
  kSampler = 7,
};

Result<std::string_view> qwen_bf16_packed_kernel_symbol(
    QwenBf16PackedPrimitive primitive);
Result<std::string_view> qwen_bf16_packed_parameter_abi_descriptor(
    QwenBf16PackedPrimitive primitive);
Result<KernelSignatureManifest> qwen_bf16_packed_kernel_manifest(
    QwenBf16PackedPrimitive primitive, std::string selected_cubin_sha256);

}  // namespace pih
