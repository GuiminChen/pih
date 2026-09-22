#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "pih/backend/cuda/kernel_pointer_contract.h"
#include "pih/backend/cuda/kernel_signature_manifest.h"
#include "pih/core/result.h"
#include "pih/core/sha256.h"
#include "pih/model/qwen3_int4_linear_shape_ledger.h"

namespace pih {

enum class QwenInt4KernelVariant : std::uint8_t {
  kCompatibilityCanonical = 1,
};

class QwenInt4GemmPlan final {
 public:
  static constexpr std::uint64_t kMaximumTokens = 4096;
  static constexpr std::uint32_t kThreadsPerBlock = 256;

  static Result<QwenInt4GemmPlan> Create(
      QwenInt4LinearShapeFamily family, std::uint64_t tokens,
      QwenInt4KernelVariant variant =
          QwenInt4KernelVariant::kCompatibilityCanonical);

  [[nodiscard]] QwenInt4LinearShapeFamily family() const noexcept { return family_; }
  [[nodiscard]] std::uint64_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::uint64_t output_features() const noexcept { return output_features_; }
  [[nodiscard]] std::uint64_t input_features() const noexcept { return input_features_; }
  [[nodiscard]] std::uint64_t input_bytes() const noexcept { return input_bytes_; }
  [[nodiscard]] std::uint64_t packed_bytes() const noexcept { return packed_bytes_; }
  [[nodiscard]] std::uint64_t scale_bytes() const noexcept { return scale_bytes_; }
  [[nodiscard]] std::uint64_t output_bytes() const noexcept { return output_bytes_; }
  [[nodiscard]] std::uint32_t blocks() const noexcept { return blocks_; }
  [[nodiscard]] std::string_view logical_id() const noexcept {
    return "qwen.linear.w4a16.compatibility.v1";
  }
  [[nodiscard]] std::string_view kernel_symbol() const noexcept {
    return "pih_qwen_w4a16_gemm_compat_v1";
  }
  [[nodiscard]] Result<Sha256Digest> semantic_digest() const;
  Result<KernelSignatureManifest> signature(
      std::string selected_cubin_sha256) const;
  Result<std::vector<KernelPointerContract>> pointer_contracts(
      std::int32_t rank, std::int32_t device_index) const;

 private:
  QwenInt4GemmPlan(QwenInt4LinearShapeFamily family, std::uint64_t rows,
                   std::uint64_t output_features,
                   std::uint64_t input_features, std::uint64_t input_bytes,
                   std::uint64_t packed_bytes, std::uint64_t scale_bytes,
                   std::uint64_t output_bytes, std::uint32_t blocks)
      : family_(family), rows_(rows), output_features_(output_features),
        input_features_(input_features), input_bytes_(input_bytes),
        packed_bytes_(packed_bytes), scale_bytes_(scale_bytes),
        output_bytes_(output_bytes), blocks_(blocks) {}
  QwenInt4LinearShapeFamily family_;
  std::uint64_t rows_, output_features_, input_features_;
  std::uint64_t input_bytes_, packed_bytes_, scale_bytes_, output_bytes_;
  std::uint32_t blocks_;
};

}  // namespace pih
