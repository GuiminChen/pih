#pragma once

#include <cstdint>
#include <string_view>

#include "pih/core/result.h"
#include "pih/core/dtype.h"

namespace pih {

enum class QwenBf16LinearKind : std::uint8_t {
  kQuery = 0,
  kKey,
  kValue,
  kAttentionOutput,
  kGate,
  kUp,
  kDown,
  kLmHead,
};

Result<std::uint64_t> qwen_bf16_linear_execution_rows(
    QwenBf16LinearKind kind, std::uint64_t packed_tokens,
    std::uint64_t active_logit_rows);

class QwenBf16LinearShape final {
 public:
  static constexpr std::uint64_t kMaximumTokensPerPlan = 4096;

  static Result<QwenBf16LinearShape> Create(QwenBf16LinearKind kind,
                                            std::uint64_t tokens);

  [[nodiscard]] QwenBf16LinearKind kind() const noexcept { return kind_; }
  [[nodiscard]] std::string_view logical_id() const noexcept {
    return logical_id_;
  }
  [[nodiscard]] std::uint64_t tokens() const noexcept { return tokens_; }
  [[nodiscard]] std::uint64_t input_features() const noexcept {
    return input_features_;
  }
  [[nodiscard]] std::uint64_t output_features() const noexcept {
    return output_features_;
  }
  [[nodiscard]] DType output_dtype() const noexcept {
    return kind_ == QwenBf16LinearKind::kLmHead ? DType::kFloat32
                                                : DType::kBFloat16;
  }
  [[nodiscard]] std::uint64_t weight_rows() const noexcept {
    return output_features_;
  }
  [[nodiscard]] std::uint64_t weight_columns() const noexcept {
    return input_features_;
  }

 private:
  QwenBf16LinearShape(QwenBf16LinearKind kind, std::string_view logical_id,
                      std::uint64_t tokens, std::uint64_t input_features,
                      std::uint64_t output_features)
      : kind_(kind),
        logical_id_(logical_id),
        tokens_(tokens),
        input_features_(input_features),
        output_features_(output_features) {}

  QwenBf16LinearKind kind_;
  std::string_view logical_id_;
  std::uint64_t tokens_;
  std::uint64_t input_features_;
  std::uint64_t output_features_;
};

}  // namespace pih
