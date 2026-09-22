#pragma once

#include <cstdint>

#include "pih/model/qwen3_bf16_command_buffer.h"
#include "pih/model/qwen3_bf16_resource_set.h"

namespace pih {

class QwenBf16PackedLinearMaterializer;
class QwenTeacherForcedLogitsPlan;

class QwenBf16LinearBinding final {
 public:
  [[nodiscard]] QwenBf16LinearKind kind() const noexcept { return kind_; }
  [[nodiscard]] const TensorView& input() const noexcept { return input_; }
  [[nodiscard]] const TensorView& weight() const noexcept { return weight_; }
  [[nodiscard]] const TensorView& output() const noexcept { return output_; }

 private:
  friend class QwenBf16LinearMaterializer;
  friend class QwenBf16PackedLinearMaterializer;
  friend class QwenTeacherForcedLogitsPlan;
  QwenBf16LinearBinding(QwenBf16LinearKind kind, TensorView input,
                        TensorView weight, TensorView output)
      : kind_(kind),
        input_(input),
        weight_(weight),
        output_(output) {}

  QwenBf16LinearKind kind_;
  TensorView input_;
  TensorView weight_;
  TensorView output_;
};

class QwenBf16LinearMaterializer final {
 public:
  static Result<QwenBf16LinearBinding> Create(
      const QwenBf16PreparedCommand& command,
      const QwenBf16ResourceSet& resources,
      const QwenBf16WeightResourceSet& weights,
      std::uint64_t request_generation);
};

}  // namespace pih
