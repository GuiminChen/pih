#pragma once

#include <cstdint>

#include "pih/model/qwen3_bf16_command_buffer.h"
#include "pih/model/qwen3_bf16_resource_set.h"
#include "pih/model/qwen3_int4_weight_resource_set.h"

namespace pih {

class QwenTeacherForcedLogitsPlan;

class QwenInt4LmHeadBinding final {
 public:
  [[nodiscard]] const TensorView& input() const noexcept { return input_; }
  [[nodiscard]] const TensorView& weight() const noexcept { return weight_; }
  [[nodiscard]] const TensorView& output() const noexcept { return output_; }
 private:
  friend class QwenInt4LmHeadMaterializer;
  friend class QwenInt4PackedLmHeadMaterializer;
  friend class QwenTeacherForcedLogitsPlan;
  QwenInt4LmHeadBinding(TensorView input,TensorView weight,TensorView output)
      : input_(input),weight_(weight),output_(output) {}
  TensorView input_,weight_,output_;
};

class QwenInt4LmHeadMaterializer final {
 public:
  static Result<QwenInt4LmHeadBinding> Create(
      const QwenBf16PreparedCommand& command,
      const QwenBf16ResourceSet& activations,
      const QwenInt4WeightResourceSet& weights,
      std::uint64_t request_generation);
};

}  // namespace pih
