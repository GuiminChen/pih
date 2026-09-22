#include "pih/model/qwen3_int4_bf16_weight_resolver.h"

#include "pih/model/qwen3_manifest.h"

namespace pih {

Result<TensorView> QwenInt4Bf16WeightResolver::Resolve(
    const QwenBf16PreparedCommand& command,
    const QwenInt4WeightResourceSet& weights) {
  if (!command.has_weight() ||
      command.tensor_index >= Qwen3Manifest::expected_tensors().size()) {
    return Status::InvalidArgument(
        "Qwen retained BF16 command has no valid tensor identity");
  }
  const bool retained_kernel =
      command.backend == QwenBf16CommandBackend::kKernel;
  const bool retained_head =
      command.backend == QwenBf16CommandBackend::kLinear &&
      command.linear_kind == QwenBf16LinearKind::kLmHead &&
      command.execution_step.operation == QwenBf16ExecutionOp::kLmHead;
  if (!retained_kernel && !retained_head) {
    return Status::InvalidArgument(
        "Qwen quantized Linear cannot resolve as retained BF16");
  }
  const auto& identity =
      Qwen3Manifest::expected_tensors()[command.tensor_index].name;
  auto view = weights.view(identity);
  if (!view.ok()) return view.status();
  if (view->dtype() != DType::kBFloat16 ||
      view->device().type() != DeviceType::kCuda ||
      view->device().index() != weights.device_index() ||
      view->generation() == 0) {
    return Status::InvalidArgument(
        "Qwen retained BF16 weight view is invalid");
  }
  return view;
}

}  // namespace pih
