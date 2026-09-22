#include "pih/model/qwen3_bf16_packed_linear_materializer.h"

#include <array>

namespace pih {
namespace {

bool exact_matrix(const TensorView& view, DType dtype, std::uint64_t rows,
                  std::uint64_t columns, std::int32_t rank) {
  return view.dtype() == dtype && view.device().type() == DeviceType::kCuda &&
         view.device().index() == rank && view.generation() != 0 &&
         view.rank() == 2 && view.dim(0) == rows && view.dim(1) == columns &&
         view.stride(0) == columns && view.stride(1) == 1;
}

Result<TensorView> leading_rows(const TensorView& source,
                                std::uint32_t rows) {
  if (source.dtype() != DType::kBFloat16 || source.rank() != 2 || rows == 0 ||
      rows > source.dim(0) || source.dim(1) != 1024 ||
      source.stride(0) != 1024 || source.stride(1) != 1) {
    return Status::InvalidArgument("packed LM head source is invalid");
  }
  const std::array<std::int64_t, 2> shape{
      static_cast<std::int64_t>(rows), 1024};
  return TensorView::Create(source.data(), DType::kBFloat16, shape, {},
                            source.device(), source.generation());
}

}  // namespace

Result<QwenBf16LinearBinding> QwenBf16PackedLinearMaterializer::Create(
    const QwenBf16PreparedCommand& command,
    const QwenBf16PackedResourceSet& resources,
    const QwenBf16WeightResourceSet& weights,
    std::uint64_t request_generation) {
  if (command.linear_kind != QwenBf16LinearKind::kLmHead) {
    return QwenBf16LinearMaterializer::Create(
        command, resources.activations(), weights, request_generation);
  }
  if (command.backend != QwenBf16CommandBackend::kLinear ||
      command.execution_step.operation != QwenBf16ExecutionOp::kLmHead ||
      !command.has_weight() || request_generation == 0 ||
      request_generation != resources.activations().request_generation() ||
      resources.activations().owning_rank() != weights.owning_rank()) {
    return Status::InvalidArgument("packed LM head identity is invalid");
  }
  auto hidden = resources.activations().view(QwenBf16ActivationSlot::kHidden);
  auto logits = resources.activations().view(QwenBf16ActivationSlot::kLogits);
  auto weight = weights.view(command.tensor_index);
  if (!hidden.ok()) return hidden.status();
  if (!logits.ok()) return logits.status();
  if (!weight.ok()) return weight.status();
  auto input = leading_rows(*hidden, resources.sample_count());
  if (!input.ok()) return input.status();
  const auto rows = resources.sample_count();
  const auto rank = resources.activations().owning_rank();
  if (!exact_matrix(*input, DType::kBFloat16, rows, 1024, rank) ||
      !exact_matrix(*weight, DType::kBFloat16, 151936, 1024, rank) ||
      !exact_matrix(*logits, DType::kFloat32, rows, 151936, rank)) {
    return Status::InvalidArgument("packed LM head differs from frozen GEMM shape");
  }
  return QwenBf16LinearBinding(QwenBf16LinearKind::kLmHead, *input, *weight,
                               *logits);
}

}  // namespace pih
