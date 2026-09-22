#include "pih/model/qwen3_bf16_linear_materializer.h"

#include <array>
#include <cstdint>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

using Slot = QwenBf16ActivationSlot;

bool exact_matrix(const TensorView& view, DType dtype, std::uint64_t rows,
                  std::uint64_t columns, std::int32_t rank) {
  return view.dtype() == dtype && view.device().type() == DeviceType::kCuda &&
         view.device().index() == rank && view.generation() != 0 &&
         view.rank() == 2 && view.dim(0) == rows && view.dim(1) == columns &&
         view.stride(0) == columns && view.stride(1) == 1;
}

Result<TensorView> last_row(const TensorView& matrix) {
  if (matrix.dtype() != DType::kBFloat16 || matrix.rank() != 2 ||
      matrix.dim(0) == 0 || matrix.dim(1) != 1024 ||
      matrix.stride(0) != 1024 || matrix.stride(1) != 1) {
    return Status::InvalidArgument("Qwen LM head source matrix is invalid");
  }
  auto row_elements = checked_mul_u64(matrix.dim(0) - 1, matrix.dim(1));
  if (!row_elements.ok()) return row_elements.status();
  auto row_bytes = checked_mul_u64(*row_elements, UINT64_C(2));
  if (!row_bytes.ok()) return row_bytes.status();
  const auto base = reinterpret_cast<std::uintptr_t>(matrix.data());
  if (*row_bytes > UINTPTR_MAX - base) {
    return Status::ResourceExhausted("Qwen LM head row address overflows");
  }
  const std::array<std::int64_t, 2> shape{1, 1024};
  return TensorView::Create(reinterpret_cast<void*>(base + *row_bytes),
                            DType::kBFloat16, shape, {}, matrix.device(),
                            matrix.generation());
}

Result<TensorView> matrix_view(const TensorView& source, DType dtype,
                               std::uint64_t rows,
                               std::uint64_t columns) {
  auto elements = checked_mul_u64(rows, columns);
  if (!elements.ok()) return elements.status();
  if (source.dtype() != dtype || source.num_elements() != *elements) {
    return Status::InvalidArgument("Qwen linear tensor extent is invalid");
  }
  std::uint64_t stride = 1;
  for (std::size_t reverse = source.rank(); reverse > 0; --reverse) {
    const auto axis = reverse - 1;
    if (source.stride(axis) != stride) {
      return Status::InvalidArgument("Qwen linear tensor is not contiguous");
    }
    auto next = checked_mul_u64(stride, source.dim(axis));
    if (!next.ok()) return next.status();
    stride = *next;
  }
  if (rows > static_cast<std::uint64_t>(INT64_MAX) ||
      columns > static_cast<std::uint64_t>(INT64_MAX)) {
    return Status::ResourceExhausted("Qwen linear matrix shape overflows i64");
  }
  const std::array<std::int64_t, 2> shape{
      static_cast<std::int64_t>(rows), static_cast<std::int64_t>(columns)};
  return TensorView::Create(source.data(), dtype, shape, {}, source.device(),
                            source.generation());
}

}  // namespace

Result<QwenBf16LinearBinding> QwenBf16LinearMaterializer::Create(
    const QwenBf16PreparedCommand& command,
    const QwenBf16ResourceSet& resources,
    const QwenBf16WeightResourceSet& weights,
    std::uint64_t request_generation) {
  if (command.backend != QwenBf16CommandBackend::kLinear ||
      !command.has_weight() || request_generation == 0 ||
      request_generation != resources.request_generation() ||
      resources.owning_rank() != weights.owning_rank()) {
    return Status::InvalidArgument(
        "Qwen linear materialization identity is invalid");
  }
  auto hidden = resources.view(Slot::kHidden);
  auto weight = weights.view(command.tensor_index);
  if (!hidden.ok()) return hidden.status();
  if (!weight.ok()) return weight.status();
  if (hidden->rank() != 2 || hidden->dim(0) == 0) {
    return Status::InvalidArgument("Qwen linear packed token matrix is invalid");
  }
  const std::uint64_t packed_tokens = hidden->dim(0);
  auto rows = qwen_bf16_linear_execution_rows(command.linear_kind,
                                               packed_tokens, 1);
  if (!rows.ok()) return rows.status();
  auto shape = QwenBf16LinearShape::Create(command.linear_kind, *rows);
  if (!shape.ok()) return shape.status();

  Result<TensorView> input = Status::InvalidArgument("unbound linear input");
  Result<TensorView> output = Status::InvalidArgument("unbound linear output");
  switch (command.linear_kind) {
    case QwenBf16LinearKind::kQuery:
      input = resources.view(Slot::kNormalized);
      output = resources.view(Slot::kQuery);
      break;
    case QwenBf16LinearKind::kKey:
      input = resources.view(Slot::kNormalized);
      output = resources.view(Slot::kKey);
      break;
    case QwenBf16LinearKind::kValue:
      input = resources.view(Slot::kNormalized);
      output = resources.view(Slot::kValue);
      break;
    case QwenBf16LinearKind::kAttentionOutput:
      input = resources.view(Slot::kAttention);
      output = resources.view(Slot::kNormalized);
      break;
    case QwenBf16LinearKind::kGate:
      input = resources.view(Slot::kNormalized);
      output = resources.view(Slot::kGate);
      break;
    case QwenBf16LinearKind::kUp:
      input = resources.view(Slot::kNormalized);
      output = resources.view(Slot::kUp);
      break;
    case QwenBf16LinearKind::kDown:
      input = resources.view(Slot::kGate);
      output = resources.view(Slot::kNormalized);
      break;
    case QwenBf16LinearKind::kLmHead: {
      auto normalized = resources.view(Slot::kNormalized);
      if (!normalized.ok()) return normalized.status();
      input = last_row(*normalized);
      output = resources.view(Slot::kLogits);
      break;
    }
  }
  if (!input.ok()) return input.status();
  if (!output.ok()) return output.status();
  const auto rank = resources.owning_rank();
  input = matrix_view(*input, DType::kBFloat16, *rows,
                      shape->input_features());
  output = matrix_view(*output, shape->output_dtype(), *rows,
                       shape->output_features());
  if (!input.ok()) return input.status();
  if (!output.ok()) return output.status();
  if (!exact_matrix(*input, DType::kBFloat16, *rows,
                    shape->input_features(), rank) ||
      !exact_matrix(*weight, DType::kBFloat16, shape->weight_rows(),
                    shape->weight_columns(), rank) ||
      !exact_matrix(*output, shape->output_dtype(), *rows,
                    shape->output_features(), rank)) {
    return Status::InvalidArgument(
        "Qwen linear binding differs from the frozen GEMM shape");
  }
  return QwenBf16LinearBinding(command.linear_kind, *input, *weight, *output);
}

}  // namespace pih
