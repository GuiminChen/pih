#include "pih/model/qwen3_int4_packed_lm_head_materializer.h"

#include <array>

#include "pih/model/qwen3_int4_bf16_weight_resolver.h"

namespace pih {
namespace {

bool exact_matrix(const TensorView& view, DType dtype, std::uint64_t rows,
                  std::uint64_t columns, std::int32_t device) {
  return view.dtype() == dtype && view.device().type() == DeviceType::kCuda &&
         view.device().index() == device && view.generation() != 0 &&
         view.rank() == 2 && view.dim(0) == rows &&
         view.dim(1) == columns && view.stride(0) == columns &&
         view.stride(1) == 1;
}

Result<TensorView> leading_rows(const TensorView& source,
                                std::uint32_t rows) {
  if (source.dtype() != DType::kBFloat16 || source.rank() != 2 || rows == 0 ||
      rows > source.dim(0) || source.dim(1) != 1024 ||
      source.stride(0) != 1024 || source.stride(1) != 1) {
    return Status::InvalidArgument("packed INT4 LM head source is invalid");
  }
  const std::array<std::int64_t,2> shape{
      static_cast<std::int64_t>(rows),1024};
  return TensorView::Create(source.data(),DType::kBFloat16,shape,{},
                            source.device(),source.generation());
}

}  // namespace

Result<QwenInt4LmHeadBinding> QwenInt4PackedLmHeadMaterializer::Create(
    const QwenBf16PreparedCommand& command,
    const QwenBf16PackedResourceSet& resources,
    const QwenInt4WeightResourceSet& weights,
    std::uint64_t request_generation) {
  const auto& activations=resources.activations();
  if (command.backend!=QwenBf16CommandBackend::kLinear ||
      command.linear_kind!=QwenBf16LinearKind::kLmHead ||
      command.execution_step.operation!=QwenBf16ExecutionOp::kLmHead ||
      request_generation==0 ||
      request_generation!=activations.request_generation() ||
      activations.owning_rank()!=weights.owning_rank() ||
      resources.sample_count()==0 ||
      resources.sample_count()>resources.sequence_count()) {
    return Status::InvalidArgument("packed INT4 LM head identity is invalid");
  }
  auto hidden=activations.view(QwenBf16ActivationSlot::kHidden);
  auto output=activations.view(QwenBf16ActivationSlot::kLogits);
  auto weight=QwenInt4Bf16WeightResolver::Resolve(command,weights);
  if(!hidden.ok())return hidden.status();
  if(!output.ok())return output.status();
  if(!weight.ok())return weight.status();
  auto input=leading_rows(*hidden,resources.sample_count());
  if(!input.ok())return input.status();
  const auto rows=resources.sample_count();
  const auto device=hidden->device().index();
  if(!exact_matrix(*input,DType::kBFloat16,rows,1024,device) ||
     !exact_matrix(*weight,DType::kBFloat16,151936,1024,
                   weights.device_index()) ||
     !exact_matrix(*output,DType::kFloat32,rows,151936,device)) {
    return Status::InvalidArgument(
        "packed INT4 LM head tensor contract mismatch");
  }
  return QwenInt4LmHeadBinding(*input,*weight,*output);
}

}  // namespace pih
