#include "pih/model/qwen3_int4_linear_materializer.h"

#include <array>

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_manifest.h"

namespace pih {
namespace {

using Slot=QwenBf16ActivationSlot;

Result<QwenInt4LinearShapeFamily> family(QwenBf16LinearKind kind) {
  switch(kind) {
    case QwenBf16LinearKind::kQuery:return QwenInt4LinearShapeFamily::kQProj;
    case QwenBf16LinearKind::kKey:
    case QwenBf16LinearKind::kValue:return QwenInt4LinearShapeFamily::kKvProj;
    case QwenBf16LinearKind::kAttentionOutput:return QwenInt4LinearShapeFamily::kOProj;
    case QwenBf16LinearKind::kGate:
    case QwenBf16LinearKind::kUp:return QwenInt4LinearShapeFamily::kGateUpProj;
    case QwenBf16LinearKind::kDown:return QwenInt4LinearShapeFamily::kDownProj;
    case QwenBf16LinearKind::kLmHead:break;
  }
  return Status::InvalidArgument("Qwen LM head is not an INT4 Linear");
}

Result<TensorView> matrix(const TensorView& source,std::uint64_t rows,
                          std::uint64_t columns) {
  auto elements=checked_mul_u64(rows,columns);
  if(!elements.ok())return elements.status();
  if(source.dtype()!=DType::kBFloat16 || source.num_elements()!=*elements)
    return Status::InvalidArgument("Qwen INT4 activation extent is invalid");
  std::uint64_t stride=1;
  for(std::size_t reverse=source.rank();reverse>0;--reverse) {
    const auto axis=reverse-1;
    if(source.stride(axis)!=stride)
      return Status::InvalidArgument("Qwen INT4 activation is not contiguous");
    auto next=checked_mul_u64(stride,source.dim(axis));
    if(!next.ok())return next.status(); stride=*next;
  }
  const std::array<std::int64_t,2> shape{
      static_cast<std::int64_t>(rows),static_cast<std::int64_t>(columns)};
  return TensorView::Create(source.data(),source.dtype(),shape,{},
                            source.device(),source.generation());
}

}  // namespace

Result<QwenInt4DispatchPlan> QwenInt4LinearMaterializer::Create(
    const QwenBf16PreparedCommand& command,
    const ResolvedKernelFunction& function,
    const QwenBf16ResourceSet& activations,
    const QwenInt4WeightResourceSet& weights,
    std::uint64_t request_generation) {
  if(command.backend!=QwenBf16CommandBackend::kLinear || !command.has_weight() ||
     request_generation==0 || request_generation!=activations.request_generation() ||
     activations.owning_rank()!=weights.owning_rank() ||
     command.tensor_index>=Qwen3Manifest::expected_tensors().size())
    return Status::InvalidArgument("Qwen INT4 materialization identity is invalid");
  auto selected_family=family(command.linear_kind);
  if(!selected_family.ok())return selected_family.status();
  const auto& source_name=Qwen3Manifest::expected_tensors()[command.tensor_index].name;
  auto weight=weights.linear(source_name); if(!weight.ok())return weight.status();
  if(weight->family!=*selected_family)
    return Status::InvalidArgument("Qwen INT4 Linear family drifted");
  auto hidden=activations.view(Slot::kHidden); if(!hidden.ok())return hidden.status();
  if(hidden->rank()!=2 || hidden->dim(0)==0)
    return Status::InvalidArgument("Qwen INT4 packed token matrix is invalid");
  const auto rows=hidden->dim(0);
  Result<TensorView> input=Status::InvalidArgument("unbound INT4 input");
  Result<TensorView> output=Status::InvalidArgument("unbound INT4 output");
  switch(command.linear_kind) {
    case QwenBf16LinearKind::kQuery: input=activations.view(Slot::kNormalized); output=activations.view(Slot::kQuery); break;
    case QwenBf16LinearKind::kKey: input=activations.view(Slot::kNormalized); output=activations.view(Slot::kKey); break;
    case QwenBf16LinearKind::kValue: input=activations.view(Slot::kNormalized); output=activations.view(Slot::kValue); break;
    case QwenBf16LinearKind::kAttentionOutput: input=activations.view(Slot::kAttention); output=activations.view(Slot::kNormalized); break;
    case QwenBf16LinearKind::kGate: input=activations.view(Slot::kNormalized); output=activations.view(Slot::kGate); break;
    case QwenBf16LinearKind::kUp: input=activations.view(Slot::kNormalized); output=activations.view(Slot::kUp); break;
    case QwenBf16LinearKind::kDown: input=activations.view(Slot::kGate); output=activations.view(Slot::kNormalized); break;
    case QwenBf16LinearKind::kLmHead:return Status::InvalidArgument("Qwen LM head is not INT4");
  }
  if(!input.ok())return input.status(); if(!output.ok())return output.status();
  auto gemm=QwenInt4GemmPlan::Create(*selected_family,rows);
  if(!gemm.ok())return gemm.status();
  input=matrix(*input,rows,gemm->input_features());
  output=matrix(*output,rows,gemm->output_features());
  if(!input.ok())return input.status(); if(!output.ok())return output.status();
  auto error=activations.view(Slot::kDeviceError); if(!error.ok())return error.status();
  return QwenInt4DispatchPlan::Create(
      *gemm,function,*input,weight->packed,weight->scales,*output,*error,
      activations.owning_rank(),request_generation,
      weight->packed.generation());
}

}  // namespace pih
