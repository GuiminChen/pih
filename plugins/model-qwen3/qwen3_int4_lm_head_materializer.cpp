#include "pih/model/qwen3_int4_lm_head_materializer.h"

#include <array>
#include <limits>

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_int4_bf16_weight_resolver.h"

namespace pih {
namespace {

bool exact_matrix(const TensorView& view,DType dtype,std::uint64_t rows,
                  std::uint64_t columns,std::int32_t device) {
  return view.dtype()==dtype && view.device().type()==DeviceType::kCuda &&
         view.device().index()==device && view.generation()!=0 &&
         view.rank()==2 && view.dim(0)==rows && view.dim(1)==columns &&
         view.stride(0)==columns && view.stride(1)==1;
}

Result<TensorView> last_row(const TensorView& matrix) {
  if(matrix.dtype()!=DType::kBFloat16 || matrix.rank()!=2 ||
     matrix.dim(0)==0 || matrix.dim(1)!=1024 ||
     matrix.stride(0)!=1024 || matrix.stride(1)!=1)
    return Status::InvalidArgument("Qwen INT4 LM head input is invalid");
  auto elements=checked_mul_u64(matrix.dim(0)-1,matrix.dim(1));
  if(!elements.ok())return elements.status();
  auto bytes=checked_mul_u64(*elements,2); if(!bytes.ok())return bytes.status();
  const auto base=reinterpret_cast<std::uintptr_t>(matrix.data());
  if(*bytes>UINTPTR_MAX-base)
    return Status::ResourceExhausted("Qwen INT4 LM head address overflows");
  const std::array<std::int64_t,2> shape{1,1024};
  return TensorView::Create(reinterpret_cast<void*>(base+*bytes),
      DType::kBFloat16,shape,{},matrix.device(),matrix.generation());
}

}  // namespace

Result<QwenInt4LmHeadBinding> QwenInt4LmHeadMaterializer::Create(
    const QwenBf16PreparedCommand& command,
    const QwenBf16ResourceSet& activations,
    const QwenInt4WeightResourceSet& weights,
    std::uint64_t request_generation) {
  if(command.backend!=QwenBf16CommandBackend::kLinear ||
     command.linear_kind!=QwenBf16LinearKind::kLmHead ||
     command.execution_step.operation!=QwenBf16ExecutionOp::kLmHead ||
     request_generation==0 || request_generation!=activations.request_generation() ||
     activations.owning_rank()!=weights.owning_rank())
    return Status::InvalidArgument("Qwen INT4 LM head identity is invalid");
  auto normalized=activations.view(QwenBf16ActivationSlot::kNormalized);
  auto output=activations.view(QwenBf16ActivationSlot::kLogits);
  auto weight=QwenInt4Bf16WeightResolver::Resolve(command,weights);
  if(!normalized.ok())return normalized.status();
  if(!output.ok())return output.status();
  if(!weight.ok())return weight.status();
  auto input=last_row(*normalized); if(!input.ok())return input.status();
  const auto device=activations.view(QwenBf16ActivationSlot::kHidden);
  if(!device.ok())return device.status();
  const auto device_index=device->device().index();
  if(!exact_matrix(*input,DType::kBFloat16,1,1024,device_index) ||
     !exact_matrix(*weight,DType::kBFloat16,151936,1024,
                   weights.device_index()) ||
     !exact_matrix(*output,DType::kFloat32,1,151936,device_index))
    return Status::InvalidArgument("Qwen INT4 LM head tensor contract mismatch");
  return QwenInt4LmHeadBinding(*input,*weight,*output);
}

}  // namespace pih
