#include "pih/model/qwen3_int4_dispatch_plan.h"

#include <array>

namespace pih {
namespace {

bool exact_matrix(const TensorView& view, DType dtype, std::uint64_t rows,
                  std::uint64_t columns, std::int32_t device_index) {
  return view.generation() != 0 &&
         view.dtype() == dtype && view.device().type() == DeviceType::kCuda &&
         view.device().index() == device_index && view.rank() == 2 &&
         view.dim(0) == rows && view.dim(1) == columns &&
         view.stride(0) == columns && view.stride(1) == 1;
}

bool exact_error(const TensorView& view, std::int32_t device_index,
                 std::uint64_t generation) {
  return generation != 0 && view.generation() == generation &&
         view.dtype() == DType::kUInt8 &&
         view.device().type() == DeviceType::kCuda &&
         view.device().index() == device_index && view.rank() == 1 &&
         view.dim(0) == 4 && view.stride(0) == 1;
}

}  // namespace

Result<QwenInt4DispatchPlan> QwenInt4DispatchPlan::Create(
    const QwenInt4GemmPlan& gemm,
    const ResolvedKernelFunction& function,
    const TensorView& input, const TensorView& packed_weight,
    const TensorView& scales, const TensorView& output,
    const TensorView& error_flag, std::int32_t owning_rank,
    std::uint64_t activation_generation,
    std::uint64_t weight_generation) {
  if (owning_rank < 0 || activation_generation == 0 ||
      weight_generation == 0) {
    return Status::InvalidArgument("Qwen INT4 dispatch identity is invalid");
  }
  const auto device_index = input.device().index();
  if (device_index < 0) {
    return Status::InvalidArgument("Qwen INT4 local device is invalid");
  }
  auto signature = gemm.signature(function.selected_cubin_sha256);
  if (!signature.ok()) return signature.status();
  if (function.symbol != gemm.kernel_symbol() ||
      function.logical_id != signature->logical_id() ||
      function.parameter_abi_sha256 != signature->parameter_abi_sha256() ||
      function.handle == 0) {
    return Status::InvalidArgument("Qwen INT4 resolved function drifted");
  }
  if (!exact_matrix(input, DType::kBFloat16, gemm.rows(),
                    gemm.input_features(), device_index) ||
      !exact_matrix(packed_weight, DType::kUInt8, gemm.output_features(),
                    gemm.input_features() / 2, device_index) ||
      packed_weight.generation() != weight_generation ||
      !exact_matrix(scales, DType::kFloat16, gemm.output_features(),
                    gemm.input_features() / 128, device_index) ||
      scales.generation() != weight_generation ||
      !exact_matrix(output, DType::kBFloat16, gemm.rows(),
                    gemm.output_features(), device_index) ||
      !exact_error(error_flag, device_index, activation_generation)) {
    return Status::InvalidArgument(
        "Qwen INT4 dispatch tensor contract mismatch");
  }
  auto contracts = gemm.pointer_contracts(owning_rank, device_index);
  if (!contracts.ok()) return contracts.status();
  auto packet = KernelArgumentPacket::Create(*signature, *contracts);
  if (!packet.ok()) return packet.status();
  const std::array<const TensorView*,5> views{
      &input, &packed_weight, &scales, &output, &error_flag};
  const std::array<KernelPointerOwnerClass,5> owners{
      KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kWeight,
      KernelPointerOwnerClass::kWeight,
      KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kWorkspace};
  const std::array<std::uint64_t,5> generations{
      input.generation(), weight_generation, weight_generation,
      output.generation(), activation_generation};
  for (std::size_t ordinal = 0; ordinal < views.size(); ++ordinal) {
    auto pointer = VerifiedDevicePointer::Create(
        *views[ordinal], (*contracts)[ordinal], owners[ordinal], owning_rank,
        generations[ordinal]);
    if (!pointer.ok()) return pointer.status();
    auto status = packet->set_device_pointer(ordinal, *pointer);
    if (!status.ok()) return status;
  }
  for (const auto [ordinal, value] :
       {std::pair<std::size_t,std::uint64_t>{5,gemm.rows()},
        {6,gemm.output_features()}, {7,gemm.input_features()}}) {
    auto status = packet->set_u64(ordinal, value);
    if (!status.ok()) return status;
  }
  auto geometry = KernelLaunchGeometry::Create(
      gemm.blocks(), 1, 1, QwenInt4GemmPlan::kThreadsPerBlock, 1, 1, 0);
  if (!geometry.ok()) return geometry.status();
  return QwenInt4DispatchPlan(function, *geometry, std::move(*packet));
}

Status QwenInt4DispatchPlan::submit(KernelLaunchDriver& driver,
                                    DriverStreamHandle stream) {
  if (submitted_) {
    return Status::FailedPrecondition(
        "Qwen INT4 dispatch plan is single-use");
  }
  submitted_ = true;
  return submit_verified_kernel(driver, function_, geometry_, stream,
                                arguments_);
}

}  // namespace pih
