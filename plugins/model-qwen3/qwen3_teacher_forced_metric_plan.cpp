#include "pih/model/qwen3_teacher_forced_metric_plan.h"

#include <array>
#include <string>
#include <vector>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

bool exact_vector(const TensorView& view, DType dtype, std::uint64_t rows,
                  std::int32_t device) {
  return view.dtype() == dtype && view.rank() == 1 && view.dim(0) == rows &&
         view.stride(0) == 1 && view.device().type() == DeviceType::kCuda &&
         view.device().index() == device && view.generation() != 0;
}

Result<KernelPointerContract> contract(
    std::string id, KernelPointerAccess access,
    KernelPointerOwnerClass owner, std::int32_t rank, std::int32_t device,
    std::uint64_t bytes, std::uint64_t alignment) {
  return KernelPointerContract::Create(std::move(id), access, owner, rank,
                                       device, bytes, alignment, 0, false);
}

}  // namespace

Result<QwenTeacherForcedMetricPlan> QwenTeacherForcedMetricPlan::Create(
    const ResolvedKernelFunction& function, const TensorView& logits,
    const TensorView& target_tokens, const TensorView& argmax_tokens,
    const TensorView& target_nll, const TensorView& nonfinite_rows,
    const TensorView& error_flag, std::int32_t owning_rank) {
  if (owning_rank < 0 || logits.dtype() != DType::kFloat32 ||
      logits.rank() != 2 || logits.dim(0) <= 0 ||
      logits.dim(0) > kMaximumRows || logits.dim(1) != kVocabularySize ||
      logits.stride(0) != kVocabularySize || logits.stride(1) != 1 ||
      logits.device().type() != DeviceType::kCuda ||
      logits.device().index() != owning_rank || logits.generation() == 0) {
    return Status::InvalidArgument("Qwen metric logits contract is invalid");
  }
  const auto rows = static_cast<std::uint32_t>(logits.dim(0));
  const auto device = logits.device().index();
  if (!exact_vector(target_tokens, DType::kUInt32, rows, device) ||
      !exact_vector(argmax_tokens, DType::kUInt32, rows, device) ||
      !exact_vector(target_nll, DType::kFloat64, rows, device) ||
      !exact_vector(nonfinite_rows, DType::kUInt32, rows, device) ||
      !exact_vector(error_flag, DType::kUInt32, 1, device)) {
    return Status::InvalidArgument("Qwen metric vector contract is invalid");
  }
  auto manifest = qwen_bf16_kernel_manifest(
      QwenBf16Primitive::kTeacherForcedMetric,
      function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  auto symbol = qwen_bf16_kernel_symbol(
      QwenBf16Primitive::kTeacherForcedMetric);
  if (!symbol.ok()) return symbol.status();
  if (function.handle == 0 || function.symbol != *symbol ||
      function.logical_id != manifest->logical_id() ||
      function.selected_cubin_sha256 != manifest->selected_cubin_sha256() ||
      function.parameter_abi_sha256 != manifest->parameter_abi_sha256()) {
    return Status::InvalidArgument("Qwen metric function identity differs");
  }
  auto logit_elements = checked_mul_u64(rows, kVocabularySize);
  if (!logit_elements.ok()) return logit_elements.status();
  auto logit_bytes = checked_mul_u64(*logit_elements, sizeof(float));
  if (!logit_bytes.ok()) return logit_bytes.status();
  const std::array<const TensorView*, 6> views{
      &logits, &target_tokens, &argmax_tokens, &target_nll,
      &nonfinite_rows, &error_flag};
  const std::array<std::uint64_t, 6> bytes{
      *logit_bytes, rows * sizeof(std::uint32_t),
      rows * sizeof(std::uint32_t), rows * sizeof(double),
      rows * sizeof(std::uint32_t), sizeof(std::uint32_t)};
  const std::array<std::uint64_t, 6> alignments{4, 4, 4, 8, 4, 4};
  const std::array accesses{
      KernelPointerAccess::kRead, KernelPointerAccess::kRead,
      KernelPointerAccess::kWrite, KernelPointerAccess::kWrite,
      KernelPointerAccess::kWrite, KernelPointerAccess::kAtomic};
  const std::array owners{
      KernelPointerOwnerClass::kWorkspace,
      KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kKvState};
  std::vector<KernelPointerContract> contracts;
  contracts.reserve(views.size());
  for (std::size_t index = 0; index < views.size(); ++index) {
    auto value = contract(std::string(manifest->parameter(index).contract_id),
                          accesses[index], owners[index], owning_rank, device,
                          bytes[index], alignments[index]);
    if (!value.ok()) return value.status();
    contracts.push_back(std::move(*value));
  }
  auto packet = KernelArgumentPacket::Create(*manifest, contracts);
  if (!packet.ok()) return packet.status();
  for (std::size_t index = 0; index < views.size(); ++index) {
    auto pointer = VerifiedDevicePointer::Create(
        *views[index], contracts[index], owners[index], owning_rank,
        views[index]->generation());
    if (!pointer.ok()) return pointer.status();
    auto status = packet->set_device_pointer(index, *pointer);
    if (!status.ok()) return status;
  }
  auto status = packet->set_u32(6, rows);
  if (status.ok()) status = packet->set_u32(7, kVocabularySize);
  if (!status.ok()) return status;
  auto geometry = KernelLaunchGeometry::Create(rows, 1, 1, 256, 1, 1, 0);
  if (!geometry.ok()) return geometry.status();
  return QwenTeacherForcedMetricPlan(function, *geometry, std::move(*packet),
                                     logits, rows);
}

Status QwenTeacherForcedMetricPlan::submit(KernelLaunchDriver& driver,
                                           DriverStreamHandle stream) {
  if (submitted_)
    return Status::FailedPrecondition("Qwen metric plan cannot replay");
  submitted_ = true;
  return submit_verified_kernel(driver, function_, geometry_, stream,
                                arguments_);
}

}  // namespace pih
