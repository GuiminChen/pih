#include "pih/model/qwen3_bf16_dispatch_plan.h"

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "pih/backend/cuda/elementwise_launch.h"
#include "pih/core/checked_math.h"
#include "pih/model/qwen3_kv_slot_pool.h"

namespace pih {
namespace {

Status require_contiguous(const TensorView& view) {
  if (view.num_elements() == 0) {
    return Status::InvalidArgument("Qwen BF16 dispatch forbids empty tensors");
  }
  std::uint64_t expected_stride = 1;
  for (std::size_t axis = view.rank(); axis > 0; --axis) {
    const std::size_t index = axis - 1;
    if (view.stride(index) != expected_stride) {
      return Status::InvalidArgument("Qwen BF16 dispatch requires contiguous tensors");
    }
    auto next = checked_mul_u64(expected_stride, view.dim(index));
    if (!next.ok()) return next.status();
    expected_stride = next.value();
  }
  return Status::Ok();
}

Status require_same_shape(const TensorView& first, const TensorView& second,
                          const TensorView& output) {
  if (first.rank() != second.rank() || first.rank() != output.rank()) {
    return Status::InvalidArgument("Qwen BF16 elementwise ranks differ");
  }
  for (std::size_t axis = 0; axis < first.rank(); ++axis) {
    if (first.dim(axis) != second.dim(axis) ||
        first.dim(axis) != output.dim(axis)) {
      return Status::InvalidArgument("Qwen BF16 elementwise shapes differ");
    }
  }
  return Status::Ok();
}

Status require_cuda_bf16(const TensorView& view, std::int32_t device_index) {
  if (view.dtype() != DType::kBFloat16 ||
      view.device().type() != DeviceType::kCuda ||
      view.device().index() != device_index || view.generation() == 0) {
    return Status::InvalidArgument(
        "Qwen BF16 tensor dtype, device, or generation is invalid");
  }
  return require_contiguous(view);
}

Status require_cuda_float32(const TensorView& view, std::int32_t device_index) {
  if (view.dtype() != DType::kFloat32 ||
      view.device().type() != DeviceType::kCuda ||
      view.device().index() != device_index || view.generation() == 0) {
    return Status::InvalidArgument(
        "Qwen float32 tensor dtype, device, or generation is invalid");
  }
  return require_contiguous(view);
}

Status require_cuda_bytes(const TensorView& view, std::int32_t device_index,
                          std::uint64_t exact_bytes) {
  if (view.dtype() != DType::kUInt8 || view.rank() != 1 ||
      view.num_elements() != exact_bytes ||
      view.device().type() != DeviceType::kCuda ||
      view.device().index() != device_index || view.generation() == 0) {
    return Status::InvalidArgument(
        "Qwen raw CUDA buffer extent, device, or generation is invalid");
  }
  return require_contiguous(view);
}

Status require_function(const ResolvedKernelFunction& function,
                        QwenBf16Primitive primitive,
                        const KernelSignatureManifest& manifest) {
  auto symbol = qwen_bf16_kernel_symbol(primitive);
  if (!symbol.ok()) return symbol.status();
  if (function.handle == 0 || function.symbol != symbol.value() ||
      function.logical_id != manifest.logical_id() ||
      function.selected_cubin_sha256 != manifest.selected_cubin_sha256() ||
      function.parameter_abi_sha256 != manifest.parameter_abi_sha256()) {
    return Status::InvalidArgument(
        "resolved Qwen function identity differs from primitive manifest");
  }
  return Status::Ok();
}

Result<KernelLaunchGeometry> elementwise_geometry(std::uint64_t elements) {
  auto launch = ElementwiseLaunch::Create(elements, 256);
  if (!launch.ok()) return launch.status();
  return KernelLaunchGeometry::Create(launch->grid_blocks(), 1, 1,
                                      launch->block_threads(), 1, 1, 0);
}

Result<std::uint64_t> exact_bytes(std::uint64_t elements,
                                  std::uint64_t element_bytes) {
  return checked_mul_u64(elements, element_bytes);
}

Result<KernelPointerContract> pointer_contract(
    std::string id, KernelPointerAccess access,
    KernelPointerOwnerClass owner, std::int32_t rank, std::int32_t device,
    std::uint64_t bytes, std::uint64_t alignment, bool allow_alias) {
  return KernelPointerContract::Create(std::move(id), access, owner, rank,
                                       device, bytes, alignment,
                                       allow_alias ? 1U : 0U, allow_alias);
}

}  // namespace

Result<QwenBf16DispatchPlan> QwenBf16DispatchPlan::CreateEmbedding(
    const ResolvedKernelFunction& function, const TensorView& table,
    const TensorView& token_ids, const TensorView& output,
    std::int32_t owning_rank) {
  if (owning_rank < 0 || table.rank() != 2 || token_ids.rank() != 1 ||
      output.rank() != 2 || table.dim(0) == 0 || table.dim(1) == 0 ||
      token_ids.dim(0) == 0 || output.dim(0) != token_ids.dim(0) ||
      output.dim(1) != table.dim(1)) {
    return Status::InvalidArgument("Qwen embedding shapes or rank are invalid");
  }
  const std::int32_t device = table.device().index();
  Status valid = require_cuda_bf16(table, device);
  if (!valid.ok()) return valid;
  valid = require_cuda_bf16(output, device);
  if (!valid.ok()) return valid;
  if (token_ids.dtype() != DType::kInt64 ||
      token_ids.device().type() != DeviceType::kCuda ||
      token_ids.device().index() != device || token_ids.generation() == 0) {
    return Status::InvalidArgument("Qwen token id tensor is invalid");
  }
  valid = require_contiguous(token_ids);
  if (!valid.ok()) return valid;

  auto manifest = qwen_bf16_kernel_manifest(
      QwenBf16Primitive::kEmbedding, function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  valid = require_function(function, QwenBf16Primitive::kEmbedding,
                           manifest.value());
  if (!valid.ok()) return valid;

  auto table_bytes = exact_bytes(table.num_elements(), 2);
  auto id_bytes = exact_bytes(token_ids.num_elements(), 8);
  auto output_bytes = exact_bytes(output.num_elements(), 2);
  if (!table_bytes.ok()) return table_bytes.status();
  if (!id_bytes.ok()) return id_bytes.status();
  if (!output_bytes.ok()) return output_bytes.status();
  std::vector<KernelPointerContract> contracts;
  auto table_contract = pointer_contract(
      std::string(manifest->parameter(0).contract_id), KernelPointerAccess::kRead,
      KernelPointerOwnerClass::kWeight, owning_rank, device,
      table_bytes.value(), 2, false);
  auto ids_contract = pointer_contract(
      std::string(manifest->parameter(1).contract_id), KernelPointerAccess::kRead,
      KernelPointerOwnerClass::kActivation, owning_rank, device,
      id_bytes.value(), 8, false);
  auto output_contract = pointer_contract(
      std::string(manifest->parameter(2).contract_id), KernelPointerAccess::kWrite,
      KernelPointerOwnerClass::kActivation, owning_rank, device,
      output_bytes.value(), 2, false);
  if (!table_contract.ok()) return table_contract.status();
  if (!ids_contract.ok()) return ids_contract.status();
  if (!output_contract.ok()) return output_contract.status();
  contracts.push_back(std::move(table_contract).value());
  contracts.push_back(std::move(ids_contract).value());
  contracts.push_back(std::move(output_contract).value());
  auto packet = KernelArgumentPacket::Create(manifest.value(), contracts);
  if (!packet.ok()) return packet.status();
  auto table_pointer = VerifiedDevicePointer::Create(
      table, contracts[0], KernelPointerOwnerClass::kWeight, owning_rank,
      table.generation());
  auto ids_pointer = VerifiedDevicePointer::Create(
      token_ids, contracts[1], KernelPointerOwnerClass::kActivation,
      owning_rank, token_ids.generation());
  auto output_pointer = VerifiedDevicePointer::Create(
      output, contracts[2], KernelPointerOwnerClass::kActivation, owning_rank,
      output.generation());
  if (!table_pointer.ok()) return table_pointer.status();
  if (!ids_pointer.ok()) return ids_pointer.status();
  if (!output_pointer.ok()) return output_pointer.status();
  valid = packet->set_device_pointer(0, table_pointer.value());
  if (!valid.ok()) return valid;
  valid = packet->set_device_pointer(1, ids_pointer.value());
  if (!valid.ok()) return valid;
  valid = packet->set_device_pointer(2, output_pointer.value());
  if (!valid.ok()) return valid;
  valid = packet->set_u64(3, token_ids.num_elements());
  if (!valid.ok()) return valid;
  valid = packet->set_u64(4, table.dim(0));
  if (!valid.ok()) return valid;
  valid = packet->set_u64(5, table.dim(1));
  if (!valid.ok()) return valid;
  auto geometry = elementwise_geometry(output.num_elements());
  if (!geometry.ok()) return geometry.status();
  return QwenBf16DispatchPlan(QwenBf16Primitive::kEmbedding, function,
                              geometry.value(), std::move(packet).value());
}

Result<QwenBf16DispatchPlan> QwenBf16DispatchPlan::CreateElementwise(
    QwenBf16Primitive primitive, const ResolvedKernelFunction& function,
    const TensorView& first, const TensorView& second,
    const TensorView& output, std::int32_t owning_rank) {
  if (primitive != QwenBf16Primitive::kResidualAdd &&
      primitive != QwenBf16Primitive::kSiluMul) {
    return Status::InvalidArgument("primitive is not Qwen BF16 elementwise");
  }
  if (owning_rank < 0) return Status::InvalidArgument("owning rank is negative");
  Status valid = require_same_shape(first, second, output);
  if (!valid.ok()) return valid;
  const std::int32_t device = first.device().index();
  for (const TensorView* view : {&first, &second, &output}) {
    valid = require_cuda_bf16(*view, device);
    if (!valid.ok()) return valid;
  }
  auto manifest =
      qwen_bf16_kernel_manifest(primitive, function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  valid = require_function(function, primitive, manifest.value());
  if (!valid.ok()) return valid;
  auto bytes = exact_bytes(first.num_elements(), 2);
  if (!bytes.ok()) return bytes.status();
  std::vector<KernelPointerContract> contracts;
  for (std::size_t ordinal = 0; ordinal < 3; ++ordinal) {
    auto contract = pointer_contract(
        std::string(manifest->parameter(ordinal).contract_id),
        ordinal == 2 ? KernelPointerAccess::kWrite : KernelPointerAccess::kRead,
        KernelPointerOwnerClass::kActivation, owning_rank, device,
        bytes.value(), 2, true);
    if (!contract.ok()) return contract.status();
    contracts.push_back(std::move(contract).value());
  }
  auto packet = KernelArgumentPacket::Create(manifest.value(), contracts);
  if (!packet.ok()) return packet.status();
  const std::array<const TensorView*, 3> views{&first, &second, &output};
  for (std::size_t ordinal = 0; ordinal < views.size(); ++ordinal) {
    auto pointer = VerifiedDevicePointer::Create(
        *views[ordinal], contracts[ordinal],
        KernelPointerOwnerClass::kActivation, owning_rank,
        views[ordinal]->generation());
    if (!pointer.ok()) return pointer.status();
    valid = packet->set_device_pointer(ordinal, pointer.value());
    if (!valid.ok()) return valid;
  }
  valid = packet->set_u64(3, first.num_elements());
  if (!valid.ok()) return valid;
  auto geometry = elementwise_geometry(first.num_elements());
  if (!geometry.ok()) return geometry.status();
  return QwenBf16DispatchPlan(primitive, function, geometry.value(),
                              std::move(packet).value());
}

Result<QwenBf16DispatchPlan> QwenBf16DispatchPlan::CreateRmsNorm(
    const ResolvedKernelFunction& function, const TensorView& input,
    const TensorView& weight, const TensorView& output, float epsilon,
    std::int32_t owning_rank) {
  if (owning_rank < 0 || (input.rank() != 2 && input.rank() != 3) ||
      output.rank() != input.rank() || weight.rank() != 1 ||
      input.num_elements() == 0 ||
      !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return Status::InvalidArgument("Qwen RMSNorm shape or epsilon is invalid");
  }
  const std::uint64_t hidden_size = input.dim(input.rank() - 1);
  if ((hidden_size != 128 && hidden_size != 1024) ||
      weight.dim(0) != hidden_size) {
    return Status::InvalidArgument("Qwen RMSNorm hidden dimension is invalid");
  }
  for (std::size_t axis = 0; axis < input.rank(); ++axis) {
    if (output.dim(axis) != input.dim(axis)) {
      return Status::InvalidArgument("Qwen RMSNorm output shape is invalid");
    }
  }
  const std::uint64_t rows = input.num_elements() / hidden_size;
  const std::int32_t device = input.device().index();
  for (const TensorView* view : {&input, &weight, &output}) {
    const Status valid = require_cuda_bf16(*view, device);
    if (!valid.ok()) return valid;
  }
  auto manifest = qwen_bf16_kernel_manifest(
      QwenBf16Primitive::kRmsNorm, function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  Status valid = require_function(function, QwenBf16Primitive::kRmsNorm,
                                  manifest.value());
  if (!valid.ok()) return valid;
  auto activation_bytes = exact_bytes(input.num_elements(), 2);
  auto weight_bytes = exact_bytes(weight.num_elements(), 2);
  if (!activation_bytes.ok()) return activation_bytes.status();
  if (!weight_bytes.ok()) return weight_bytes.status();
  std::vector<KernelPointerContract> contracts;
  auto input_contract = pointer_contract(
      std::string(manifest->parameter(0).contract_id), KernelPointerAccess::kRead,
      KernelPointerOwnerClass::kActivation, owning_rank, device,
      activation_bytes.value(), 2, true);
  auto weight_contract = pointer_contract(
      std::string(manifest->parameter(1).contract_id), KernelPointerAccess::kRead,
      KernelPointerOwnerClass::kWeight, owning_rank, device,
      weight_bytes.value(), 2, false);
  auto output_contract = pointer_contract(
      std::string(manifest->parameter(2).contract_id), KernelPointerAccess::kWrite,
      KernelPointerOwnerClass::kActivation, owning_rank, device,
      activation_bytes.value(), 2, true);
  if (!input_contract.ok()) return input_contract.status();
  if (!weight_contract.ok()) return weight_contract.status();
  if (!output_contract.ok()) return output_contract.status();
  contracts.push_back(std::move(input_contract).value());
  contracts.push_back(std::move(weight_contract).value());
  contracts.push_back(std::move(output_contract).value());
  auto packet = KernelArgumentPacket::Create(manifest.value(), contracts);
  if (!packet.ok()) return packet.status();
  const std::array<const TensorView*, 3> views{&input, &weight, &output};
  const std::array owners{KernelPointerOwnerClass::kActivation,
                          KernelPointerOwnerClass::kWeight,
                          KernelPointerOwnerClass::kActivation};
  for (std::size_t ordinal = 0; ordinal < views.size(); ++ordinal) {
    auto pointer = VerifiedDevicePointer::Create(
        *views[ordinal], contracts[ordinal], owners[ordinal], owning_rank,
        views[ordinal]->generation());
    if (!pointer.ok()) return pointer.status();
    valid = packet->set_device_pointer(ordinal, pointer.value());
    if (!valid.ok()) return valid;
  }
  valid = packet->set_u64(3, rows);
  if (!valid.ok()) return valid;
  valid = packet->set_u64(4, hidden_size);
  if (!valid.ok()) return valid;
  valid = packet->set_float32(5, epsilon);
  if (!valid.ok()) return valid;
  if (rows > KernelLaunchGeometry::kMaximumGridX) {
    return Status::ResourceExhausted("Qwen RMSNorm rows exceed CUDA grid.x");
  }
  auto geometry = KernelLaunchGeometry::Create(
      static_cast<std::uint32_t>(rows), 1, 1, 256, 1, 1, 0);
  if (!geometry.ok()) return geometry.status();
  return QwenBf16DispatchPlan(QwenBf16Primitive::kRmsNorm, function,
                              geometry.value(), std::move(packet).value());
}

Result<QwenBf16DispatchPlan> QwenBf16DispatchPlan::CreateRope(
    const ResolvedKernelFunction& function, const TensorView& input,
    const TensorView& cosine, const TensorView& sine,
    const TensorView& output, std::int32_t owning_rank) {
  constexpr std::uint64_t kHeadDim = 128;
  if (owning_rank < 0 || input.rank() < 3 || output.rank() != input.rank() ||
      input.dim(input.rank() - 1) != kHeadDim) {
    return Status::InvalidArgument("Qwen RoPE input rank or head dimension is invalid");
  }
  Status valid = require_same_shape(input, input, output);
  if (!valid.ok()) return valid;
  const std::int32_t device = input.device().index();
  valid = require_cuda_bf16(input, device);
  if (!valid.ok()) return valid;
  valid = require_cuda_bf16(output, device);
  if (!valid.ok()) return valid;
  valid = require_cuda_float32(cosine, device);
  if (!valid.ok()) return valid;
  valid = require_cuda_float32(sine, device);
  if (!valid.ok()) return valid;
  const std::uint64_t vectors = input.num_elements() / kHeadDim;
  const std::uint64_t heads = input.dim(input.rank() - 2);
  if (heads != 8 && heads != 16) {
    return Status::InvalidArgument("Qwen RoPE head count must be 8 or 16");
  }
  const std::uint64_t tokens = vectors / heads;
  auto pairs = checked_mul_u64(vectors, kHeadDim / 2);
  if (!pairs.ok()) return pairs.status();
  auto angle_pairs = checked_mul_u64(tokens, kHeadDim / 2);
  if (!angle_pairs.ok()) return angle_pairs.status();
  if (cosine.num_elements() != angle_pairs.value() ||
      sine.num_elements() != angle_pairs.value()) {
    return Status::InvalidArgument("Qwen RoPE angle extent differs from input");
  }
  auto manifest = qwen_bf16_kernel_manifest(
      QwenBf16Primitive::kRope, function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  valid = require_function(function, QwenBf16Primitive::kRope,
                           manifest.value());
  if (!valid.ok()) return valid;
  auto activation_bytes = exact_bytes(input.num_elements(), 2);
  auto angle_bytes = exact_bytes(angle_pairs.value(), 4);
  if (!activation_bytes.ok()) return activation_bytes.status();
  if (!angle_bytes.ok()) return angle_bytes.status();
  std::vector<KernelPointerContract> contracts;
  const std::array<std::uint64_t, 4> spans{
      activation_bytes.value(), angle_bytes.value(), angle_bytes.value(),
      activation_bytes.value()};
  const std::array<std::uint64_t, 4> alignments{2, 4, 4, 2};
  const std::array owners{KernelPointerOwnerClass::kActivation,
                          KernelPointerOwnerClass::kWeight,
                          KernelPointerOwnerClass::kWeight,
                          KernelPointerOwnerClass::kActivation};
  for (std::size_t ordinal = 0; ordinal < spans.size(); ++ordinal) {
    auto contract = pointer_contract(
        std::string(manifest->parameter(ordinal).contract_id),
        ordinal == 3 ? KernelPointerAccess::kWrite : KernelPointerAccess::kRead,
        owners[ordinal], owning_rank, device, spans[ordinal],
        alignments[ordinal], ordinal == 0 || ordinal == 3);
    if (!contract.ok()) return contract.status();
    contracts.push_back(std::move(contract).value());
  }
  auto packet = KernelArgumentPacket::Create(manifest.value(), contracts);
  if (!packet.ok()) return packet.status();
  const std::array<const TensorView*, 4> views{&input, &cosine, &sine, &output};
  for (std::size_t ordinal = 0; ordinal < views.size(); ++ordinal) {
    auto pointer = VerifiedDevicePointer::Create(
        *views[ordinal], contracts[ordinal], owners[ordinal], owning_rank,
        views[ordinal]->generation());
    if (!pointer.ok()) return pointer.status();
    valid = packet->set_device_pointer(ordinal, pointer.value());
    if (!valid.ok()) return valid;
  }
  valid = packet->set_u64(4, vectors);
  if (!valid.ok()) return valid;
  valid = packet->set_u64(5, heads);
  if (!valid.ok()) return valid;
  valid = packet->set_u64(6, kHeadDim);
  if (!valid.ok()) return valid;
  auto geometry = elementwise_geometry(pairs.value());
  if (!geometry.ok()) return geometry.status();
  return QwenBf16DispatchPlan(QwenBf16Primitive::kRope, function,
                              geometry.value(), std::move(packet).value());
}

Result<QwenBf16DispatchPlan> QwenBf16DispatchPlan::CreateRopeAngles(
    const ResolvedKernelFunction& function, const TensorView& positions,
    const TensorView& cosine, const TensorView& sine,
    std::int32_t owning_rank) {
  if (owning_rank < 0 || positions.rank() != 1 || cosine.rank() != 2 ||
      sine.rank() != 2 || cosine.dim(0) != positions.dim(0) ||
      sine.dim(0) != positions.dim(0) || cosine.dim(1) != 64 ||
      sine.dim(1) != 64) {
    return Status::InvalidArgument("Qwen RoPE angle tensor geometry is invalid");
  }
  const auto device = positions.device().index();
  if (positions.dtype() != DType::kInt64 ||
      positions.device().type() != DeviceType::kCuda ||
      positions.generation() == 0) {
    return Status::InvalidArgument("Qwen RoPE positions tensor is invalid");
  }
  Status valid = require_contiguous(positions);
  if (!valid.ok()) return valid;
  valid = require_cuda_float32(cosine, device);
  if (!valid.ok()) return valid;
  valid = require_cuda_float32(sine, device);
  if (!valid.ok()) return valid;
  auto manifest = qwen_bf16_kernel_manifest(
      QwenBf16Primitive::kRopeAngles, function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  valid = require_function(function, QwenBf16Primitive::kRopeAngles, *manifest);
  if (!valid.ok()) return valid;

  const std::array<const TensorView*, 3> views{&positions, &cosine, &sine};
  const std::array<std::uint64_t, 3> bytes{
      positions.num_elements() * 8, cosine.num_elements() * 4,
      sine.num_elements() * 4};
  const std::array<std::uint64_t, 3> alignments{8, 4, 4};
  const std::array accesses{KernelPointerAccess::kRead,
                            KernelPointerAccess::kWrite,
                            KernelPointerAccess::kWrite};
  const std::array owners{KernelPointerOwnerClass::kActivation,
                          KernelPointerOwnerClass::kWorkspace,
                          KernelPointerOwnerClass::kWorkspace};
  std::vector<KernelPointerContract> contracts;
  for (std::size_t ordinal = 0; ordinal < views.size(); ++ordinal) {
    auto contract = pointer_contract(
        std::string(manifest->parameter(ordinal).contract_id), accesses[ordinal],
        owners[ordinal], owning_rank, device, bytes[ordinal],
        alignments[ordinal], false);
    if (!contract.ok()) return contract.status();
    contracts.push_back(std::move(*contract));
  }
  auto packet = KernelArgumentPacket::Create(*manifest, contracts);
  if (!packet.ok()) return packet.status();
  for (std::size_t ordinal = 0; ordinal < views.size(); ++ordinal) {
    auto pointer = VerifiedDevicePointer::Create(
        *views[ordinal], contracts[ordinal], owners[ordinal], owning_rank,
        views[ordinal]->generation());
    if (!pointer.ok()) return pointer.status();
    valid = packet->set_device_pointer(ordinal, *pointer);
    if (!valid.ok()) return valid;
  }
  valid = packet->set_u64(3, positions.num_elements());
  if (!valid.ok()) return valid;
  valid = packet->set_u64(4, 128);
  if (!valid.ok()) return valid;
  auto geometry = elementwise_geometry(cosine.num_elements());
  if (!geometry.ok()) return geometry.status();
  return QwenBf16DispatchPlan(QwenBf16Primitive::kRopeAngles, function,
                              *geometry, std::move(*packet));
}

Result<QwenBf16DispatchPlan> QwenBf16DispatchPlan::CreateGreedyArgmax(
    const ResolvedKernelFunction& function, const TensorView& logits,
    const TensorView& sampled_token, const TensorView& error_flag,
    std::int32_t owning_rank) {
  if (owning_rank < 0 || logits.rank() != 2 || logits.dim(0) != 1 ||
      logits.dim(1) != 151936 || sampled_token.rank() != 1 ||
      sampled_token.dim(0) != 1 || sampled_token.dtype() != DType::kInt64) {
    return Status::InvalidArgument("Qwen greedy argmax geometry is invalid");
  }
  const auto device = logits.device().index();
  Status valid = require_cuda_float32(logits, device);
  if (!valid.ok()) return valid;
  if (sampled_token.device().type() != DeviceType::kCuda ||
      sampled_token.device().index() != device ||
      sampled_token.generation() == 0) {
    return Status::InvalidArgument("Qwen sampled token tensor is invalid");
  }
  valid = require_contiguous(sampled_token);
  if (!valid.ok()) return valid;
  valid = require_cuda_bytes(error_flag, device, 4);
  if (!valid.ok()) return valid;
  auto manifest = qwen_bf16_kernel_manifest(
      QwenBf16Primitive::kGreedyArgmax, function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  valid = require_function(function, QwenBf16Primitive::kGreedyArgmax,
                           *manifest);
  if (!valid.ok()) return valid;

  const std::array<const TensorView*, 3> views{&logits, &sampled_token,
                                               &error_flag};
  const std::array<std::uint64_t, 3> bytes{151936 * 4, 8, 4};
  const std::array<std::uint64_t, 3> alignments{4, 8, 4};
  const std::array accesses{KernelPointerAccess::kRead,
                            KernelPointerAccess::kWrite,
                            KernelPointerAccess::kAtomic};
  const std::array owners{KernelPointerOwnerClass::kWorkspace,
                          KernelPointerOwnerClass::kActivation,
                          KernelPointerOwnerClass::kKvState};
  std::vector<KernelPointerContract> contracts;
  for (std::size_t ordinal = 0; ordinal < views.size(); ++ordinal) {
    auto contract = pointer_contract(
        std::string(manifest->parameter(ordinal).contract_id), accesses[ordinal],
        owners[ordinal], owning_rank, device, bytes[ordinal],
        alignments[ordinal], false);
    if (!contract.ok()) return contract.status();
    contracts.push_back(std::move(*contract));
  }
  auto packet = KernelArgumentPacket::Create(*manifest, contracts);
  if (!packet.ok()) return packet.status();
  for (std::size_t ordinal = 0; ordinal < views.size(); ++ordinal) {
    auto pointer = VerifiedDevicePointer::Create(
        *views[ordinal], contracts[ordinal], owners[ordinal], owning_rank,
        views[ordinal]->generation());
    if (!pointer.ok()) return pointer.status();
    valid = packet->set_device_pointer(ordinal, *pointer);
    if (!valid.ok()) return valid;
  }
  valid = packet->set_u64(3, 151936);
  if (!valid.ok()) return valid;
  auto geometry = KernelLaunchGeometry::Create(1, 1, 1, 256, 1, 1, 0);
  if (!geometry.ok()) return geometry.status();
  return QwenBf16DispatchPlan(QwenBf16Primitive::kGreedyArgmax, function,
                              *geometry, std::move(*packet));
}

Result<QwenBf16DispatchPlan> QwenBf16DispatchPlan::CreateKvAppend(
    const ResolvedKernelFunction& function, const TensorView& key_input,
    const TensorView& value_input, const TensorView& kv_backing,
    const TensorView& slot_states, const TensorView& handles,
    const TensorView& token_offsets, const TensorView& error_flag,
    std::uint32_t owner_sequence_index, std::uint32_t layer,
    std::uint32_t slot_count, std::int32_t owning_rank) {
  constexpr std::uint64_t kMaximumBatchTokens = 4096;
  constexpr std::uint64_t kKvHeads = 8;
  constexpr std::uint64_t kHeadDimension = 128;
  if (owning_rank < 0 || slot_count == 0 ||
      slot_count > QwenKvSlotPool::kMaximumSlots ||
      layer >= QwenKvSlotPool::kLayerCount || key_input.rank() != 3 ||
      value_input.rank() != 3 || key_input.dim(0) == 0 ||
      key_input.dim(0) > kMaximumBatchTokens ||
      key_input.dim(1) != kKvHeads ||
      key_input.dim(2) != kHeadDimension ||
      value_input.dim(0) != key_input.dim(0) ||
      value_input.dim(1) != kKvHeads ||
      value_input.dim(2) != kHeadDimension) {
    return Status::InvalidArgument("Qwen KV append shape or scalar is invalid");
  }
  const std::int32_t device = key_input.device().index();
  Status valid = require_cuda_bf16(key_input, device);
  if (!valid.ok()) return valid;
  valid = require_cuda_bf16(value_input, device);
  if (!valid.ok()) return valid;
  const std::uint64_t token_count = key_input.dim(0);
  auto kv_bytes = checked_mul_u64(slot_count,
                                  QwenKvSlotPool::kSlotPayloadBytes);
  auto state_bytes = checked_mul_u64(slot_count, sizeof(QwenKvSlotState));
  auto handle_bytes = checked_mul_u64(token_count, sizeof(QwenKvBlockHandle));
  auto offset_bytes = checked_mul_u64(token_count, sizeof(std::uint16_t));
  if (!kv_bytes.ok()) return kv_bytes.status();
  if (!state_bytes.ok()) return state_bytes.status();
  if (!handle_bytes.ok()) return handle_bytes.status();
  if (!offset_bytes.ok()) return offset_bytes.status();
  const std::array<const TensorView*, 5> raw_views{
      &kv_backing, &slot_states, &handles, &token_offsets, &error_flag};
  const std::array<std::uint64_t, 5> raw_bytes{
      kv_bytes.value(), state_bytes.value(), handle_bytes.value(),
      offset_bytes.value(), 4};
  for (std::size_t index = 0; index < raw_views.size(); ++index) {
    valid = require_cuda_bytes(*raw_views[index], device, raw_bytes[index]);
    if (!valid.ok()) return valid;
  }
  auto manifest = qwen_bf16_kernel_manifest(
      QwenBf16Primitive::kKvAppend, function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  valid = require_function(function, QwenBf16Primitive::kKvAppend,
                           manifest.value());
  if (!valid.ok()) return valid;

  const std::array<const TensorView*, 7> views{
      &key_input, &value_input, &kv_backing, &slot_states,
      &handles, &token_offsets, &error_flag};
  const std::array<KernelPointerOwnerClass, 7> owners{
      KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kKvState,
      KernelPointerOwnerClass::kKvState,
      KernelPointerOwnerClass::kKvState,
      KernelPointerOwnerClass::kWorkspace,
      KernelPointerOwnerClass::kWorkspace};
  const std::array<KernelPointerAccess, 7> accesses{
      KernelPointerAccess::kRead, KernelPointerAccess::kRead,
      KernelPointerAccess::kWrite, KernelPointerAccess::kRead,
      KernelPointerAccess::kRead, KernelPointerAccess::kRead,
      KernelPointerAccess::kAtomic};
  const std::array<std::uint64_t, 7> spans{
      key_input.byte_span(), value_input.byte_span(), kv_bytes.value(),
      state_bytes.value(), handle_bytes.value(), offset_bytes.value(), 4};
  const std::array<std::uint64_t, 7> alignments{2, 2, 2, 4, 4, 2, 4};
  std::vector<KernelPointerContract> contracts;
  contracts.reserve(views.size());
  for (std::size_t ordinal = 0; ordinal < views.size(); ++ordinal) {
    auto contract = pointer_contract(
        std::string(manifest->parameter(ordinal).contract_id), accesses[ordinal],
        owners[ordinal], owning_rank, device, spans[ordinal],
        alignments[ordinal], false);
    if (!contract.ok()) return contract.status();
    contracts.push_back(std::move(contract).value());
  }
  auto packet = KernelArgumentPacket::Create(manifest.value(), contracts);
  if (!packet.ok()) return packet.status();
  for (std::size_t ordinal = 0; ordinal < views.size(); ++ordinal) {
    auto pointer = VerifiedDevicePointer::Create(
        *views[ordinal], contracts[ordinal], owners[ordinal], owning_rank,
        views[ordinal]->generation());
    if (!pointer.ok()) return pointer.status();
    valid = packet->set_device_pointer(ordinal, pointer.value());
    if (!valid.ok()) return valid;
  }
  valid = packet->set_u32(7, owner_sequence_index);
  if (!valid.ok()) return valid;
  valid = packet->set_u32(8, layer);
  if (!valid.ok()) return valid;
  valid = packet->set_u64(9, token_count);
  if (!valid.ok()) return valid;
  valid = packet->set_u32(10, slot_count);
  if (!valid.ok()) return valid;
  auto elements = checked_mul_u64(token_count, 2 * kKvHeads * kHeadDimension);
  if (!elements.ok()) return elements.status();
  auto geometry = elementwise_geometry(elements.value());
  if (!geometry.ok()) return geometry.status();
  return QwenBf16DispatchPlan(QwenBf16Primitive::kKvAppend, function,
                              geometry.value(), std::move(packet).value());
}

Result<QwenBf16DispatchPlan> QwenBf16DispatchPlan::CreatePagedGqa(
    const ResolvedKernelFunction& function, const TensorView& query,
    const TensorView& output, const TensorView& kv_backing,
    const TensorView& slot_states, const TensorView& handles,
    const TensorView& error_flag, std::uint32_t owner_sequence_index,
    std::uint32_t layer, std::uint64_t query_start_position,
    std::uint32_t key_token_count, float scale, std::uint32_t slot_count,
    std::int32_t owning_rank) {
  constexpr std::uint64_t kQueryHeads = 16;
  constexpr std::uint64_t kHeadDimension = 128;
  constexpr std::uint32_t kMaximumSequenceTokens = 40'960;
  if (owning_rank < 0 || slot_count == 0 ||
      slot_count > QwenKvSlotPool::kMaximumSlots ||
      layer >= QwenKvSlotPool::kLayerCount || key_token_count == 0 ||
      key_token_count > kMaximumSequenceTokens ||
      query.rank() != 3 || output.rank() != 3 || query.dim(0) == 0 ||
      query.dim(0) > 4096 || query_start_position >= key_token_count ||
      query.dim(0) > key_token_count - query_start_position ||
      !std::isfinite(scale) || scale <= 0.0F ||
      query.dim(1) != kQueryHeads || query.dim(2) != kHeadDimension ||
      output.dim(0) != query.dim(0) || output.dim(1) != kQueryHeads ||
      output.dim(2) != kHeadDimension) {
    return Status::InvalidArgument("Qwen paged GQA shape or scalar is invalid");
  }
  const std::int32_t device = query.device().index();
  Status valid = require_cuda_bf16(query, device);
  if (!valid.ok()) return valid;
  valid = require_cuda_bf16(output, device);
  if (!valid.ok()) return valid;
  const std::uint32_t handle_count =
      (key_token_count + QwenKvSlotPool::kTokensPerSlot - 1U) /
      QwenKvSlotPool::kTokensPerSlot;
  if (handle_count > slot_count) {
    return Status::InvalidArgument(
        "Qwen paged GQA handle count exceeds physical slot count");
  }
  auto kv_bytes = checked_mul_u64(slot_count,
                                  QwenKvSlotPool::kSlotPayloadBytes);
  auto state_bytes = checked_mul_u64(slot_count, sizeof(QwenKvSlotState));
  auto handle_bytes = checked_mul_u64(handle_count,
                                      sizeof(QwenKvBlockHandle));
  if (!kv_bytes.ok()) return kv_bytes.status();
  if (!state_bytes.ok()) return state_bytes.status();
  if (!handle_bytes.ok()) return handle_bytes.status();
  const std::array<const TensorView*, 4> raw_views{
      &kv_backing, &slot_states, &handles, &error_flag};
  const std::array<std::uint64_t, 4> raw_bytes{
      kv_bytes.value(), state_bytes.value(), handle_bytes.value(), 4};
  for (std::size_t index = 0; index < raw_views.size(); ++index) {
    valid = require_cuda_bytes(*raw_views[index], device, raw_bytes[index]);
    if (!valid.ok()) return valid;
  }
  auto manifest = qwen_bf16_kernel_manifest(
      QwenBf16Primitive::kPagedGqa, function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  valid = require_function(function, QwenBf16Primitive::kPagedGqa,
                           manifest.value());
  if (!valid.ok()) return valid;

  const std::array<const TensorView*, 6> views{
      &query, &output, &kv_backing, &slot_states, &handles, &error_flag};
  const std::array<KernelPointerOwnerClass, 6> owners{
      KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kKvState,
      KernelPointerOwnerClass::kKvState,
      KernelPointerOwnerClass::kKvState,
      KernelPointerOwnerClass::kWorkspace};
  const std::array<KernelPointerAccess, 6> accesses{
      KernelPointerAccess::kRead, KernelPointerAccess::kWrite,
      KernelPointerAccess::kRead, KernelPointerAccess::kRead,
      KernelPointerAccess::kRead, KernelPointerAccess::kAtomic};
  const std::array<std::uint64_t, 6> spans{
      query.byte_span(), output.byte_span(), kv_bytes.value(),
      state_bytes.value(), handle_bytes.value(), 4};
  const std::array<std::uint64_t, 6> alignments{2, 2, 2, 4, 4, 4};
  std::vector<KernelPointerContract> contracts;
  contracts.reserve(views.size());
  for (std::size_t ordinal = 0; ordinal < views.size(); ++ordinal) {
    auto contract = pointer_contract(
        std::string(manifest->parameter(ordinal).contract_id), accesses[ordinal],
        owners[ordinal], owning_rank, device, spans[ordinal],
        alignments[ordinal], false);
    if (!contract.ok()) return contract.status();
    contracts.push_back(std::move(contract).value());
  }
  auto packet = KernelArgumentPacket::Create(manifest.value(), contracts);
  if (!packet.ok()) return packet.status();
  for (std::size_t ordinal = 0; ordinal < views.size(); ++ordinal) {
    auto pointer = VerifiedDevicePointer::Create(
        *views[ordinal], contracts[ordinal], owners[ordinal], owning_rank,
        views[ordinal]->generation());
    if (!pointer.ok()) return pointer.status();
    valid = packet->set_device_pointer(ordinal, pointer.value());
    if (!valid.ok()) return valid;
  }
  valid = packet->set_u32(6, owner_sequence_index);
  if (!valid.ok()) return valid;
  valid = packet->set_u32(7, layer);
  if (!valid.ok()) return valid;
  valid = packet->set_u64(8, query_start_position);
  if (!valid.ok()) return valid;
  valid = packet->set_u32(9, static_cast<std::uint32_t>(query.dim(0)));
  if (!valid.ok()) return valid;
  valid = packet->set_u32(10, handle_count);
  if (!valid.ok()) return valid;
  valid = packet->set_u32(11, key_token_count);
  if (!valid.ok()) return valid;
  valid = packet->set_float32(12, scale);
  if (!valid.ok()) return valid;
  valid = packet->set_u32(13, slot_count);
  if (!valid.ok()) return valid;
  auto blocks = checked_mul_u64(query.dim(0), kQueryHeads);
  if (!blocks.ok() || *blocks > KernelLaunchGeometry::kMaximumGridX) {
    return Status::InvalidArgument("Qwen paged GQA grid exceeds device bound");
  }
  auto geometry = KernelLaunchGeometry::Create(
      static_cast<std::uint32_t>(*blocks), 1, 1, 1, 1, 1, 0);
  if (!geometry.ok()) return geometry.status();
  return QwenBf16DispatchPlan(QwenBf16Primitive::kPagedGqa, function,
                              geometry.value(), std::move(packet).value());
}

Status QwenBf16DispatchPlan::submit(KernelLaunchDriver& driver,
                                    DriverStreamHandle stream) {
  if (submitted_) {
    return Status::FailedPrecondition("Qwen BF16 dispatch plan was already submitted");
  }
  submitted_ = true;
  return submit_verified_kernel(driver, function_, geometry_, stream,
                                arguments_);
}

}  // namespace pih
