#include "pih/model/qwen3_bf16_packed_dispatch_plan.h"

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "pih/core/checked_math.h"
#include "pih/model/qwen3_kv_slot_pool.h"

namespace pih {
namespace {

constexpr std::uint64_t kMaximumBatchTokens = 4096;
constexpr std::uint64_t kHidden = 1024;
constexpr std::uint64_t kHeadDimension = 128;
constexpr std::uint64_t kKvHeads = 8;
constexpr std::uint64_t kQueryHeads = 16;
constexpr std::uint64_t kKvBytesPerSlot = 1'835'008;

struct PointerSpec final {
  const TensorView* view;
  KernelPointerAccess access;
  KernelPointerOwnerClass owner;
  std::uint64_t bytes;
  std::uint64_t alignment;
};

Status require_contiguous(const TensorView& view) {
  if (view.num_elements() == 0) return Status::InvalidArgument("empty packed tensor");
  std::uint64_t stride = 1;
  for (std::size_t axis = view.rank(); axis > 0; --axis) {
    const auto index = axis - 1;
    if (view.stride(index) != stride) {
      return Status::InvalidArgument("packed tensor must be contiguous");
    }
    auto next = checked_mul_u64(stride, view.dim(index));
    if (!next.ok()) return next.status();
    stride = *next;
  }
  return Status::Ok();
}

Status require_view(const TensorView& view, DType dtype, std::int32_t device,
                    std::uint64_t minimum_bytes) {
  if (view.dtype() != dtype || view.device().type() != DeviceType::kCuda ||
      view.device().index() != device || view.generation() == 0 ||
      view.byte_span() < minimum_bytes) {
    return Status::InvalidArgument("packed tensor dtype, extent, or identity is invalid");
  }
  return require_contiguous(view);
}

Status require_function(const ResolvedKernelFunction& function,
                        QwenBf16PackedPrimitive primitive,
                        const KernelSignatureManifest& manifest) {
  auto symbol = qwen_bf16_packed_kernel_symbol(primitive);
  if (!symbol.ok()) return symbol.status();
  if (function.handle == 0 || function.symbol != *symbol ||
      function.logical_id != manifest.logical_id() ||
      function.selected_cubin_sha256 != manifest.selected_cubin_sha256() ||
      function.parameter_abi_sha256 != manifest.parameter_abi_sha256()) {
    return Status::InvalidArgument("packed Qwen function identity differs from manifest");
  }
  return Status::Ok();
}

Result<KernelArgumentPacket> pointer_packet(
    const KernelSignatureManifest& manifest, std::span<const PointerSpec> specs,
    std::int32_t owning_rank, std::int32_t device) {
  std::vector<KernelPointerContract> contracts;
  contracts.reserve(specs.size());
  for (std::size_t i = 0; i < specs.size(); ++i) {
    auto contract = KernelPointerContract::Create(
        std::string(manifest.parameter(i).contract_id), specs[i].access,
        specs[i].owner, owning_rank, device, specs[i].bytes,
        specs[i].alignment, 0, false);
    if (!contract.ok()) return contract.status();
    contracts.push_back(std::move(*contract));
  }
  auto packet = KernelArgumentPacket::Create(manifest, contracts);
  if (!packet.ok()) return packet.status();
  for (std::size_t i = 0; i < specs.size(); ++i) {
    auto pointer = VerifiedDevicePointer::Create(
        *specs[i].view, contracts[i], specs[i].owner, owning_rank,
        specs[i].view->generation());
    if (!pointer.ok()) return pointer.status();
    auto status = packet->set_device_pointer(i, *pointer);
    if (!status.ok()) return status;
  }
  return packet;
}

Result<KernelLaunchGeometry> elementwise(std::uint64_t elements) {
  auto rounded = checked_add_u64(elements, 255);
  if (!rounded.ok()) return rounded.status();
  const auto blocks = *rounded / 256;
  if (blocks == 0 || blocks > KernelLaunchGeometry::kMaximumGridX) {
    return Status::ResourceExhausted("packed launch exceeds CUDA grid.x");
  }
  return KernelLaunchGeometry::Create(static_cast<std::uint32_t>(blocks), 1, 1,
                                      256, 1, 1, 0);
}

Result<std::uint64_t> times(std::uint64_t first, std::uint64_t second) {
  return checked_mul_u64(first, second);
}

}  // namespace

Result<QwenBf16PackedDispatchPlan> QwenBf16PackedDispatchPlan::CreateEmbedding(
    const ResolvedKernelFunction& function, const TensorView& table,
    const TensorView& token_ids, const TensorView& output,
    std::uint64_t real_token_count, std::int32_t owning_rank) {
  if (owning_rank < 0 || real_token_count == 0 ||
      real_token_count > kMaximumBatchTokens || table.rank() != 2 ||
      table.dim(0) == 0 || table.dim(1) != kHidden || output.rank() != 2 ||
      output.dim(1) != kHidden || real_token_count > output.dim(0)) {
    return Status::InvalidArgument("packed embedding geometry is invalid");
  }
  const auto device = table.device().index();
  auto token_bytes = times(output.dim(0), 4);
  auto output_bytes = times(real_token_count * kHidden, 2);
  if (!token_bytes.ok()) return token_bytes.status();
  if (!output_bytes.ok()) return output_bytes.status();
  Status status = require_view(table, DType::kBFloat16, device, table.byte_span());
  if (!status.ok()) return status;
  status = require_view(token_ids, DType::kUInt8, device, *token_bytes);
  if (!status.ok()) return status;
  status = require_view(output, DType::kBFloat16, device, *output_bytes);
  if (!status.ok()) return status;
  auto manifest = qwen_bf16_packed_kernel_manifest(
      QwenBf16PackedPrimitive::kEmbedding, function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  status = require_function(function, QwenBf16PackedPrimitive::kEmbedding, *manifest);
  if (!status.ok()) return status;
  const std::array specs{
      PointerSpec{&table, KernelPointerAccess::kRead, KernelPointerOwnerClass::kWeight,
                  table.byte_span(), 2},
      PointerSpec{&token_ids, KernelPointerAccess::kRead,
                  KernelPointerOwnerClass::kActivation, *token_bytes, 4},
      PointerSpec{&output, KernelPointerAccess::kWrite,
                  KernelPointerOwnerClass::kActivation, *output_bytes, 2}};
  auto packet = pointer_packet(*manifest, specs, owning_rank, device);
  if (!packet.ok()) return packet.status();
  if (!(status = packet->set_u64(3, real_token_count)).ok()) return status;
  if (!(status = packet->set_u64(4, table.dim(0))).ok()) return status;
  if (!(status = packet->set_u64(5, kHidden)).ok()) return status;
  auto elements = times(real_token_count, kHidden);
  if (!elements.ok()) return elements.status();
  auto geometry = elementwise(*elements);
  if (!geometry.ok()) return geometry.status();
  return QwenBf16PackedDispatchPlan(QwenBf16PackedPrimitive::kEmbedding,
                                    function, *geometry, std::move(*packet));
}

Result<QwenBf16PackedDispatchPlan> QwenBf16PackedDispatchPlan::CreateRopeAngles(
    const ResolvedKernelFunction& function, const TensorView& positions,
    const TensorView& cosine, const TensorView& sine,
    std::uint64_t real_token_count, std::int32_t owning_rank) {
  if (owning_rank < 0 || real_token_count == 0 ||
      real_token_count > kMaximumBatchTokens || cosine.rank() != 2 ||
      sine.rank() != 2 || cosine.dim(1) != 64 || sine.dim(1) != 64 ||
      cosine.dim(0) != sine.dim(0) || real_token_count > cosine.dim(0)) {
    return Status::InvalidArgument("packed RoPE angle geometry is invalid");
  }
  const auto device = positions.device().index();
  auto position_bytes = times(cosine.dim(0), 8);
  auto angle_bytes = times(real_token_count * 64, 4);
  if (!position_bytes.ok()) return position_bytes.status();
  if (!angle_bytes.ok()) return angle_bytes.status();
  Status status = require_view(positions, DType::kUInt8, device, *position_bytes);
  if (!status.ok()) return status;
  status = require_view(cosine, DType::kFloat32, device, *angle_bytes);
  if (!status.ok()) return status;
  status = require_view(sine, DType::kFloat32, device, *angle_bytes);
  if (!status.ok()) return status;
  auto manifest = qwen_bf16_packed_kernel_manifest(
      QwenBf16PackedPrimitive::kRopeAngles, function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  status = require_function(function, QwenBf16PackedPrimitive::kRopeAngles, *manifest);
  if (!status.ok()) return status;
  const std::array specs{
      PointerSpec{&positions, KernelPointerAccess::kRead,
                  KernelPointerOwnerClass::kActivation, *position_bytes, 8},
      PointerSpec{&cosine, KernelPointerAccess::kWrite,
                  KernelPointerOwnerClass::kWorkspace, *angle_bytes, 4},
      PointerSpec{&sine, KernelPointerAccess::kWrite,
                  KernelPointerOwnerClass::kWorkspace, *angle_bytes, 4}};
  auto packet = pointer_packet(*manifest, specs, owning_rank, device);
  if (!packet.ok()) return packet.status();
  if (!(status = packet->set_u64(3, real_token_count)).ok()) return status;
  if (!(status = packet->set_u64(4, kHeadDimension)).ok()) return status;
  auto geometry = elementwise(real_token_count * 64);
  if (!geometry.ok()) return geometry.status();
  return QwenBf16PackedDispatchPlan(QwenBf16PackedPrimitive::kRopeAngles,
                                    function, *geometry, std::move(*packet));
}

Result<QwenBf16PackedDispatchPlan> QwenBf16PackedDispatchPlan::CreateKvAppend(
    const ResolvedKernelFunction& function, const TensorView& key_input,
    const TensorView& value_input, const TensorView& kv_backing,
    const TensorView& slot_states, const TensorView& handles,
    const TensorView& token_offsets, const TensorView& request_index,
    const TensorView& owner_sequence_indices, const TensorView& error_flag,
    std::uint32_t layer, std::uint64_t token_count,
    std::uint32_t sequence_count, std::uint32_t slot_count,
    std::int32_t owning_rank) {
  if (owning_rank < 0 || layer >= 28 || token_count == 0 ||
      token_count > kMaximumBatchTokens || sequence_count == 0 ||
      sequence_count > token_count || slot_count == 0 ||
      slot_count > QwenKvSlotPool::kMaximumSlots || key_input.rank() != 3 ||
      value_input.rank() != 3 || token_count > key_input.dim(0) ||
      key_input.dim(0) != value_input.dim(0) || key_input.dim(1) != kKvHeads ||
      value_input.dim(1) != kKvHeads || key_input.dim(2) != kHeadDimension ||
      value_input.dim(2) != kHeadDimension) {
    return Status::InvalidArgument("packed KV append geometry is invalid");
  }
  const auto device = key_input.device().index();
  const auto activation_bytes = token_count * kKvHeads * kHeadDimension * 2;
  const auto backing_bytes = static_cast<std::uint64_t>(slot_count) * kKvBytesPerSlot;
  const std::array<const TensorView*, 9> views{&key_input, &value_input, &kv_backing,
      &slot_states, &handles, &token_offsets, &request_index,
      &owner_sequence_indices, &error_flag};
  const std::array<DType, 9> dtypes{DType::kBFloat16, DType::kBFloat16,
      DType::kUInt8, DType::kUInt8, DType::kUInt8, DType::kUInt8,
      DType::kUInt8, DType::kUInt8, DType::kUInt8};
  const std::array<std::uint64_t, 9> bytes{activation_bytes, activation_bytes,
      backing_bytes, static_cast<std::uint64_t>(slot_count) * 16,
      token_count * 8, token_count * 2, token_count * 4,
      static_cast<std::uint64_t>(sequence_count) * 4, 4};
  for (std::size_t i = 0; i < views.size(); ++i) {
    auto status = require_view(*views[i], dtypes[i], device, bytes[i]);
    if (!status.ok()) return status;
  }
  auto manifest = qwen_bf16_packed_kernel_manifest(
      QwenBf16PackedPrimitive::kKvAppend, function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  auto status = require_function(function, QwenBf16PackedPrimitive::kKvAppend,
                                 *manifest);
  if (!status.ok()) return status;
  const std::array accesses{KernelPointerAccess::kRead, KernelPointerAccess::kRead,
      KernelPointerAccess::kWrite, KernelPointerAccess::kRead,
      KernelPointerAccess::kRead, KernelPointerAccess::kRead,
      KernelPointerAccess::kRead, KernelPointerAccess::kRead,
      KernelPointerAccess::kAtomic};
  const std::array owners{KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kActivation, KernelPointerOwnerClass::kKvState,
      KernelPointerOwnerClass::kKvState, KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kActivation, KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kActivation, KernelPointerOwnerClass::kKvState};
  const std::array<std::uint64_t, 9> alignments{2, 2, 2, 16, 8, 2, 4, 4, 4};
  std::array<PointerSpec, 9> specs{};
  for (std::size_t i = 0; i < specs.size(); ++i)
    specs[i] = {views[i], accesses[i], owners[i], bytes[i], alignments[i]};
  auto packet = pointer_packet(*manifest, specs, owning_rank, device);
  if (!packet.ok()) return packet.status();
  if (!(status = packet->set_u32(9, layer)).ok()) return status;
  if (!(status = packet->set_u64(10, token_count)).ok()) return status;
  if (!(status = packet->set_u32(11, sequence_count)).ok()) return status;
  if (!(status = packet->set_u32(12, slot_count)).ok()) return status;
  auto geometry = elementwise(token_count * 2 * kKvHeads * kHeadDimension);
  if (!geometry.ok()) return geometry.status();
  return QwenBf16PackedDispatchPlan(QwenBf16PackedPrimitive::kKvAppend,
                                    function, *geometry, std::move(*packet));
}

Result<QwenBf16PackedDispatchPlan> QwenBf16PackedDispatchPlan::CreatePagedGqa(
    const ResolvedKernelFunction& function, const TensorView& query,
    const TensorView& output, const TensorView& kv_backing,
    const TensorView& slot_states, const TensorView& handles,
    const TensorView& visible_handle_offsets, const TensorView& request_index,
    const TensorView& query_start_offsets, const TensorView& key_token_counts,
    const TensorView& owner_sequence_indices, const TensorView& error_flag,
    std::uint32_t layer, std::uint32_t query_count,
    std::uint32_t sequence_count, float scale, std::uint32_t slot_count,
    std::int32_t owning_rank) {
  if (owning_rank < 0 || layer >= 28 || query_count == 0 ||
      query_count > kMaximumBatchTokens || sequence_count == 0 ||
      sequence_count > query_count || slot_count == 0 ||
      slot_count > QwenKvSlotPool::kMaximumSlots || !std::isfinite(scale) ||
      scale <= 0 || query.rank() != 3 || output.rank() != 3 ||
      query_count > query.dim(0) || query.dim(0) != output.dim(0) ||
      query.dim(1) != kQueryHeads || output.dim(1) != kQueryHeads ||
      query.dim(2) != kHeadDimension || output.dim(2) != kHeadDimension ||
      handles.byte_span() == 0 || handles.byte_span() % 8 != 0) {
    return Status::InvalidArgument("packed paged GQA geometry is invalid");
  }
  const auto device = query.device().index();
  const auto activation_bytes = static_cast<std::uint64_t>(query_count) *
                                kQueryHeads * kHeadDimension * 2;
  const auto backing_bytes = static_cast<std::uint64_t>(slot_count) * kKvBytesPerSlot;
  const auto prefix_bytes = static_cast<std::uint64_t>(sequence_count + 1U) * 4;
  const auto sequence_bytes = static_cast<std::uint64_t>(sequence_count) * 4;
  const std::array<const TensorView*, 11> views{&query, &output, &kv_backing,
      &slot_states, &handles, &visible_handle_offsets, &request_index,
      &query_start_offsets, &key_token_counts, &owner_sequence_indices,
      &error_flag};
  const std::array<DType, 11> dtypes{DType::kBFloat16, DType::kBFloat16,
      DType::kUInt8, DType::kUInt8, DType::kUInt8, DType::kUInt8,
      DType::kUInt8, DType::kUInt8, DType::kUInt8, DType::kUInt8,
      DType::kUInt8};
  const std::array<std::uint64_t, 11> bytes{activation_bytes, activation_bytes,
      backing_bytes, static_cast<std::uint64_t>(slot_count) * 16,
      handles.byte_span(), prefix_bytes, static_cast<std::uint64_t>(query_count) * 4,
      prefix_bytes, sequence_bytes, sequence_bytes, 4};
  for (std::size_t i = 0; i < views.size(); ++i) {
    auto status = require_view(*views[i], dtypes[i], device, bytes[i]);
    if (!status.ok()) return status;
  }
  auto manifest = qwen_bf16_packed_kernel_manifest(
      QwenBf16PackedPrimitive::kPagedGqa, function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  auto status = require_function(function, QwenBf16PackedPrimitive::kPagedGqa,
                                 *manifest);
  if (!status.ok()) return status;
  const std::array accesses{KernelPointerAccess::kRead, KernelPointerAccess::kWrite,
      KernelPointerAccess::kRead, KernelPointerAccess::kRead,
      KernelPointerAccess::kRead, KernelPointerAccess::kRead,
      KernelPointerAccess::kRead, KernelPointerAccess::kRead,
      KernelPointerAccess::kRead, KernelPointerAccess::kRead,
      KernelPointerAccess::kAtomic};
  const std::array owners{KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kActivation, KernelPointerOwnerClass::kKvState,
      KernelPointerOwnerClass::kKvState, KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kActivation, KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kActivation, KernelPointerOwnerClass::kActivation,
      KernelPointerOwnerClass::kActivation, KernelPointerOwnerClass::kKvState};
  const std::array<std::uint64_t, 11> alignments{2, 2, 2, 16, 8, 4, 4, 4, 4, 4, 4};
  std::array<PointerSpec, 11> specs{};
  for (std::size_t i = 0; i < specs.size(); ++i)
    specs[i] = {views[i], accesses[i], owners[i], bytes[i], alignments[i]};
  auto packet = pointer_packet(*manifest, specs, owning_rank, device);
  if (!packet.ok()) return packet.status();
  if (!(status = packet->set_u32(11, layer)).ok()) return status;
  if (!(status = packet->set_u32(12, query_count)).ok()) return status;
  if (!(status = packet->set_u32(13, sequence_count)).ok()) return status;
  if (!(status = packet->set_float32(14, scale)).ok()) return status;
  if (!(status = packet->set_u32(15, slot_count)).ok()) return status;
  const auto blocks = static_cast<std::uint64_t>(query_count) * kQueryHeads;
  auto geometry = KernelLaunchGeometry::Create(static_cast<std::uint32_t>(blocks),
                                                1, 1, 1, 1, 1, 0);
  if (!geometry.ok()) return geometry.status();
  return QwenBf16PackedDispatchPlan(QwenBf16PackedPrimitive::kPagedGqa,
                                    function, *geometry, std::move(*packet));
}

Result<QwenBf16PackedDispatchPlan>
QwenBf16PackedDispatchPlan::CreateSampleHidden(
    const ResolvedKernelFunction& function, const TensorView& input,
    const TensorView& sample_rows, const TensorView& output,
    const TensorView& error_flag, std::uint32_t sample_count,
    std::uint32_t packed_token_count, std::int32_t owning_rank) {
  if (owning_rank < 0 || sample_count == 0 ||
      sample_count > packed_token_count ||
      packed_token_count > kMaximumBatchTokens || input.rank() != 2 ||
      output.rank() != 2 || input.dim(0) != packed_token_count ||
      input.dim(1) != kHidden || output.dim(0) != sample_count ||
      output.dim(1) != kHidden) {
    return Status::InvalidArgument("packed sample hidden geometry is invalid");
  }
  const auto device = input.device().index();
  const auto input_bytes = static_cast<std::uint64_t>(packed_token_count) *
                           kHidden * 2;
  const auto row_bytes = static_cast<std::uint64_t>(sample_count) * 4;
  const auto output_bytes = static_cast<std::uint64_t>(sample_count) *
                            kHidden * 2;
  Status status = require_view(input, DType::kBFloat16, device, input_bytes);
  if (!status.ok()) return status;
  status = require_view(sample_rows, DType::kUInt8, device, row_bytes);
  if (!status.ok()) return status;
  status = require_view(output, DType::kBFloat16, device, output_bytes);
  if (!status.ok()) return status;
  status = require_view(error_flag, DType::kUInt8, device, 4);
  if (!status.ok()) return status;
  auto manifest = qwen_bf16_packed_kernel_manifest(
      QwenBf16PackedPrimitive::kSampleHidden,
      function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  status = require_function(function, QwenBf16PackedPrimitive::kSampleHidden,
                            *manifest);
  if (!status.ok()) return status;
  const std::array specs{
      PointerSpec{&input, KernelPointerAccess::kRead,
                  KernelPointerOwnerClass::kActivation, input_bytes, 2},
      PointerSpec{&sample_rows, KernelPointerAccess::kRead,
                  KernelPointerOwnerClass::kActivation, row_bytes, 4},
      PointerSpec{&output, KernelPointerAccess::kWrite,
                  KernelPointerOwnerClass::kWorkspace, output_bytes, 2},
      PointerSpec{&error_flag, KernelPointerAccess::kAtomic,
                  KernelPointerOwnerClass::kKvState, 4, 4}};
  auto packet = pointer_packet(*manifest, specs, owning_rank, device);
  if (!packet.ok()) return packet.status();
  if (!(status = packet->set_u32(4, sample_count)).ok()) return status;
  if (!(status = packet->set_u32(5, packed_token_count)).ok()) return status;
  if (!(status = packet->set_u32(6, static_cast<std::uint32_t>(kHidden))).ok())
    return status;
  auto geometry = elementwise(static_cast<std::uint64_t>(sample_count) * kHidden);
  if (!geometry.ok()) return geometry.status();
  return QwenBf16PackedDispatchPlan(QwenBf16PackedPrimitive::kSampleHidden,
                                    function, *geometry, std::move(*packet));
}

Result<QwenBf16PackedDispatchPlan>
QwenBf16PackedDispatchPlan::CreateGreedyArgmax(
    const ResolvedKernelFunction& function, const TensorView& logits,
    const TensorView& sampled_token_ids, const TensorView& error_flag,
    std::uint32_t sample_count, std::int32_t owning_rank) {
  constexpr std::uint32_t kVocabulary = 151936;
  if (owning_rank < 0 || sample_count == 0 ||
      sample_count > kMaximumBatchTokens || logits.rank() != 2 ||
      logits.dim(0) != sample_count || logits.dim(1) != kVocabulary) {
    return Status::InvalidArgument("packed greedy argmax geometry is invalid");
  }
  const auto device = logits.device().index();
  const auto logit_bytes = static_cast<std::uint64_t>(sample_count) *
                           kVocabulary * 4;
  const auto result_bytes = static_cast<std::uint64_t>(sample_count) * 4;
  Status status = require_view(logits, DType::kFloat32, device, logit_bytes);
  if (!status.ok()) return status;
  status = require_view(sampled_token_ids, DType::kUInt8, device, result_bytes);
  if (!status.ok()) return status;
  status = require_view(error_flag, DType::kUInt8, device, 4);
  if (!status.ok()) return status;
  auto manifest = qwen_bf16_packed_kernel_manifest(
      QwenBf16PackedPrimitive::kGreedyArgmax,
      function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  status = require_function(function, QwenBf16PackedPrimitive::kGreedyArgmax,
                            *manifest);
  if (!status.ok()) return status;
  const std::array specs{
      PointerSpec{&logits, KernelPointerAccess::kRead,
                  KernelPointerOwnerClass::kWorkspace, logit_bytes, 4},
      PointerSpec{&sampled_token_ids, KernelPointerAccess::kWrite,
                  KernelPointerOwnerClass::kActivation, result_bytes, 4},
      PointerSpec{&error_flag, KernelPointerAccess::kAtomic,
                  KernelPointerOwnerClass::kKvState, 4, 4}};
  auto packet = pointer_packet(*manifest, specs, owning_rank, device);
  if (!packet.ok()) return packet.status();
  if (!(status = packet->set_u32(3, sample_count)).ok()) return status;
  if (!(status = packet->set_u32(4, kVocabulary)).ok()) return status;
  auto geometry = KernelLaunchGeometry::Create(sample_count, 1, 1, 256, 1, 1, 0);
  if (!geometry.ok()) return geometry.status();
  return QwenBf16PackedDispatchPlan(QwenBf16PackedPrimitive::kGreedyArgmax,
                                    function, *geometry, std::move(*packet));
}

Result<QwenBf16PackedDispatchPlan>
QwenBf16PackedDispatchPlan::CreateSampler(
    const ResolvedKernelFunction& function, const TensorView& logits,
    const TensorView& sampling_descriptors,
    const TensorView& sample_sequence_indices,
    const TensorView& workspace_ids, const TensorView& sampled_token_ids,
    const TensorView& selected_logprobs, const TensorView& rng_words,
    const TensorView& top_token_ids, const TensorView& top_logprobs,
    const TensorView& top_counts, const TensorView& error_flag,
    std::uint32_t sample_count, std::uint32_t sequence_count,
    std::int32_t owning_rank) {
  constexpr std::uint32_t kVocabulary = 151936;
  constexpr std::uint64_t kDescriptorBytes = 40;
  constexpr std::uint64_t kTopCount = 20;
  if (owning_rank < 0 || sample_count == 0 ||
      sample_count > kMaximumBatchTokens || sequence_count == 0 ||
      sequence_count > kMaximumBatchTokens || sample_count > sequence_count ||
      logits.rank() != 2 || logits.dim(0) != sample_count ||
      logits.dim(1) != kVocabulary) {
    return Status::InvalidArgument("packed sampler geometry is invalid");
  }
  const auto device = logits.device().index();
  const std::uint64_t logit_bytes =
      static_cast<std::uint64_t>(sample_count) * kVocabulary * 4;
  const std::uint64_t descriptor_bytes =
      static_cast<std::uint64_t>(sequence_count) * kDescriptorBytes;
  const std::uint64_t per_sample_bytes =
      static_cast<std::uint64_t>(sample_count) * 4;
  const std::uint64_t top_bytes = per_sample_bytes * kTopCount;
  const std::array<std::pair<const TensorView*, std::uint64_t>, 10> views{{
      {&sampling_descriptors, descriptor_bytes},
      {&sample_sequence_indices, per_sample_bytes},
      {&workspace_ids, logit_bytes}, {&sampled_token_ids, per_sample_bytes},
      {&selected_logprobs, per_sample_bytes}, {&rng_words, per_sample_bytes},
      {&top_token_ids, top_bytes}, {&top_logprobs, top_bytes},
      {&top_counts, per_sample_bytes}, {&error_flag, 4}}};
  auto status = require_view(logits, DType::kFloat32, device, logit_bytes);
  if (!status.ok()) return status;
  for (const auto& [view, bytes] : views) {
    status = require_view(*view, DType::kUInt8, device, bytes);
    if (!status.ok()) return status;
  }
  auto manifest = qwen_bf16_packed_kernel_manifest(
      QwenBf16PackedPrimitive::kSampler, function.selected_cubin_sha256);
  if (!manifest.ok()) return manifest.status();
  status = require_function(function, QwenBf16PackedPrimitive::kSampler,
                            *manifest);
  if (!status.ok()) return status;
  const std::array specs{
      PointerSpec{&logits, KernelPointerAccess::kReadWrite,
                  KernelPointerOwnerClass::kWorkspace, logit_bytes, 4},
      PointerSpec{&sampling_descriptors, KernelPointerAccess::kRead,
                  KernelPointerOwnerClass::kWorkspace, descriptor_bytes, 8},
      PointerSpec{&sample_sequence_indices, KernelPointerAccess::kRead,
                  KernelPointerOwnerClass::kWorkspace, per_sample_bytes, 4},
      PointerSpec{&workspace_ids, KernelPointerAccess::kWrite,
                  KernelPointerOwnerClass::kWorkspace, logit_bytes, 4},
      PointerSpec{&sampled_token_ids, KernelPointerAccess::kWrite,
                  KernelPointerOwnerClass::kActivation, per_sample_bytes, 4},
      PointerSpec{&selected_logprobs, KernelPointerAccess::kWrite,
                  KernelPointerOwnerClass::kActivation, per_sample_bytes, 4},
      PointerSpec{&rng_words, KernelPointerAccess::kWrite,
                  KernelPointerOwnerClass::kActivation, per_sample_bytes, 4},
      PointerSpec{&top_token_ids, KernelPointerAccess::kWrite,
                  KernelPointerOwnerClass::kActivation, top_bytes, 4},
      PointerSpec{&top_logprobs, KernelPointerAccess::kWrite,
                  KernelPointerOwnerClass::kActivation, top_bytes, 4},
      PointerSpec{&top_counts, KernelPointerAccess::kWrite,
                  KernelPointerOwnerClass::kActivation, per_sample_bytes, 4},
      PointerSpec{&error_flag, KernelPointerAccess::kAtomic,
                  KernelPointerOwnerClass::kKvState, 4, 4}};
  auto packet = pointer_packet(*manifest, specs, owning_rank, device);
  if (!packet.ok()) return packet.status();
  if (!(status = packet->set_u32(11, sample_count)).ok()) return status;
  if (!(status = packet->set_u32(12, sequence_count)).ok()) return status;
  if (!(status = packet->set_u32(13, kVocabulary)).ok()) return status;
  auto geometry = KernelLaunchGeometry::Create(sample_count, 1, 1, 1, 1, 1, 0);
  if (!geometry.ok()) return geometry.status();
  return QwenBf16PackedDispatchPlan(QwenBf16PackedPrimitive::kSampler,
                                    function, *geometry, std::move(*packet));
}

Status QwenBf16PackedDispatchPlan::submit(KernelLaunchDriver& driver,
                                         DriverStreamHandle stream) {
  if (submitted_) return Status::FailedPrecondition("packed dispatch already submitted");
  submitted_ = true;
  return submit_verified_kernel(driver, function_, geometry_, stream, arguments_);
}

}  // namespace pih
