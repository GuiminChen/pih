#include "pih/model/qwen3_bf16_tap_source_plan.h"

#include <limits>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

Result<CudaCopyEndpoint> activation_source(
    const QwenNumericalTapCapture& capture,
    const QwenBf16TapBinding& binding,
    const QwenBf16ResourceSet& resources,
    std::uint64_t activation_first_position,
    std::uint64_t activation_rows,
    std::uint64_t owner_id) {
  auto view = resources.view(binding.output_slot);
  if (!view.ok()) return view.status();
  if (view->device().type() != DeviceType::kCuda ||
      view->device().index() != resources.owning_rank() ||
      view->generation() == 0 || view->dtype() != capture.dtype ||
      view->rank() < 2 || view->stride(view->rank() - 1) != 1) {
    return Status::FailedPrecondition(
        "Qwen tap activation source metadata is invalid");
  }
  auto element_bytes = dtype_size(capture.dtype);
  if (!element_bytes.ok()) return element_bytes.status();
  if (capture.request.position < activation_first_position) {
    return Status::InvalidArgument(
        "Qwen tap activation precedes the current step");
  }
  const bool logits =
      capture.request.point == QwenNumericalTapPoint::kLogits;
  if (logits) {
    auto last_position =
        checked_add_u64(activation_first_position, activation_rows - 1);
    if (!last_position.ok()) return last_position.status();
    if (capture.request.rows != 1 ||
        capture.request.position != *last_position || view->dim(0) != 1) {
      return Status::InvalidArgument(
          "Qwen logits tap must select the final step row");
    }
  }
  const std::uint64_t row = logits
                                ? 0
                                : capture.request.position -
                                      activation_first_position;
  const std::uint64_t rows = view->dim(0);
  if (capture.request.rows > rows || row > rows - capture.request.rows) {
    return Status::InvalidArgument(
        "Qwen tap activation exceeds the current step");
  }
  const std::uint64_t row_elements = view->stride(0);
  auto row_bytes = checked_mul_u64(row_elements, *element_bytes);
  if (!row_bytes.ok()) return row_bytes.status();
  auto expected = checked_mul_u64(capture.request.rows, *row_bytes);
  if (!expected.ok()) return expected.status();
  if (*expected != capture.size_bytes) {
    return Status::FailedPrecondition(
        "Qwen tap activation shape differs from its capture layout");
  }
  auto offset = checked_mul_u64(row, *row_bytes);
  if (!offset.ok()) return offset.status();
  auto end = checked_add_u64(*offset, capture.size_bytes);
  if (!end.ok()) return end.status();
  if (*end > view->byte_span()) {
    return Status::FailedPrecondition(
        "Qwen tap activation slice exceeds its owner");
  }
  return CudaCopyEndpoint{
      reinterpret_cast<std::uintptr_t>(view->data()), view->byte_span(),
      *offset, owner_id, view->generation(), CudaCopyMemoryType::kDevice,
      static_cast<std::uint32_t>(resources.owning_rank()),
      resources.owning_rank()};
}

Result<CudaCopyEndpoint> kv_source(
    const QwenNumericalTapCapture& capture,
    const QwenBf16TapBinding& binding,
    const QwenBf16ResourceSet& resources,
    const QwenKvBlockTable& block_table,
    std::span<const QwenKvSlotState> slot_states,
    std::uint64_t owner_id) {
  auto view = resources.view(QwenBf16ActivationSlot::kKvBacking);
  if (!view.ok()) return view.status();
  if (view->device().type() != DeviceType::kCuda ||
      view->device().index() != resources.owning_rank() ||
      view->generation() == 0 || view->dtype() != DType::kUInt8 ||
      capture.dtype != DType::kBFloat16 ||
      capture.size_bytes != QwenKvAddressMapper::kBytesPerToken ||
      binding.kv_component == QwenBf16TapKvComponent::kNotApplicable) {
    return Status::FailedPrecondition(
        "Qwen KV tap source metadata is invalid");
  }
  const auto handles = block_table.reserved_handles();
  const std::uint64_t handle_index =
      capture.request.position / QwenKvSlotPool::kTokensPerSlot;
  if (handle_index >= handles.size()) {
    return Status::InvalidArgument("Qwen KV tap position is not reserved");
  }
  const auto handle = handles[handle_index];
  if (handle.slot >= slot_states.size()) {
    return Status::FailedPrecondition("Qwen KV tap slot state is missing");
  }
  auto mapper = QwenKvAddressMapper::Create(
      static_cast<std::uint32_t>(slot_states.size()), view->byte_span());
  if (!mapper.ok()) return mapper.status();
  const auto plane = binding.kv_component == QwenBf16TapKvComponent::kKey
                         ? QwenKvPlane::kKey
                         : QwenKvPlane::kValue;
  auto address = mapper->map(
      handle, slot_states[handle.slot],
      block_table.descriptor().owner_sequence_index, capture.request.layer,
      plane,
      capture.request.position % QwenKvSlotPool::kTokensPerSlot, 0, 0);
  if (!address.ok()) return address.status();
  auto end = checked_add_u64(address->byte_offset, capture.size_bytes);
  if (!end.ok()) return end.status();
  if (*end > view->byte_span()) {
    return Status::FailedPrecondition("Qwen KV tap row exceeds its owner");
  }
  return CudaCopyEndpoint{
      reinterpret_cast<std::uintptr_t>(view->data()), view->byte_span(),
      address->byte_offset, owner_id, view->generation(),
      CudaCopyMemoryType::kDevice,
      static_cast<std::uint32_t>(resources.owning_rank()),
      resources.owning_rank()};
}

}  // namespace

Result<QwenBf16TapSourcePlan> QwenBf16TapSourcePlan::Create(
    const QwenNumericalTapPlan& taps,
    const QwenBf16TapBindingPlan& bindings,
    const QwenBf16ResourceSet& resources,
    const QwenKvBlockTable& block_table,
    std::span<const QwenKvSlotState> slot_states,
    std::uint64_t activation_first_position,
    std::uint64_t source_owner_id_base) {
  if (bindings.size() != taps.captures().size() || slot_states.empty() ||
      source_owner_id_base == 0 ||
      source_owner_id_base >
          std::numeric_limits<std::uint64_t>::max() - bindings.size()) {
    return Status::InvalidArgument("Qwen tap source plan identity is invalid");
  }
  auto hidden = resources.view(QwenBf16ActivationSlot::kHidden);
  if (!hidden.ok()) return hidden.status();
  if (hidden->rank() != 2 || hidden->dim(0) == 0) {
    return Status::FailedPrecondition(
        "Qwen tap source plan has no activation row authority");
  }
  const std::uint64_t activation_rows = hidden->dim(0);
  std::vector<CudaCopyEndpoint> sources;
  sources.reserve(bindings.size());
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    const auto& capture = taps.captures()[index];
    const auto& binding = bindings[index];
    if (binding.capture_index != index || binding.request != capture.request) {
      return Status::FailedPrecondition(
          "Qwen tap source binding differs from capture plan");
    }
    const auto owner_id = source_owner_id_base + index;
    const bool kv = binding.kv_component !=
                    QwenBf16TapKvComponent::kNotApplicable;
    auto source = kv ? kv_source(capture, binding, resources, block_table,
                                 slot_states, owner_id)
                     : activation_source(capture, binding, resources,
                                         activation_first_position,
                                         activation_rows, owner_id);
    if (!source.ok()) return source.status();
    sources.push_back(*source);
  }
  return QwenBf16TapSourcePlan(std::move(sources));
}

}  // namespace pih
