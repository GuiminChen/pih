#include "pih/model/deepseek_rope_table_device_resources.h"

#include "pih/core/checked_math.h"

namespace pih {
namespace {

bool needs_base(const DeepSeekStagePlan& stage) {
  return stage.layers.first_layer <= 1 || stage.owns_dspark;
}

bool needs_yarn(const DeepSeekStagePlan& stage) {
  return stage.layers.last_layer >= 2;
}

}  // namespace

Result<DeepSeekRopeTableDeviceResources>
DeepSeekRopeTableDeviceResources::AllocateDeferred(
    Allocator& allocator, DeepSeekStagePlan stage,
    std::uint32_t maximum_positions, std::uintptr_t stream,
    DeepSeekRopeTableOperations& operations,
    std::uint64_t context_identity, std::int32_t device_ordinal) {
  if (stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer > 42 || maximum_positions == 0 ||
      maximum_positions > 1048576 || stream == 0 ||
      context_identity == 0 || device_ordinal < 0) {
    return Status::InvalidArgument("DeepSeek RoPE table identity is invalid");
  }
  auto table_bytes = checked_mul_u64(maximum_positions, 64 * sizeof(float));
  if (!table_bytes.ok()) return table_bytes.status();
  auto cursor = checked_align_up_u64(sizeof(std::uint32_t), 256);
  if (!cursor.ok()) return cursor.status();
  const auto base_offset = needs_base(stage) ? *cursor : 0;
  if (needs_base(stage)) {
    cursor = checked_add_u64(*cursor, *table_bytes);
    if (!cursor.ok()) return cursor.status();
    cursor = checked_align_up_u64(*cursor, 256);
    if (!cursor.ok()) return cursor.status();
  }
  const auto yarn_offset = needs_yarn(stage) ? *cursor : 0;
  if (needs_yarn(stage)) {
    cursor = checked_add_u64(*cursor, *table_bytes);
    if (!cursor.ok()) return cursor.status();
  }
  auto total = checked_align_up_u64(*cursor, 256);
  if (!total.ok()) return total.status();
  auto backing = Buffer::Allocate(allocator, *total, 256);
  if (!backing.ok()) return backing.status();
  if (backing->data() == nullptr || backing->size_bytes() != *total ||
      backing->generation() == 0 ||
      backing->device().type() != DeviceType::kCuda ||
      backing->device().index() != device_ordinal) {
    return Status::FailedPrecondition(
        "DeepSeek RoPE allocator returned invalid backing");
  }
  const auto address = reinterpret_cast<std::uintptr_t>(backing->data());
  DeepSeekRopeTableDeviceView view{
      base_offset == 0 ? 0 : address + base_offset,
      yarn_offset == 0 ? 0 : address + yarn_offset, address};
  // Bootstrap can still fail after this allocation. Submit kernels only once
  // the engine owns the complete graph and can quarantine failed completion.
  return DeepSeekRopeTableDeviceResources(
      std::move(*backing), view, maximum_positions,
      context_identity, device_ordinal, operations, stream);
}

Status DeepSeekRopeTableDeviceResources::ensure_initialized() {
  if (initialized_) return Status::Ok();
  if (pending_ || operations_ == nullptr)
    return Status::FailedPrecondition("DeepSeek RoPE initialization is not retired");
  for (bool yarn : {false, true}) {
    const auto address = yarn ? view_.yarn_f32 : view_.base_f32;
    if (address == 0) continue;
    pending_ = true;
    const auto launched = operations_->initialize(
        {address, view_.error_flag_u32, stream_, maximum_positions_,
         64, 10000.0, yarn ? 16.0 : 1.0, 65536, 32, 1, yarn});
    const auto retired = retire_pending();
    if (!retired.ok()) return retired;
    if (!launched.ok()) return launched;
  }
  initialized_ = true;
  return Status::Ok();
}

Status DeepSeekRopeTableDeviceResources::retire_pending() noexcept {
  if (!pending_) return Status::Ok();
  try {
    const auto status = operations_->synchronize(stream_);
    if (!status.ok()) return status;
    pending_ = false;
    return Status::Ok();
  } catch (...) {
    return Status::Internal("DeepSeek RoPE synchronization failed");
  }
}

}  // namespace pih
