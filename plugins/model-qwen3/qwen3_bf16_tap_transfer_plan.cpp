#include "pih/model/qwen3_bf16_tap_transfer_plan.h"

namespace pih {

Result<QwenBf16TapTransferPlan> QwenBf16TapTransferPlan::Create(
    const QwenNumericalTapPlan& taps,
    std::span<const CudaCopyEndpoint> sources,
    const QwenNumericalTapArenas& arenas,
    QwenBf16TapTransferIdentity identity) {
  const bool valid_identity =
      identity.capture_generation != 0 &&
      identity.producer_plan_generation != 0 &&
      identity.snapshot_copy_plan_id != 0 &&
      identity.host_copy_plan_id != 0 &&
      identity.snapshot_copy_plan_id != identity.host_copy_plan_id &&
      identity.snapshot_event_generation != 0 &&
      identity.host_event_generation != 0 &&
      identity.snapshot_event_generation != identity.host_event_generation &&
      identity.primary_context_identity != 0 &&
      identity.execution_stream != 0 && identity.diagnostic_stream != 0 &&
      identity.device_arena_owner_id != 0 &&
      identity.pinned_arena_owner_id != 0;
  if (!valid_identity || sources.size() != taps.captures().size() ||
      arenas.arena_bytes() != taps.arena_bytes()) {
    return Status::InvalidArgument(
        "Qwen tap transfer plan identity is invalid");
  }

  std::vector<QwenNumericalTapTransfer> transfers;
  transfers.reserve(taps.captures().size());
  for (std::size_t index = 0; index < taps.captures().size(); ++index) {
    const auto& capture = taps.captures()[index];
    auto alignment = dtype_size(capture.dtype);
    if (!alignment.ok()) return alignment.status();
    auto device = arenas.device_endpoint(
        index, identity.device_arena_owner_id, identity.rank);
    if (!device.ok()) return device.status();
    auto pinned = arenas.pinned_endpoint(
        index, identity.pinned_arena_owner_id, identity.rank);
    if (!pinned.ok()) return pinned.status();
    auto snapshot = CudaTypedCopyPlan::Create(
        identity.snapshot_copy_plan_id, CudaCopyPurpose::kSameRankMove,
        CudaCopyKind::kDeviceToDevice, sources[index], *device,
        capture.size_bytes, *alignment, identity.primary_context_identity,
        identity.execution_stream, identity.snapshot_event_generation);
    if (!snapshot.ok()) return snapshot.status();
    auto host = CudaTypedCopyPlan::Create(
        identity.host_copy_plan_id, CudaCopyPurpose::kDiagnostic,
        CudaCopyKind::kDeviceToHost, *device, *pinned, capture.size_bytes,
        *alignment, identity.primary_context_identity,
        identity.diagnostic_stream, identity.host_event_generation);
    if (!host.ok()) return host.status();
    auto transfer = QwenNumericalTapTransfer::Create(
        index, identity.capture_generation, capture.size_bytes,
        identity.producer_plan_generation, std::move(*snapshot),
        std::move(*host));
    if (!transfer.ok()) return transfer.status();
    transfers.push_back(std::move(*transfer));
  }
  return QwenBf16TapTransferPlan(identity, std::move(transfers));
}

}  // namespace pih
