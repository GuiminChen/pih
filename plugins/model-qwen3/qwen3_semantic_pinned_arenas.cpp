#include "pih/model/qwen3_semantic_pinned_arenas.h"

namespace pih {
namespace {

bool valid_pinned(const Buffer& buffer, std::uint64_t expected_bytes) {
  return buffer.data() != nullptr && buffer.size_bytes() == expected_bytes &&
         buffer.generation() != 0 &&
         buffer.device().type() == DeviceType::kCpu &&
         buffer.device().index() == 0 &&
         reinterpret_cast<std::uintptr_t>(buffer.data()) %
                 QwenSemanticPinnedArenas::kAlignment ==
             0;
}

}  // namespace

Result<QwenSemanticPinnedArenas>
QwenSemanticPinnedArenas::AllocateVerified(
    const QwenKvSemanticObservationPlan& kv_plan,
    RegisteredPinnedAllocator& allocator,
    PinnedPlacementVerifier& verifier, std::int32_t owning_rank,
    std::int32_t numa_node) {
  if (owning_rank < 0 || numa_node < 0 || kv_plan.payload_bytes() == 0) {
    return Status::InvalidArgument(
        "Qwen semantic pinned arena placement is invalid");
  }
  auto logits = Buffer::Allocate(
      allocator, QwenSemanticObservationTransfer::kFinalLogitsBytes,
      kAlignment);
  if (!logits.ok()) return logits.status();
  auto kv = Buffer::Allocate(allocator, kv_plan.payload_bytes(), kAlignment);
  if (!kv.ok()) return kv.status();
  if (!valid_pinned(*logits,
                    QwenSemanticObservationTransfer::kFinalLogitsBytes) ||
      !valid_pinned(*kv, kv_plan.payload_bytes())) {
    return Status::FailedPrecondition(
        "Qwen semantic allocator returned an invalid pinned arena");
  }
  Status status =
      verifier.verify(logits->data(), logits->size_bytes(), numa_node);
  if (status.ok()) {
    status = verifier.verify(kv->data(), kv->size_bytes(), numa_node);
  }
  if (!status.ok()) return status;
  return QwenSemanticPinnedArenas(
      std::move(*logits), std::move(*kv), owning_rank, numa_node);
}

Result<CudaCopyEndpoint> QwenSemanticPinnedArenas::endpoint(
    const Buffer& buffer, std::uint64_t owner_id, std::uint32_t rank) const {
  if (owner_id == 0 || rank != static_cast<std::uint32_t>(owning_rank_)) {
    return Status::InvalidArgument(
        "Qwen semantic pinned endpoint identity is invalid");
  }
  return CudaCopyEndpoint{
      reinterpret_cast<std::uintptr_t>(buffer.data()), buffer.size_bytes(), 0,
      owner_id, buffer.generation(),
      CudaCopyMemoryType::kRegisteredPinnedHost, rank, numa_node_};
}

Result<CudaCopyEndpoint> QwenSemanticPinnedArenas::logits_endpoint(
    std::uint64_t owner_id, std::uint32_t rank) const {
  return endpoint(logits_, owner_id, rank);
}

Result<CudaCopyEndpoint> QwenSemanticPinnedArenas::kv_endpoint(
    std::uint64_t owner_id, std::uint32_t rank) const {
  return endpoint(kv_, owner_id, rank);
}

std::span<const std::byte> QwenSemanticPinnedArenas::logits() const noexcept {
  return {static_cast<const std::byte*>(logits_.data()),
          static_cast<std::size_t>(logits_.size_bytes())};
}

std::span<const std::byte> QwenSemanticPinnedArenas::kv() const noexcept {
  return {static_cast<const std::byte*>(kv_.data()),
          static_cast<std::size_t>(kv_.size_bytes())};
}

}  // namespace pih
