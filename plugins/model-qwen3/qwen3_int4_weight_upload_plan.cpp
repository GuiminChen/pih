#include "pih/model/qwen3_int4_weight_upload_plan.h"

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenInt4WeightUploadPlan> QwenInt4WeightUploadPlan::Create(
    const QwenInt4ArtifactLayout& layout,
    CudaCopyEndpoint canonical_file,
    CudaCopyEndpoint compact_device_pool,
    std::uintptr_t primary_context_identity,
    DriverStreamHandle stream,
    std::uint64_t completion_event_generation,
    std::uint64_t first_plan_id) {
  if (canonical_file.offset != 0 || compact_device_pool.offset != 0 ||
      canonical_file.allocation_bytes != layout.file_bytes() ||
      compact_device_pool.allocation_bytes != layout.logical_payload_bytes() ||
      canonical_file.memory_type !=
          CudaCopyMemoryType::kRegisteredPinnedHost ||
      compact_device_pool.memory_type != CudaCopyMemoryType::kDevice ||
      canonical_file.rank != compact_device_pool.rank ||
      canonical_file.owner_id == compact_device_pool.owner_id ||
      primary_context_identity == 0 || stream == 0 ||
      completion_event_generation == 0 || first_plan_id == 0) {
    return Status::InvalidArgument("Qwen INT4 upload identity is invalid");
  }
  auto final_plan_id = checked_add_u64(
      first_plan_id, layout.payload_record_count() - 1);
  if (!final_plan_id.ok()) return final_plan_id.status();
  std::vector<CudaTypedCopyPlan> copies;
  copies.reserve(layout.payload_record_count());
  std::uint64_t compact_offset = 0;
  for (const auto& record : layout.records()) {
    if (record.kind == QwenInt4ArtifactRecordKind::kAlias) continue;
    auto source = canonical_file;
    source.offset = record.file_offset;
    auto destination = compact_device_pool;
    destination.offset = compact_offset;
    auto copy = CudaTypedCopyPlan::Create(
        first_plan_id + copies.size(), CudaCopyPurpose::kWeight,
        CudaCopyKind::kHostToDevice, source, destination,
        record.logical_bytes, 256, primary_context_identity, stream,
        completion_event_generation);
    if (!copy.ok()) return copy.status();
    copies.push_back(std::move(*copy));
    auto next = checked_add_u64(compact_offset, record.logical_bytes);
    if (!next.ok()) return next.status();
    compact_offset = *next;
  }
  if (copies.size() != layout.payload_record_count() ||
      compact_offset != layout.logical_payload_bytes()) {
    return Status::InvalidArgument("Qwen INT4 upload extents drifted");
  }
  return QwenInt4WeightUploadPlan(std::move(copies),compact_offset);
}

Status QwenInt4WeightUploadPlan::submit(TypedCopyDriver& driver) {
  if (state_ != QwenInt4WeightUploadState::kPrepared) {
    return Status::FailedPrecondition("Qwen INT4 upload is not submit-ready");
  }
  for (; submitted_copies_ < copies_.size(); ++submitted_copies_) {
    const auto status = copies_[submitted_copies_].submit(driver);
    if (!status.ok()) {
      state_ = QwenInt4WeightUploadState::kPoisoned;
      return status;
    }
  }
  state_ = QwenInt4WeightUploadState::kSubmitted;
  return Status::Ok();
}

}  // namespace pih
