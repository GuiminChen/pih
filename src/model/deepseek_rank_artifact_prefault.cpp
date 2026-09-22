#include "pih/model/deepseek_rank_artifact_prefault.h"

#include <utility>

#include "pih/core/canonical_hash.h"
#include "pih/core/checked_math.h"
#include "pih/model/deepseek_stage_mapped_inventory.h"

namespace pih {
namespace {

Result<Sha256Digest> compile_receipt_root(
    const DeepSeekRankMaterializationAdmission& admission,
    const DeepSeekRankArtifactMappingOwner& mapping_owner,
    const DeepSeekRankArtifactPrefaultLayout& layout,
    std::uint64_t resident_before, std::uint64_t resident_after,
    std::uint64_t touched_pages,
    const DeepSeekRankArtifactPrefaultResourceSnapshot& before,
    const DeepSeekRankArtifactPrefaultResourceSnapshot& after) {
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-prefault-receipt:v1", 32);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u64(1, admission.engine_epoch());
  if (status.ok()) status = builder->add_u64(2, admission.worker_generation());
  if (status.ok()) status = builder->add_u32(3, admission.world_size());
  if (status.ok()) status = builder->add_u32(4, admission.rank());
  if (status.ok()) status = builder->add_hash(5, admission.grant_root());
  if (status.ok()) {
    status = builder->add_hash(6, mapping_owner.mapping_owner_root());
  }
  if (status.ok()) status = builder->add_hash(7, layout.mapping_plan_root);
  if (status.ok()) status = builder->add_hash(8, layout.layout_root);
  if (status.ok()) status = builder->add_u64(9, layout.page_bytes);
  if (status.ok()) status = builder->add_u32(10, layout.interval_count);
  if (status.ok()) {
    status = builder->add_u64(11, layout.mapped_interval_bytes);
  }
  if (status.ok()) {
    status = builder->add_u64(12, layout.selected_page_union_bytes);
  }
  if (status.ok()) status = builder->add_u64(13, resident_before);
  if (status.ok()) status = builder->add_u64(14, resident_after);
  if (status.ok()) status = builder->add_u64(15, touched_pages);
  if (status.ok()) status = builder->add_u64(16, before.monotonic_ns);
  if (status.ok()) status = builder->add_u64(17, after.monotonic_ns);
  if (status.ok()) status = builder->add_u64(18, before.major_fault_count);
  if (status.ok()) status = builder->add_u64(19, after.major_fault_count);
  if (status.ok()) status = builder->add_u64(20, before.vmpte_bytes);
  if (status.ok()) status = builder->add_u64(21, after.vmpte_bytes);
  if (status.ok()) {
    status = builder->add_u64(22, before.cgroup_memory_current_bytes);
  }
  if (status.ok()) {
    status = builder->add_u64(23, after.cgroup_memory_current_bytes);
  }
  if (status.ok()) status = builder->add_u64(24, before.cgroup_file_bytes);
  if (status.ok()) status = builder->add_u64(25, after.cgroup_file_bytes);
  if (status.ok()) status = builder->add_u64(26, before.cgroup_anon_bytes);
  if (status.ok()) status = builder->add_u64(27, after.cgroup_anon_bytes);
  if (status.ok()) status = builder->add_u64(28, before.cgroup_kernel_bytes);
  if (status.ok()) status = builder->add_u64(29, after.cgroup_kernel_bytes);
  if (status.ok()) {
    status = builder->add_u32(30, admission.production_eligible() ? 1U : 0U);
  }
  if (status.ok()) {
    status = builder->add_u32(31, admission.dspark_enabled() ? 1U : 0U);
  }
  if (status.ok()) {
    status = builder->add_u32(
        32, static_cast<std::uint32_t>(admission.residency()));
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

}  // namespace

DeepSeekRankArtifactPrefaultReceipt::DeepSeekRankArtifactPrefaultReceipt(
    std::uint64_t engine_epoch, std::uint64_t worker_generation,
    std::uint32_t world_size, std::uint32_t rank,
    std::uint64_t page_bytes, std::uint32_t interval_count,
    std::uint64_t mapped_interval_bytes,
    std::uint64_t selected_page_union_bytes,
    std::uint64_t resident_page_bytes_before,
    std::uint64_t resident_page_bytes_after,
    std::uint64_t touched_page_count,
    DeepSeekRankArtifactPrefaultResourceSnapshot before,
    DeepSeekRankArtifactPrefaultResourceSnapshot after,
    Sha256Digest grant_root, Sha256Digest mapping_owner_root,
    Sha256Digest mapping_plan_root, Sha256Digest layout_root,
    Sha256Digest receipt_root) noexcept
    : engine_epoch_(engine_epoch), worker_generation_(worker_generation),
      world_size_(world_size), rank_(rank), page_bytes_(page_bytes),
      interval_count_(interval_count),
      mapped_interval_bytes_(mapped_interval_bytes),
      selected_page_union_bytes_(selected_page_union_bytes),
      resident_page_bytes_before_(resident_page_bytes_before),
      resident_page_bytes_after_(resident_page_bytes_after),
      touched_page_count_(touched_page_count), before_(before), after_(after),
      grant_root_(grant_root), mapping_owner_root_(mapping_owner_root),
      mapping_plan_root_(mapping_plan_root), layout_root_(layout_root),
      receipt_root_(receipt_root) {}

Result<DeepSeekRankPrefaultedMaterializationInputs>
DeepSeekRankArtifactPrefaultTransaction::Run(
    DeepSeekRankAuthorizedMaterializationInputs inputs,
    DeepSeekRankArtifactPrefaultOperations& operations) {
  if (inputs.mapping_owner == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek artifact prefault mapping owner is absent");
  }
  auto& admission = inputs.admission;
  auto& owner = *inputs.mapping_owner;
  if (admission.engine_epoch() != owner.engine_epoch() ||
      admission.worker_generation() != owner.worker_generation() ||
      admission.world_size() != owner.world_size() ||
      admission.rank() != owner.rank() ||
      admission.mapping_owner_root() != owner.mapping_owner_root() ||
      admission.dspark_enabled() != owner.dspark_enabled()) {
    return Status::FailedPrecondition(
        "DeepSeek artifact prefault authority differs from mapping owner");
  }
  auto layout = compile_deepseek_rank_artifact_prefault_layout(
      owner.mapping_plan(), kPageBytes);
  if (!layout.ok()) return layout.status();
  if (layout->mapped_interval_bytes != owner.mapped_interval_bytes() ||
      layout->interval_count != owner.inventory().mappings_.size()) {
    return Status::FailedPrecondition(
        "DeepSeek artifact prefault inventory differs from its plan");
  }

  auto before = operations.sample_resources();
  if (!before.ok()) return before.status();
  if (before->monotonic_ns == 0 ||
      before->monotonic_ns >= admission.deadline_ns()) {
    return Status::DeadlineExceeded(
        "DeepSeek artifact prefault did not start before its grant deadline");
  }

  std::uint64_t resident_before = 0;
  std::uint64_t resident_after = 0;
  std::uint64_t touched_pages = 0;
  for (std::size_t index = 0;
       index < owner.inventory().mappings_.size(); ++index) {
    const auto& expected = owner.mapping_plan().intervals[index];
    const auto& mapped = owner.inventory().mappings_[index];
    if (mapped.shard_name != expected.shard_name ||
        mapped.range.file_offset() != expected.file_begin ||
        mapped.range.size_bytes() !=
            expected.file_end - expected.file_begin) {
      return Status::FailedPrecondition(
          "DeepSeek artifact prefault mapped range differs from plan");
    }
    auto bytes = mapped.range.slice(0, mapped.range.size_bytes());
    if (!bytes.ok()) return bytes.status();
    auto observation = operations.prefault_range(*bytes, kPageBytes);
    if (!observation.ok()) return observation.status();
    auto expected_range_bytes = checked_align_up_u64(
        mapped.range.size_bytes(), kPageBytes);
    if (!expected_range_bytes.ok()) return expected_range_bytes.status();
    const auto expected_pages = *expected_range_bytes / kPageBytes;
    if (observation->resident_page_bytes_before > *expected_range_bytes ||
        observation->resident_page_bytes_after != *expected_range_bytes ||
        observation->touched_page_count != expected_pages) {
      return Status::FailedPrecondition(
          "DeepSeek artifact prefault range is not fully resident");
    }
    auto next = checked_add_u64(
        resident_before, observation->resident_page_bytes_before);
    if (!next.ok()) return next.status();
    resident_before = *next;
    next = checked_add_u64(
        resident_after, observation->resident_page_bytes_after);
    if (!next.ok()) return next.status();
    resident_after = *next;
    next = checked_add_u64(touched_pages,
                           observation->touched_page_count);
    if (!next.ok()) return next.status();
    touched_pages = *next;
  }
  if (resident_after != layout->selected_page_union_bytes ||
      touched_pages != layout->selected_page_union_bytes / kPageBytes) {
    return Status::FailedPrecondition(
        "DeepSeek artifact prefault page-union ledger differs");
  }

  auto after = operations.sample_resources();
  if (!after.ok()) return after.status();
  if (after->monotonic_ns <= before->monotonic_ns ||
      after->monotonic_ns >= admission.deadline_ns()) {
    return Status::DeadlineExceeded(
        "DeepSeek artifact prefault did not finish before its grant deadline");
  }
  if (after->major_fault_count < before->major_fault_count) {
    return Status::FailedPrecondition(
        "DeepSeek artifact prefault major-fault counter regressed");
  }
  auto root = compile_receipt_root(
      admission, owner, *layout, resident_before, resident_after,
      touched_pages, *before, *after);
  if (!root.ok()) return root.status();
  DeepSeekRankArtifactPrefaultReceipt receipt(
      admission.engine_epoch(), admission.worker_generation(),
      admission.world_size(), admission.rank(), layout->page_bytes,
      layout->interval_count, layout->mapped_interval_bytes,
      layout->selected_page_union_bytes, resident_before, resident_after,
      touched_pages, *before, *after, admission.grant_root(),
      owner.mapping_owner_root(), layout->mapping_plan_root,
      layout->layout_root, *root);
  return DeepSeekRankPrefaultedMaterializationInputs(
      std::move(admission), std::move(inputs.mapping_owner),
      std::move(receipt));
}

Status validate_deepseek_rank_prefaulted_materialization_source(
    const DeepSeekRankPrefaultedMaterializationInputs& inputs) {
  if (inputs.mapping_owner == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek prefaulted materialization owner is absent");
  }
  const auto& admission = inputs.admission;
  const auto& owner = *inputs.mapping_owner;
  const auto& receipt = inputs.prefault_receipt;
  auto layout = compile_deepseek_rank_artifact_prefault_layout(
      owner.mapping_plan(), DeepSeekRankArtifactPrefaultTransaction::kPageBytes);
  if (!layout.ok()) return layout.status();
  if (receipt.engine_epoch() != admission.engine_epoch() ||
      receipt.worker_generation() != admission.worker_generation() ||
      receipt.world_size() != admission.world_size() ||
      receipt.rank() != admission.rank() ||
      receipt.grant_root() != admission.grant_root() ||
      receipt.mapping_owner_root() != owner.mapping_owner_root() ||
      receipt.mapping_plan_root() != layout->mapping_plan_root ||
      receipt.layout_root() != layout->layout_root ||
      receipt.page_bytes() != layout->page_bytes ||
      receipt.interval_count() != layout->interval_count ||
      receipt.mapped_interval_bytes() != layout->mapped_interval_bytes ||
      receipt.selected_page_union_bytes() !=
          layout->selected_page_union_bytes ||
      receipt.resident_page_bytes_before() >
          layout->selected_page_union_bytes ||
      receipt.resident_page_bytes_after() !=
          layout->selected_page_union_bytes ||
      receipt.touched_page_count() !=
          layout->selected_page_union_bytes / layout->page_bytes ||
      receipt.before().monotonic_ns == 0 ||
      receipt.before().monotonic_ns >= receipt.after().monotonic_ns ||
      receipt.after().monotonic_ns >= admission.deadline_ns() ||
      receipt.after().major_fault_count <
          receipt.before().major_fault_count ||
      receipt.receipt_root() == Sha256Digest{}) {
    return Status::FailedPrecondition(
        "DeepSeek prefault receipt differs from materialization source");
  }
  auto expected_root = compile_receipt_root(
      admission, owner, *layout, receipt.resident_page_bytes_before(),
      receipt.resident_page_bytes_after(), receipt.touched_page_count(),
      receipt.before(), receipt.after());
  if (!expected_root.ok()) return expected_root.status();
  if (*expected_root != receipt.receipt_root()) {
    return Status::FailedPrecondition(
        "DeepSeek prefault receipt root differs from its fields");
  }
  return Status::Ok();
}

}  // namespace pih
