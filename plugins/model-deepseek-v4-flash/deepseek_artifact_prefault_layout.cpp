#include "pih/model/deepseek_artifact_prefault_layout.h"

#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "pih/core/canonical_hash.h"
#include "pih/core/checked_math.h"

namespace pih {
namespace {

struct PageRange final {
  std::string shard_name;
  std::uint64_t shard_bytes = 0;
  std::uint64_t page_begin = 0;
  std::uint64_t page_end = 0;
};

Result<Sha256Digest> compile_root_set(
    std::string_view domain, std::span<const Sha256Digest> roots) {
  if (roots.size() > 65'436U) {
    return Status::InvalidArgument(
        "DeepSeek prefault root-set geometry is invalid");
  }
  auto builder = CanonicalHashBuilder::Create(
      domain, static_cast<std::uint32_t>(roots.size() + 1U));
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(
      1, static_cast<std::uint32_t>(roots.size()));
  for (std::size_t index = 0; status.ok() && index < roots.size(); ++index) {
    status = builder->add_hash(
        static_cast<std::uint16_t>(index + 100U), roots[index]);
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<Sha256Digest> compile_chunked_root_set(
    std::string_view chunk_domain, std::string_view set_domain,
    std::span<const Sha256Digest> roots) {
  constexpr std::size_t kRecordsPerChunk = 4096;
  std::vector<Sha256Digest> chunks;
  chunks.reserve((roots.size() + kRecordsPerChunk - 1U) /
                 kRecordsPerChunk);
  for (std::size_t begin = 0; begin < roots.size();
       begin += kRecordsPerChunk) {
    const auto count = std::min(kRecordsPerChunk, roots.size() - begin);
    auto chunk = compile_root_set(chunk_domain, roots.subspan(begin, count));
    if (!chunk.ok()) return chunk.status();
    chunks.push_back(*chunk);
  }
  return compile_root_set(set_domain, chunks);
}

bool range_less(const PageRange& lhs, const PageRange& rhs) noexcept {
  if (lhs.shard_name != rhs.shard_name) {
    return lhs.shard_name < rhs.shard_name;
  }
  if (lhs.page_begin != rhs.page_begin) {
    return lhs.page_begin < rhs.page_begin;
  }
  return lhs.page_end < rhs.page_end;
}

}  // namespace

Result<DeepSeekRankArtifactPrefaultLayout>
compile_deepseek_rank_artifact_prefault_layout(
    const DeepSeekRankMappingPlan& mapping_plan,
    std::uint64_t page_bytes) {
  if (page_bytes != kDeepSeekArtifactPrefaultPageBytes ||
      mapping_plan.intervals.empty() ||
      mapping_plan.intervals.size() >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return Status::InvalidArgument(
        "DeepSeek artifact prefault layout shape is invalid");
  }
  auto mapping_root = compile_deepseek_rank_mapping_plan_root(mapping_plan);
  if (!mapping_root.ok()) return mapping_root.status();
  std::uint64_t selected_page_union_bytes = 0;
  for (const auto& interval : mapping_plan.intervals) {
    if ((interval.file_begin % page_bytes) != 0 ||
        interval.file_begin >= interval.file_end) {
      return Status::InvalidArgument(
          "DeepSeek artifact prefault interval is not page canonical");
    }
    const auto interval_bytes = interval.file_end - interval.file_begin;
    auto rounded = checked_align_up_u64(interval_bytes, page_bytes);
    if (!rounded.ok()) return rounded.status();
    auto total = checked_add_u64(selected_page_union_bytes, *rounded);
    if (!total.ok()) return total.status();
    selected_page_union_bytes = *total;
  }
  if (selected_page_union_bytes < mapping_plan.mapped_interval_bytes) {
    return Status::Internal(
        "DeepSeek artifact prefault page union undercounts mappings");
  }
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-artifact-prefault-layout:v1", 6);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(1, mapping_plan.rank);
  if (status.ok()) status = builder->add_hash(2, *mapping_root);
  if (status.ok()) status = builder->add_u64(3, page_bytes);
  if (status.ok()) {
    status = builder->add_u32(
        4, static_cast<std::uint32_t>(mapping_plan.intervals.size()));
  }
  if (status.ok()) {
    status = builder->add_u64(5, mapping_plan.mapped_interval_bytes);
  }
  if (status.ok()) status = builder->add_u64(6, selected_page_union_bytes);
  if (!status.ok()) return status;
  auto layout_root = builder->finalize();
  if (!layout_root.ok()) return layout_root.status();
  return DeepSeekRankArtifactPrefaultLayout{
      mapping_plan.rank,
      static_cast<std::uint32_t>(mapping_plan.intervals.size()),
      page_bytes,
      mapping_plan.mapped_interval_bytes,
      selected_page_union_bytes,
      *mapping_root,
      *layout_root};
}

Result<DeepSeekNodeArtifactPrefaultLayout>
compile_deepseek_node_artifact_prefault_layout(
    std::span<const DeepSeekRankMappingPlan> rank_mapping_plans,
    std::uint64_t page_bytes) {
  if (rank_mapping_plans.empty() || rank_mapping_plans.size() > 4U ||
      page_bytes != kDeepSeekArtifactPrefaultPageBytes) {
    return Status::InvalidArgument(
        "DeepSeek node prefault layout geometry is invalid");
  }

  std::vector<Sha256Digest> rank_layout_roots;
  std::vector<PageRange> ranges;
  std::map<std::string, std::uint64_t, std::less<>> shard_sizes;
  std::uint64_t summed_rank_bytes = 0;
  rank_layout_roots.reserve(rank_mapping_plans.size());

  for (std::size_t rank = 0; rank < rank_mapping_plans.size(); ++rank) {
    const auto& plan = rank_mapping_plans[rank];
    if (plan.rank != rank) {
      return Status::InvalidArgument(
          "DeepSeek node prefault ranks are not canonical");
    }
    auto layout = compile_deepseek_rank_artifact_prefault_layout(
        plan, page_bytes);
    if (!layout.ok()) return layout.status();
    auto summed = checked_add_u64(
        summed_rank_bytes, layout->selected_page_union_bytes);
    if (!summed.ok()) return summed.status();
    summed_rank_bytes = *summed;
    rank_layout_roots.push_back(layout->layout_root);

    for (const auto& shard : plan.shards) {
      const auto [position, inserted] =
          shard_sizes.emplace(shard.shard_name, shard.file_bytes);
      if (!inserted && position->second != shard.file_bytes) {
        return Status::FailedPrecondition(
            "DeepSeek node prefault shard size differs across ranks");
      }
    }
    for (const auto& interval : plan.intervals) {
      const auto shard = shard_sizes.find(interval.shard_name);
      if (shard == shard_sizes.end() || interval.file_end > shard->second ||
          (interval.file_begin % page_bytes) != 0 ||
          interval.file_begin >= interval.file_end) {
        return Status::InvalidArgument(
            "DeepSeek node prefault interval identity is invalid");
      }
      const auto page_end =
          1U + ((interval.file_end - 1U) / page_bytes);
      ranges.push_back(PageRange{interval.shard_name, shard->second,
                                 interval.file_begin / page_bytes,
                                 page_end});
    }
  }

  auto rank_layout_set_root = compile_root_set(
      "pih:deepseek-node-prefault-rank-layout-set:v1",
      rank_layout_roots);
  if (!rank_layout_set_root.ok()) return rank_layout_set_root.status();

  std::sort(ranges.begin(), ranges.end(), range_less);
  std::vector<PageRange> merged;
  merged.reserve(ranges.size());
  for (const auto& range : ranges) {
    if (!merged.empty() &&
        merged.back().shard_name == range.shard_name &&
        range.page_begin <= merged.back().page_end) {
      merged.back().page_end =
          std::max(merged.back().page_end, range.page_end);
      continue;
    }
    merged.push_back(range);
  }

  std::uint64_t node_union_bytes = 0;
  std::vector<Sha256Digest> range_roots;
  range_roots.reserve(merged.size());
  for (std::size_t index = 0; index < merged.size(); ++index) {
    const auto& range = merged[index];
    auto range_bytes = checked_mul_u64(
        range.page_end - range.page_begin, page_bytes);
    if (!range_bytes.ok()) return range_bytes.status();
    auto total = checked_add_u64(node_union_bytes, *range_bytes);
    if (!total.ok()) return total.status();
    node_union_bytes = *total;
    auto record = CanonicalHashBuilder::Create(
        "pih:deepseek-node-prefault-page-range:v1", 6);
    if (!record.ok()) return record.status();
    auto status = record->add_u64(1, index);
    if (status.ok()) status = record->add_ascii_utf8(2, range.shard_name);
    if (status.ok()) status = record->add_u64(3, range.shard_bytes);
    if (status.ok()) status = record->add_u64(4, range.page_begin);
    if (status.ok()) status = record->add_u64(5, range.page_end);
    if (status.ok()) status = record->add_u64(6, *range_bytes);
    if (!status.ok()) return status;
    auto root = record->finalize();
    if (!root.ok()) return root.status();
    range_roots.push_back(*root);
  }
  if (node_union_bytes > summed_rank_bytes) {
    return Status::Internal(
        "DeepSeek node prefault union exceeds rank sum");
  }
  auto node_page_union_root = compile_chunked_root_set(
      "pih:deepseek-node-prefault-page-range-chunk:v1",
      "pih:deepseek-node-prefault-page-range-set:v1", range_roots);
  if (!node_page_union_root.ok()) return node_page_union_root.status();
  const auto duplicate_bytes = summed_rank_bytes - node_union_bytes;

  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-node-artifact-prefault-layout:v1", 7);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(
      1, static_cast<std::uint32_t>(rank_mapping_plans.size()));
  if (status.ok()) status = builder->add_u64(2, page_bytes);
  if (status.ok()) status = builder->add_u64(3, summed_rank_bytes);
  if (status.ok()) status = builder->add_u64(4, node_union_bytes);
  if (status.ok()) status = builder->add_u64(5, duplicate_bytes);
  if (status.ok()) status = builder->add_hash(6, *rank_layout_set_root);
  if (status.ok()) status = builder->add_hash(7, *node_page_union_root);
  if (!status.ok()) return status;
  auto layout_root = builder->finalize();
  if (!layout_root.ok()) return layout_root.status();
  return DeepSeekNodeArtifactPrefaultLayout{
      static_cast<std::uint32_t>(rank_mapping_plans.size()),
      page_bytes,
      summed_rank_bytes,
      node_union_bytes,
      duplicate_bytes,
      *rank_layout_set_root,
      *node_page_union_root,
      *layout_root};
}

}  // namespace pih
