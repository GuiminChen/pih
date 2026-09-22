#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "pih/model/deepseek_rank_artifact_handoff_plan.h"

namespace pih {

inline constexpr std::string_view kDeepSeekArtifactPrefaultLayoutAbi =
    "pih_deepseek_artifact_prefault_layout_v1";
inline constexpr std::uint64_t kDeepSeekArtifactPrefaultPageBytes = 4096;

// Controller- and worker-recomputable projection of the exact pages selected
// by one rank's canonical mapping plan. mapped_interval_bytes is logical file
// payload; selected_page_union_bytes includes each selected tail page in full.
struct DeepSeekRankArtifactPrefaultLayout final {
  std::uint32_t rank = 0;
  std::uint32_t interval_count = 0;
  std::uint64_t page_bytes = 0;
  std::uint64_t mapped_interval_bytes = 0;
  std::uint64_t selected_page_union_bytes = 0;
  Sha256Digest mapping_plan_root{};
  Sha256Digest layout_root{};
};

// Node-wide physical-file projection. Rank pages are summed for process-local
// residency checks, while node_selected_page_union_bytes de-duplicates equal
// (shard, page) identities across ranks for host-memory accounting.
struct DeepSeekNodeArtifactPrefaultLayout final {
  std::uint32_t world_size = 0;
  std::uint64_t page_bytes = 0;
  std::uint64_t summed_rank_selected_page_bytes = 0;
  std::uint64_t node_selected_page_union_bytes = 0;
  std::uint64_t node_duplicate_selected_page_bytes = 0;
  Sha256Digest rank_layout_set_root{};
  Sha256Digest node_page_union_root{};
  Sha256Digest layout_root{};
};

Result<DeepSeekRankArtifactPrefaultLayout>
compile_deepseek_rank_artifact_prefault_layout(
    const DeepSeekRankMappingPlan& mapping_plan,
    std::uint64_t page_bytes = kDeepSeekArtifactPrefaultPageBytes);

Result<DeepSeekNodeArtifactPrefaultLayout>
compile_deepseek_node_artifact_prefault_layout(
    std::span<const DeepSeekRankMappingPlan> rank_mapping_plans,
    std::uint64_t page_bytes = kDeepSeekArtifactPrefaultPageBytes);

}  // namespace pih
