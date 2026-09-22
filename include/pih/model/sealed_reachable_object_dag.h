#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

inline constexpr std::uint32_t kDigestDagMaximumNodes = 65'536;
inline constexpr std::uint32_t kDigestDagMaximumEdges = 1'048'576;
inline constexpr std::uint32_t kDigestDagMaximumOutdegree = 16'384;
inline constexpr std::uint32_t kDigestDagMaximumPathNodes = 32;
inline constexpr std::uint64_t kDigestDagArenaBytes = 134'217'728;
inline constexpr std::uint64_t kDigestDagMaximumManifestBytes = 67'108'864;

struct DigestDagReference final {
  Sha256Digest producer_root{};
  bool same_stage_allowed = false;
};

struct DigestDagNode final {
  std::string schema_abi;
  Sha256Digest root{};
  std::uint64_t exact_bytes = 0;
  std::uint16_t publication_stage = 0;
  bool external_leaf = false;
  std::vector<DigestDagReference> references;
};

class SealedReachableObjectDag final {
 public:
  [[nodiscard]] const Sha256Digest& snapshot_root() const noexcept {
    return snapshot_root_;
  }
  [[nodiscard]] std::uint32_t node_count() const noexcept { return node_count_; }
  [[nodiscard]] std::uint32_t edge_count() const noexcept { return edge_count_; }
  [[nodiscard]] std::uint32_t longest_path_nodes() const noexcept {
    return longest_path_nodes_;
  }
  [[nodiscard]] std::uint64_t fixed_owner_bytes() const noexcept {
    return fixed_owner_bytes_;
  }
  [[nodiscard]] const std::vector<Sha256Digest>& topological_roots()
      const noexcept { return topological_roots_; }
  [[nodiscard]] std::span<const std::byte> canonical_manifest_bytes()
      const noexcept { return canonical_manifest_bytes_; }

 private:
  friend Result<SealedReachableObjectDag> verify_sealed_reachable_object_dag(
      const std::vector<DigestDagNode>&,
      const std::vector<Sha256Digest>&);
  friend Result<SealedReachableObjectDag> parse_sealed_reachable_object_dag(
      std::span<const std::byte>);
  SealedReachableObjectDag(
      Sha256Digest snapshot_root, std::uint32_t node_count,
      std::uint32_t edge_count, std::uint32_t longest_path_nodes,
      std::uint64_t fixed_owner_bytes,
      std::vector<Sha256Digest> topological_roots,
      std::vector<std::byte> canonical_manifest_bytes) noexcept;

  Sha256Digest snapshot_root_{};
  std::uint32_t node_count_ = 0;
  std::uint32_t edge_count_ = 0;
  std::uint32_t longest_path_nodes_ = 0;
  std::uint64_t fixed_owner_bytes_ = 0;
  std::vector<Sha256Digest> topological_roots_;
  std::vector<std::byte> canonical_manifest_bytes_;
};

Result<SealedReachableObjectDag> verify_sealed_reachable_object_dag(
    const std::vector<DigestDagNode>& nodes,
    const std::vector<Sha256Digest>& candidate_roots);

Result<SealedReachableObjectDag> parse_sealed_reachable_object_dag(
    std::span<const std::byte> manifest_bytes);

}  // namespace pih
