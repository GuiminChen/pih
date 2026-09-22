#include "pih/model/sealed_reachable_object_dag.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <queue>
#include <span>
#include <string_view>
#include <unordered_map>

namespace pih {
namespace {

struct DigestHash final {
  std::size_t operator()(const Sha256Digest& value) const noexcept {
    std::size_t result = 0;
    constexpr auto width = std::min(sizeof(result), std::size_t{8});
    for (std::size_t index = 0; index < width; ++index) {
      result = (result << 8) |
               static_cast<std::size_t>(std::to_integer<std::uint8_t>(
                   value.bytes[index]));
    }
    return result;
  }
};

bool nonzero(const Sha256Digest& value) {
  std::byte aggregate{};
  for (const auto byte : value.bytes) aggregate |= byte;
  return aggregate != std::byte{};
}

bool digest_less(const Sha256Digest& left, const Sha256Digest& right) {
  return std::lexicographical_compare(
      left.bytes.begin(), left.bytes.end(), right.bytes.begin(), right.bytes.end());
}

bool valid_schema(std::string_view value) {
  if (value.empty() || value.size() > 128) return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char character) {
    return character >= 0x21 && character <= 0x7e;
  });
}

bool checked_add(std::uint64_t left, std::uint64_t right,
                 std::uint64_t& result) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
  result = left + right;
  return true;
}

bool checked_multiply(std::uint64_t left, std::uint64_t right,
                      std::uint64_t& result) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

void append_u16(std::vector<std::byte>& out, std::uint16_t value) {
  out.push_back(static_cast<std::byte>((value >> 8) & 0xff));
  out.push_back(static_cast<std::byte>(value & 0xff));
}

void append_u32(std::vector<std::byte>& out, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    out.push_back(static_cast<std::byte>((value >> shift) & 0xff));
  }
}

void append_u64(std::vector<std::byte>& out, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    out.push_back(static_cast<std::byte>((value >> shift) & 0xff));
  }
}

class ManifestReader final {
 public:
  explicit ManifestReader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  bool domain(std::string_view value) {
    const auto expected = std::as_bytes(std::span(value.data(), value.size()));
    if (remaining() < expected.size() + 1 ||
        !std::equal(expected.begin(), expected.end(), bytes_.begin() + offset_) ||
        bytes_[offset_ + expected.size()] != std::byte{0}) return false;
    offset_ += expected.size() + 1;
    return true;
  }
  bool u8(std::uint8_t& value) {
    if (remaining() < 1) return false;
    value = std::to_integer<std::uint8_t>(bytes_[offset_++]);
    return true;
  }
  bool u16(std::uint16_t& value) {
    if (remaining() < 2) return false;
    value = (static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes_[offset_])) << 8) |
            std::to_integer<std::uint8_t>(bytes_[offset_ + 1]);
    offset_ += 2;
    return true;
  }
  bool u32(std::uint32_t& value) {
    if (remaining() < 4) return false;
    value = 0;
    for (int index = 0; index < 4; ++index) {
      value = (value << 8) |
              std::to_integer<std::uint8_t>(bytes_[offset_ + index]);
    }
    offset_ += 4;
    return true;
  }
  bool u64(std::uint64_t& value) {
    if (remaining() < 8) return false;
    value = 0;
    for (int index = 0; index < 8; ++index) {
      value = (value << 8) |
              std::to_integer<std::uint8_t>(bytes_[offset_ + index]);
    }
    offset_ += 8;
    return true;
  }
  bool bytes(std::size_t count, std::span<const std::byte>& value) {
    if (remaining() < count) return false;
    value = bytes_.subspan(offset_, count);
    offset_ += count;
    return true;
  }
  [[nodiscard]] bool done() const noexcept { return offset_ == bytes_.size(); }

 private:
  [[nodiscard]] std::size_t remaining() const noexcept {
    return bytes_.size() - offset_;
  }
  std::span<const std::byte> bytes_;
  std::size_t offset_ = 0;
};

Status invalid_manifest() {
  return Status::InvalidArgument("digest DAG manifest encoding is invalid");
}

}  // namespace

SealedReachableObjectDag::SealedReachableObjectDag(
    Sha256Digest snapshot_root, std::uint32_t node_count,
    std::uint32_t edge_count, std::uint32_t longest_path_nodes,
    std::uint64_t fixed_owner_bytes,
    std::vector<Sha256Digest> topological_roots,
    std::vector<std::byte> canonical_manifest_bytes) noexcept
    : snapshot_root_(snapshot_root), node_count_(node_count),
      edge_count_(edge_count), longest_path_nodes_(longest_path_nodes),
      fixed_owner_bytes_(fixed_owner_bytes),
      topological_roots_(std::move(topological_roots)),
      canonical_manifest_bytes_(std::move(canonical_manifest_bytes)) {}

Result<SealedReachableObjectDag> verify_sealed_reachable_object_dag(
    const std::vector<DigestDagNode>& nodes,
    const std::vector<Sha256Digest>& candidate_roots) {
  if (nodes.empty() || nodes.size() > kDigestDagMaximumNodes ||
      candidate_roots.empty() || candidate_roots.size() > nodes.size()) {
    return Status::InvalidArgument("digest DAG node or root count is invalid");
  }

  const auto node_count = static_cast<std::uint32_t>(nodes.size());
  std::unordered_map<Sha256Digest, std::uint32_t, DigestHash> ordinal_by_root;
  ordinal_by_root.reserve(nodes.size());
  std::uint64_t edge_count64 = 0;
  for (std::uint32_t ordinal = 0; ordinal < node_count; ++ordinal) {
    const auto& node = nodes[ordinal];
    if (!valid_schema(node.schema_abi) || !nonzero(node.root) ||
        node.exact_bytes == 0 || node.publication_stage == 0 ||
        node.publication_stage > 64 ||
        node.references.size() > kDigestDagMaximumOutdegree ||
        (node.external_leaf && !node.references.empty())) {
      return Status::InvalidArgument("digest DAG node descriptor is invalid");
    }
    if (!ordinal_by_root.emplace(node.root, ordinal).second) {
      return Status::FailedPrecondition(
          "digest DAG contains an ambiguous or duplicate producer root");
    }
    if (!checked_add(edge_count64, node.references.size(), edge_count64) ||
        edge_count64 > kDigestDagMaximumEdges) {
      return Status::ResourceExhausted("digest DAG edge ceiling exceeded");
    }
  }
  const auto edge_count = static_cast<std::uint32_t>(edge_count64);

  std::uint64_t node_records = 0, edge_records = 0, offsets = 0, work = 0;
  std::uint64_t fixed_owner_bytes = 0;
  if (!checked_multiply(node_count, 128, node_records) ||
      !checked_multiply(edge_count64, 8, edge_records) ||
      !checked_multiply(static_cast<std::uint64_t>(node_count) + 1, 8,
                        offsets) ||
      !checked_multiply(node_count, 64, work) ||
      !checked_add(node_records, edge_records, fixed_owner_bytes) ||
      !checked_add(fixed_owner_bytes, offsets, fixed_owner_bytes) ||
      !checked_add(fixed_owner_bytes, work, fixed_owner_bytes) ||
      fixed_owner_bytes > kDigestDagArenaBytes) {
    return Status::ResourceExhausted("digest DAG fixed owner arena exceeded");
  }

  std::vector<std::vector<std::uint32_t>> adjacency(node_count);
  std::vector<std::vector<std::uint32_t>> reverse(node_count);
  std::vector<std::uint32_t> indegree(node_count, 0);
  for (std::uint32_t consumer = 0; consumer < node_count; ++consumer) {
    auto& outgoing = adjacency[consumer];
    outgoing.reserve(nodes[consumer].references.size());
    for (const auto& reference : nodes[consumer].references) {
      if (!nonzero(reference.producer_root)) {
        return Status::FailedPrecondition(
            "digest DAG contains a zero or future producer root");
      }
      const auto found = ordinal_by_root.find(reference.producer_root);
      if (found == ordinal_by_root.end()) {
        return Status::FailedPrecondition("digest DAG producer is missing");
      }
      const auto producer = found->second;
      if (consumer == producer) {
        return Status::FailedPrecondition("digest DAG contains a self edge");
      }
      const auto consumer_stage = nodes[consumer].publication_stage;
      const auto producer_stage = nodes[producer].publication_stage;
      if (producer_stage > consumer_stage ||
          (producer_stage == consumer_stage &&
           !reference.same_stage_allowed)) {
        return Status::FailedPrecondition(
            "digest DAG edge violates publication stage policy");
      }
      outgoing.push_back(producer);
      reverse[producer].push_back(consumer);
      ++indegree[producer];
    }
    std::sort(outgoing.begin(), outgoing.end(), [&](auto left, auto right) {
      return digest_less(nodes[left].root, nodes[right].root);
    });
    if (std::adjacent_find(outgoing.begin(), outgoing.end()) != outgoing.end()) {
      return Status::FailedPrecondition(
          "digest DAG consumer repeats a producer reference");
    }
  }

  std::vector<std::uint8_t> color(node_count, 0);
  std::vector<std::uint32_t> finish;
  finish.reserve(node_count);
  struct Frame { std::uint32_t node; std::uint32_t next; };
  std::vector<Frame> stack;
  for (std::uint32_t start = 0; start < node_count; ++start) {
    if (color[start] != 0) continue;
    color[start] = 1;
    stack.push_back({start, 0});
    while (!stack.empty()) {
      auto& frame = stack.back();
      if (frame.next < adjacency[frame.node].size()) {
        const auto next = adjacency[frame.node][frame.next++];
        if (color[next] == 0) {
          color[next] = 1;
          stack.push_back({next, 0});
        }
      } else {
        color[frame.node] = 2;
        finish.push_back(frame.node);
        stack.pop_back();
      }
    }
  }
  std::fill(color.begin(), color.end(), 0);
  std::vector<std::uint32_t> component;
  for (auto iterator = finish.rbegin(); iterator != finish.rend(); ++iterator) {
    if (color[*iterator] != 0) continue;
    std::uint32_t component_size = 0;
    component.push_back(*iterator);
    color[*iterator] = 1;
    while (!component.empty()) {
      const auto current = component.back();
      component.pop_back();
      ++component_size;
      for (const auto next : reverse[current]) {
        if (color[next] == 0) {
          color[next] = 1;
          component.push_back(next);
        }
      }
    }
    if (component_size > 1) {
      return Status::FailedPrecondition(
          "digest DAG strongly connected component contains a cycle");
    }
  }

  auto root_less = [&](std::uint32_t left, std::uint32_t right) {
    return digest_less(nodes[right].root, nodes[left].root);
  };
  std::priority_queue<std::uint32_t, std::vector<std::uint32_t>,
                      decltype(root_less)> ready(root_less);
  for (std::uint32_t ordinal = 0; ordinal < node_count; ++ordinal) {
    if (indegree[ordinal] == 0) ready.push(ordinal);
  }
  std::vector<std::uint32_t> topological;
  topological.reserve(node_count);
  while (!ready.empty()) {
    const auto consumer = ready.top();
    ready.pop();
    topological.push_back(consumer);
    for (const auto producer : adjacency[consumer]) {
      if (--indegree[producer] == 0) ready.push(producer);
    }
  }
  if (topological.size() != node_count) {
    return Status::FailedPrecondition("digest DAG topological sort failed");
  }

  std::vector<std::uint32_t> depth(node_count, 0);
  for (const auto& root : candidate_roots) {
    if (!nonzero(root)) {
      return Status::FailedPrecondition("digest DAG candidate root is zero");
    }
    const auto found = ordinal_by_root.find(root);
    if (found == ordinal_by_root.end()) {
      return Status::FailedPrecondition("digest DAG candidate root is missing");
    }
    depth[found->second] = 1;
  }
  std::uint32_t longest_path = 0;
  for (const auto consumer : topological) {
    if (depth[consumer] == 0) continue;
    longest_path = std::max(longest_path, depth[consumer]);
    for (const auto producer : adjacency[consumer]) {
      const auto next_depth = depth[consumer] + 1;
      if (next_depth > kDigestDagMaximumPathNodes) {
        return Status::ResourceExhausted("digest DAG path depth exceeded");
      }
      depth[producer] = std::max(depth[producer], next_depth);
    }
  }
  if (std::any_of(depth.begin(), depth.end(), [](auto value) { return value == 0; })) {
    return Status::FailedPrecondition(
        "digest DAG snapshot contains an unreachable object");
  }

  constexpr std::string_view abi = "sealed_reachable_object_dag_v2";
  std::vector<std::byte> canonical;
  canonical.reserve(abi.size() + nodes.size() * 192 + edge_count64 * 32);
  const auto abi_bytes = std::as_bytes(std::span(abi.data(), abi.size()));
  canonical.insert(canonical.end(), abi_bytes.begin(), abi_bytes.end());
  canonical.push_back(std::byte{0});
  append_u32(canonical, node_count);
  append_u32(canonical, edge_count);
  append_u32(canonical, longest_path);
  auto ordered_candidates = candidate_roots;
  std::sort(ordered_candidates.begin(), ordered_candidates.end(), digest_less);
  if (std::adjacent_find(ordered_candidates.begin(), ordered_candidates.end()) !=
      ordered_candidates.end()) {
    return Status::InvalidArgument("digest DAG candidate root is duplicated");
  }
  append_u32(canonical,
             static_cast<std::uint32_t>(ordered_candidates.size()));
  for (const auto& root : ordered_candidates) {
    canonical.insert(canonical.end(), root.bytes.begin(), root.bytes.end());
  }
  std::vector<Sha256Digest> ordered_roots;
  ordered_roots.reserve(node_count);
  for (const auto ordinal : topological) {
    const auto& node = nodes[ordinal];
    ordered_roots.push_back(node.root);
    append_u16(canonical, static_cast<std::uint16_t>(node.schema_abi.size()));
    const auto schema_bytes = std::as_bytes(
        std::span(node.schema_abi.data(), node.schema_abi.size()));
    canonical.insert(canonical.end(), schema_bytes.begin(), schema_bytes.end());
    canonical.insert(canonical.end(), node.root.bytes.begin(), node.root.bytes.end());
    append_u64(canonical, node.exact_bytes);
    append_u16(canonical, node.publication_stage);
    canonical.push_back(node.external_leaf ? std::byte{1} : std::byte{0});
    std::vector<const DigestDagReference*> ordered_references;
    ordered_references.reserve(node.references.size());
    for (const auto& reference : node.references) {
      ordered_references.push_back(&reference);
    }
    std::sort(ordered_references.begin(), ordered_references.end(),
              [](const auto* left, const auto* right) {
                return digest_less(left->producer_root, right->producer_root);
              });
    append_u32(canonical,
               static_cast<std::uint32_t>(ordered_references.size()));
    for (const auto* reference : ordered_references) {
      canonical.insert(canonical.end(), reference->producer_root.bytes.begin(),
                       reference->producer_root.bytes.end());
      canonical.push_back(reference->same_stage_allowed ? std::byte{1}
                                                        : std::byte{0});
    }
  }
  auto snapshot_root = sha256(canonical);
  if (!snapshot_root.ok()) return snapshot_root.status();
  return SealedReachableObjectDag(
      *snapshot_root, node_count, edge_count, longest_path, fixed_owner_bytes,
      std::move(ordered_roots), std::move(canonical));
}

Result<SealedReachableObjectDag> parse_sealed_reachable_object_dag(
    std::span<const std::byte> manifest_bytes) {
  if (manifest_bytes.empty() ||
      manifest_bytes.size() > kDigestDagMaximumManifestBytes) {
    return invalid_manifest();
  }
  ManifestReader reader(manifest_bytes);
  std::uint32_t node_count = 0, edge_count = 0, longest_path = 0;
  std::uint32_t candidate_count = 0;
  if (!reader.domain("sealed_reachable_object_dag_v2") ||
      !reader.u32(node_count) || !reader.u32(edge_count) ||
      !reader.u32(longest_path) || !reader.u32(candidate_count) ||
      node_count == 0 || node_count > kDigestDagMaximumNodes ||
      edge_count > kDigestDagMaximumEdges || candidate_count == 0 ||
      candidate_count > node_count || longest_path == 0 ||
      longest_path > kDigestDagMaximumPathNodes) {
    return invalid_manifest();
  }
  std::vector<Sha256Digest> candidates(candidate_count);
  for (auto& candidate : candidates) {
    std::span<const std::byte> encoded;
    if (!reader.bytes(candidate.bytes.size(), encoded)) return invalid_manifest();
    std::copy(encoded.begin(), encoded.end(), candidate.bytes.begin());
  }
  std::vector<DigestDagNode> nodes;
  nodes.reserve(node_count);
  std::uint64_t observed_edges = 0;
  for (std::uint32_t index = 0; index < node_count; ++index) {
    std::uint16_t schema_bytes = 0, stage = 0;
    std::uint8_t external_leaf = 0;
    std::uint32_t reference_count = 0;
    std::span<const std::byte> schema, root;
    std::uint64_t exact_bytes = 0;
    if (!reader.u16(schema_bytes) || schema_bytes == 0 || schema_bytes > 128 ||
        !reader.bytes(schema_bytes, schema) || !reader.bytes(32, root) ||
        !reader.u64(exact_bytes) || !reader.u16(stage) ||
        !reader.u8(external_leaf) || external_leaf > 1 ||
        !reader.u32(reference_count) ||
        reference_count > kDigestDagMaximumOutdegree ||
        observed_edges + reference_count > edge_count) {
      return invalid_manifest();
    }
    DigestDagNode node;
    node.schema_abi.assign(reinterpret_cast<const char*>(schema.data()),
                           schema.size());
    std::copy(root.begin(), root.end(), node.root.bytes.begin());
    node.exact_bytes = exact_bytes;
    node.publication_stage = stage;
    node.external_leaf = external_leaf != 0;
    node.references.reserve(reference_count);
    for (std::uint32_t reference_index = 0;
         reference_index < reference_count; ++reference_index) {
      std::span<const std::byte> producer;
      std::uint8_t same_stage_allowed = 0;
      if (!reader.bytes(32, producer) || !reader.u8(same_stage_allowed) ||
          same_stage_allowed > 1) return invalid_manifest();
      DigestDagReference reference;
      std::copy(producer.begin(), producer.end(),
                reference.producer_root.bytes.begin());
      reference.same_stage_allowed = same_stage_allowed != 0;
      node.references.push_back(reference);
    }
    observed_edges += reference_count;
    nodes.push_back(std::move(node));
  }
  if (!reader.done() || observed_edges != edge_count) return invalid_manifest();
  auto verified = verify_sealed_reachable_object_dag(nodes, candidates);
  if (!verified.ok()) return verified.status();
  const auto canonical = verified->canonical_manifest_bytes();
  if (canonical.size() != manifest_bytes.size() ||
      !std::equal(canonical.begin(), canonical.end(), manifest_bytes.begin())) {
    return invalid_manifest();
  }
  return verified;
}

}  // namespace pih
