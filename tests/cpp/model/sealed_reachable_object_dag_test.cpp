#include "pih/model/sealed_reachable_object_dag.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace pih {
namespace {

Sha256Digest dag_root(std::uint32_t value) {
  Sha256Digest root{};
  root.bytes[0] = static_cast<std::byte>((value >> 24) & 0xff);
  root.bytes[1] = static_cast<std::byte>((value >> 16) & 0xff);
  root.bytes[2] = static_cast<std::byte>((value >> 8) & 0xff);
  root.bytes[3] = static_cast<std::byte>(value & 0xff);
  root.bytes[31] = std::byte{1};
  return root;
}

DigestDagNode dag_node(std::uint32_t id, std::uint16_t stage,
                       std::vector<DigestDagReference> references = {},
                       bool leaf = false) {
  return {"schema_v1", dag_root(id), id + 1, stage, leaf,
          std::move(references)};
}

TEST(SealedReachableObjectDagTest, SealsDeterministicBranchedTopology) {
  std::vector<DigestDagNode> nodes{
      dag_node(1, 3, {{dag_root(2), false}, {dag_root(3), false}}),
      dag_node(2, 2, {{dag_root(3), false}}),
      dag_node(3, 1, {}, true),
  };
  auto first = verify_sealed_reachable_object_dag(nodes, {dag_root(1)});
  ASSERT_TRUE(first.ok()) << first.status().message();
  EXPECT_EQ(first->node_count(), 3U);
  EXPECT_EQ(first->edge_count(), 3U);
  EXPECT_EQ(first->longest_path_nodes(), 3U);
  EXPECT_EQ(first->fixed_owner_bytes(), 632U);

  std::reverse(nodes.begin(), nodes.end());
  auto reordered = verify_sealed_reachable_object_dag(nodes, {dag_root(1)});
  ASSERT_TRUE(reordered.ok());
  EXPECT_EQ(first->snapshot_root(), reordered->snapshot_root());
  EXPECT_EQ(first->topological_roots(), reordered->topological_roots());
}

TEST(SealedReachableObjectDagTest, SnapshotCommitsToRootsAndEdgePolicy) {
  const std::vector<DigestDagNode> permissive{
      dag_node(1, 2, {{dag_root(2), true}}),
      dag_node(2, 2, {}, true),
  };
  auto one_root = verify_sealed_reachable_object_dag(permissive, {dag_root(1)});
  auto redundant_root =
      verify_sealed_reachable_object_dag(permissive, {dag_root(1), dag_root(2)});
  ASSERT_TRUE(one_root.ok());
  ASSERT_TRUE(redundant_root.ok());
  EXPECT_NE(one_root->snapshot_root(), redundant_root->snapshot_root());

  auto strict = permissive;
  strict[0].publication_stage = 3;
  strict[0].references[0].same_stage_allowed = false;
  auto strict_graph =
      verify_sealed_reachable_object_dag(strict, {dag_root(1)});
  ASSERT_TRUE(strict_graph.ok());

  auto permissive_different_stage = strict;
  permissive_different_stage[0].references[0].same_stage_allowed = true;
  auto permissive_graph = verify_sealed_reachable_object_dag(
      permissive_different_stage, {dag_root(1)});
  ASSERT_TRUE(permissive_graph.ok());
  EXPECT_NE(permissive_graph->snapshot_root(), strict_graph->snapshot_root());
}

TEST(SealedReachableObjectDagTest, ParsesOnlyExactCanonicalManifest) {
  const std::vector<DigestDagNode> nodes{
      dag_node(1, 2, {{dag_root(2), true}}),
      dag_node(2, 2, {}, true),
  };
  auto sealed = verify_sealed_reachable_object_dag(nodes, {dag_root(1)});
  ASSERT_TRUE(sealed.ok());

  auto parsed =
      parse_sealed_reachable_object_dag(sealed->canonical_manifest_bytes());
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  EXPECT_EQ(parsed->snapshot_root(), sealed->snapshot_root());
  EXPECT_EQ(parsed->node_count(), 2U);
  EXPECT_EQ(parsed->edge_count(), 1U);

  auto truncated = std::vector<std::byte>(
      sealed->canonical_manifest_bytes().begin(),
      sealed->canonical_manifest_bytes().end() - 1);
  EXPECT_FALSE(parse_sealed_reachable_object_dag(truncated).ok());

  auto trailing = std::vector<std::byte>(
      sealed->canonical_manifest_bytes().begin(),
      sealed->canonical_manifest_bytes().end());
  trailing.push_back(std::byte{0});
  EXPECT_FALSE(parse_sealed_reachable_object_dag(trailing).ok());

  auto policy_downgrade = std::vector<std::byte>(
      sealed->canonical_manifest_bytes().begin(),
      sealed->canonical_manifest_bytes().end());
  constexpr std::size_t policy_offset =
      std::string_view("sealed_reachable_object_dag_v2").size() + 1 +
      4 * 4 + 32 + 2 + 9 + 32 + 8 + 2 + 1 + 4 + 32;
  ASSERT_LT(policy_offset, policy_downgrade.size());
  ASSERT_EQ(policy_downgrade[policy_offset], std::byte{1});
  policy_downgrade[policy_offset] = std::byte{0};
  EXPECT_FALSE(parse_sealed_reachable_object_dag(policy_downgrade).ok());
}

TEST(SealedReachableObjectDagTest, RejectsMissingAliasBackEdgeAndUnreachable) {
  auto missing = std::vector<DigestDagNode>{
      dag_node(1, 2, {{dag_root(9), false}})};
  EXPECT_FALSE(verify_sealed_reachable_object_dag(missing, {dag_root(1)}).ok());

  auto alias = std::vector<DigestDagNode>{dag_node(1, 1), dag_node(1, 1)};
  EXPECT_FALSE(verify_sealed_reachable_object_dag(alias, {dag_root(1)}).ok());

  auto back_edge = std::vector<DigestDagNode>{
      dag_node(1, 1, {{dag_root(2), false}}), dag_node(2, 2)};
  EXPECT_FALSE(
      verify_sealed_reachable_object_dag(back_edge, {dag_root(1)}).ok());

  auto unreachable = std::vector<DigestDagNode>{dag_node(1, 1), dag_node(2, 1)};
  EXPECT_FALSE(
      verify_sealed_reachable_object_dag(unreachable, {dag_root(1)}).ok());
}

class CycleLengths : public ::testing::TestWithParam<std::uint32_t> {};

TEST_P(CycleLengths, RejectsArbitraryIterativeStronglyConnectedComponents) {
  const auto size = GetParam();
  std::vector<DigestDagNode> nodes;
  nodes.reserve(size);
  for (std::uint32_t index = 0; index < size; ++index) {
    nodes.push_back(dag_node(
        index + 1, 1,
        {{dag_root(((index + 1) % size) + 1), true}}));
  }
  EXPECT_FALSE(
      verify_sealed_reachable_object_dag(nodes, {dag_root(1)}).ok());
}

INSTANTIATE_TEST_SUITE_P(RequiredCycleLengths, CycleLengths,
                         ::testing::Values(1U, 2U, 3U, 5U, 8U, 31U));

TEST(SealedReachableObjectDagTest, RejectsDepthAndResourceLimitPlusOne) {
  std::vector<DigestDagNode> deep;
  for (std::uint32_t index = 0; index < 33; ++index) {
    std::vector<DigestDagReference> references;
    if (index + 1 < 33) references.push_back({dag_root(index + 2), false});
    deep.push_back(dag_node(index + 1, static_cast<std::uint16_t>(33 - index),
                            std::move(references), index == 32));
  }
  EXPECT_FALSE(verify_sealed_reachable_object_dag(deep, {dag_root(1)}).ok());

  auto excessive_fanout = dag_node(1, 2);
  excessive_fanout.references.resize(kDigestDagMaximumOutdegree + 1,
                                     {dag_root(2), false});
  EXPECT_FALSE(verify_sealed_reachable_object_dag(
      {std::move(excessive_fanout), dag_node(2, 1)}, {dag_root(1)}).ok());

  std::vector<DigestDagNode> excessive_nodes;
  excessive_nodes.reserve(kDigestDagMaximumNodes + 1);
  for (std::uint32_t index = 0; index <= kDigestDagMaximumNodes; ++index) {
    excessive_nodes.push_back(dag_node(index + 1, 1));
  }
  EXPECT_FALSE(verify_sealed_reachable_object_dag(
      excessive_nodes, {dag_root(1)}).ok());
}

}  // namespace
}  // namespace pih
