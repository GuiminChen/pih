#include "pih/model/deepseek_rank_boundary_endpoints.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

DeepSeekNcclCommunicatorManifest topology_manifest(
    std::uint32_t local, std::uint32_t peer, std::uint32_t edge,
    std::uint32_t communicator_rank) {
  return {.engine_epoch = 3,
          .communicator_generation = 4,
          .bootstrap_lease_id = 5 + edge,
          .bootstrap_commitment_id = 9 + edge,
          .edge_id = edge,
          .local_global_rank = local,
          .peer_global_rank = peer,
          .communicator_local_rank = communicator_rank,
          .device_identity = 20 + local,
          .context_identity = 30 + local,
          .config_identity = 40};
}

TEST(DeepSeekRankBoundaryEndpointsTest, Pp1RequiresNoEndpoints) {
  auto endpoints = DeepSeekRankBoundaryEndpoints::Create(0, 1, nullptr,
                                                          nullptr);
  ASSERT_TRUE(endpoints.ok()) << endpoints.status().message();
  EXPECT_EQ(endpoints->incoming_transport(), nullptr);
  EXPECT_EQ(endpoints->outgoing_transport(), nullptr);

  auto manifest = topology_manifest(0, 1, 0, 0);
  EXPECT_FALSE(DeepSeekRankBoundaryEndpoints::ValidateTopology(
      0, 1, nullptr, &manifest).ok());
}

TEST(DeepSeekRankBoundaryEndpointsTest, MiddleRankRequiresExactTwoEdges) {
  auto incoming_manifest = topology_manifest(1, 0, 0, 1);
  auto outgoing_manifest = topology_manifest(1, 2, 1, 0);
  auto status = DeepSeekRankBoundaryEndpoints::ValidateTopology(
      1, 3, &incoming_manifest, &outgoing_manifest);
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST(DeepSeekRankBoundaryEndpointsTest, RejectsSwappedOrMixedGenerationEdges) {
  auto wrong_incoming = topology_manifest(1, 2, 1, 0);
  auto outgoing_manifest = topology_manifest(1, 2, 1, 0);
  EXPECT_FALSE(DeepSeekRankBoundaryEndpoints::ValidateTopology(
      1, 3, &wrong_incoming, &outgoing_manifest).ok());

  auto incoming_manifest = topology_manifest(1, 0, 0, 1);
  outgoing_manifest.communicator_generation = 8;
  EXPECT_FALSE(DeepSeekRankBoundaryEndpoints::ValidateTopology(
      1, 3, &incoming_manifest, &outgoing_manifest).ok());
}

}  // namespace
}  // namespace pih
