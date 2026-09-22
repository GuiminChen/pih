#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "pih/model/deepseek_rank_post_mapping_resource_seal.h"

namespace pih {

inline constexpr std::size_t
    kDeepSeekRankPostMappingResourceObservationFrameBytes = 388;
inline constexpr std::size_t
    kDeepSeekRankPostMappingResourceAuthorityFrameBytes = 500;
inline constexpr std::string_view
    kDeepSeekRankPostMappingResourceObservationFrameAbi =
        "pih_deepseek_rank_post_mapping_resource_observation_frame_v1";
inline constexpr std::string_view
    kDeepSeekRankPostMappingResourceAuthorityFrameAbi =
        "pih_deepseek_rank_post_mapping_resource_authority_frame_v1";

struct DeepSeekRankPostMappingResourceAuthority final {
  std::uint32_t protocol_version = 0;
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t world_size = 0;
  std::uint32_t rank = 0;
  std::uint64_t process_manifest_identity = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  std::uint64_t control_identity = 0;
  std::uint64_t challenge_identity = 0;
  Sha256Digest manifest_root{};
  Sha256Digest spawn_resource_plan_root{};
  Sha256Digest capacity_plan_instance_root{};
  Sha256Digest os_resource_envelope_root{};
  Sha256Digest first_resource_seal_root{};
  Sha256Digest first_resource_plan_set_root{};
  Sha256Digest descriptor_transaction_root{};
  Sha256Digest metadata_transaction_root{};
  Sha256Digest metadata_root{};
  Sha256Digest expected_mapping_owner_root{};
  std::uint64_t deadline_ns = 0;
  std::uint64_t expected_mapped_interval_bytes = 0;
  ArtifactImmutabilityMode expected_immutability_mode =
      ArtifactImmutabilityMode::kUncalibrated;
  bool expected_source_catalog_production_eligible = false;
  bool dspark_enabled = false;
  DeepSeekRankPostExecResourcePlan resource_plan;
};

Result<Sha256Digest>
compile_deepseek_rank_post_mapping_resource_observation_frame_root(
    const DeepSeekRankPostMappingResourceObservation& value);
Result<Sha256Digest>
compile_deepseek_rank_post_mapping_resource_authority_frame_root(
    const DeepSeekRankPostMappingResourceAuthority& value);

Result<std::array<
    std::byte, kDeepSeekRankPostMappingResourceObservationFrameBytes>>
encode_deepseek_rank_post_mapping_resource_observation(
    const DeepSeekRankPostMappingResourceObservation& value);
Result<DeepSeekRankPostMappingResourceObservation>
decode_deepseek_rank_post_mapping_resource_observation(
    std::span<const std::byte> bytes);
Result<std::array<
    std::byte, kDeepSeekRankPostMappingResourceAuthorityFrameBytes>>
encode_deepseek_rank_post_mapping_resource_authority(
    const DeepSeekRankPostMappingResourceAuthority& value);
Result<DeepSeekRankPostMappingResourceAuthority>
decode_deepseek_rank_post_mapping_resource_authority(
    std::span<const std::byte> bytes);

}  // namespace pih
