#pragma once

#include <array>
#include <span>
#include <string_view>

#include "pih/model/deepseek_rank_exec_identity_verifier.h"
#include "pih/model/deepseek_rank_post_exec_resource_receipt.h"
#include "pih/model/deepseek_rank_serving_protocol.h"

namespace pih {

inline constexpr std::size_t kDeepSeekRankChallengeBytes = 132;
inline constexpr std::size_t kDeepSeekRankReadyBytes = 120;
inline constexpr std::size_t
    kDeepSeekRankPostExecResourceObservationBytes = 176;
inline constexpr std::size_t
    kDeepSeekRankPostExecResourceAuthorityBytes = 256;
inline constexpr std::size_t kDeepSeekRankServingCommandBytes = 168;
inline constexpr std::size_t kDeepSeekRankServingCompletionBytes = 152;
inline constexpr std::string_view
    kDeepSeekRankPostExecResourceObservationFrameAbi =
        "pih_deepseek_rank_post_exec_resource_observation_frame_v1";
inline constexpr std::string_view
    kDeepSeekRankPostExecResourceAuthorityFrameAbi =
        "pih_deepseek_rank_post_exec_resource_authority_frame_v1";

std::array<std::byte, kDeepSeekRankChallengeBytes>
encode_deepseek_rank_challenge(const DeepSeekRankExecChallenge& value);
Result<DeepSeekRankExecChallenge> decode_deepseek_rank_challenge(
    std::span<const std::byte> bytes);
std::array<std::byte, kDeepSeekRankReadyBytes>
encode_deepseek_rank_ready(const DeepSeekRankExecReady& value);
Result<DeepSeekRankExecReady> decode_deepseek_rank_ready(
    std::span<const std::byte> bytes);
std::array<std::byte, kDeepSeekRankPostExecResourceObservationBytes>
encode_deepseek_rank_post_exec_resource_observation(
    const DeepSeekRankPostExecResourceObservation& value);
Result<DeepSeekRankPostExecResourceObservation>
decode_deepseek_rank_post_exec_resource_observation(
    std::span<const std::byte> bytes);
std::array<std::byte, kDeepSeekRankPostExecResourceAuthorityBytes>
encode_deepseek_rank_post_exec_resource_authority(
    const DeepSeekRankPostExecResourceAuthority& value);
Result<DeepSeekRankPostExecResourceAuthority>
decode_deepseek_rank_post_exec_resource_authority(
    std::span<const std::byte> bytes);
std::array<std::byte, kDeepSeekRankServingCommandBytes>
encode_deepseek_rank_serving_command(const DeepSeekRankServingCommand& value);
Result<DeepSeekRankServingCommand> decode_deepseek_rank_serving_command(
    std::span<const std::byte> bytes);
std::array<std::byte, kDeepSeekRankServingCompletionBytes>
encode_deepseek_rank_serving_completion(
    const DeepSeekRankServingCompletion& value);
Result<DeepSeekRankServingCompletion> decode_deepseek_rank_serving_completion(
    std::span<const std::byte> bytes);

}  // namespace pih
