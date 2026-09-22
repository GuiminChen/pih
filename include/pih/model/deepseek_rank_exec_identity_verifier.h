#pragma once

#include "pih/model/deepseek_rank_process_supervisor.h"

namespace pih {

struct DeepSeekRankExecObservation final {
  DeepSeekRankProcessManifest argument_manifest;
  std::uint64_t actual_process_identity = 0;
  std::uint64_t actual_parent_process_identity = 0;
  std::uint64_t controller_pidfd_target_identity = 0;
  std::uint64_t actual_physical_device_identity = 0;
  Sha256Digest actual_physical_device_uuid_commitment{};
  std::int32_t actual_startup_device_ordinal = -1;
  std::int32_t parent_death_signal = 0;
  bool control_channel_is_seqpacket = false;
  bool challenge_received_on_control_channel = false;
};

Result<DeepSeekRankExecReady> verify_deepseek_rank_exec_identity(
    const DeepSeekRankExecChallenge& challenge,
    const DeepSeekRankExecObservation& observation);

}  // namespace pih
