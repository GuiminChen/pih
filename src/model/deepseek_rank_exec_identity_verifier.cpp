#include "pih/model/deepseek_rank_exec_identity_verifier.h"

namespace pih {
namespace {

bool same_manifest(const DeepSeekRankProcessManifest& a,
                   const DeepSeekRankProcessManifest& b) noexcept {
  return a.engine_epoch == b.engine_epoch &&
         a.worker_generation == b.worker_generation &&
         a.world_size == b.world_size && a.rank == b.rank &&
         a.physical_device_identity == b.physical_device_identity &&
         a.process_manifest_identity == b.process_manifest_identity &&
         a.startup_device_ordinal == b.startup_device_ordinal &&
         a.startup_deadline_ns == b.startup_deadline_ns;
}

}  // namespace

Result<DeepSeekRankExecReady> verify_deepseek_rank_exec_identity(
    const DeepSeekRankExecChallenge& challenge,
    const DeepSeekRankExecObservation& observation) {
  const auto& m = challenge.manifest;
  const auto& h = challenge.handle;
  if (challenge.protocol_version != 1 || challenge.challenge_identity == 0 ||
      challenge.controller_process_identity == 0 ||
      h.process_identity == 0 || h.pidfd_identity == 0 ||
      h.control_identity == 0 || !same_manifest(m, observation.argument_manifest) ||
      observation.actual_process_identity != h.process_identity ||
      observation.actual_parent_process_identity != challenge.controller_process_identity ||
      observation.controller_pidfd_target_identity != challenge.controller_process_identity ||
      observation.actual_physical_device_identity != m.physical_device_identity ||
      m.physical_device_uuid_commitment == Sha256Digest{} ||
      observation.actual_physical_device_uuid_commitment !=
          m.physical_device_uuid_commitment ||
      observation.actual_startup_device_ordinal != m.startup_device_ordinal ||
      m.startup_deadline_ns == 0 ||
      observation.parent_death_signal != 9 ||
      !observation.control_channel_is_seqpacket ||
      !observation.challenge_received_on_control_channel) {
    return Status::FailedPrecondition(
        "DeepSeek rank exec identity reconciliation failed");
  }
  return DeepSeekRankExecReady{{m.engine_epoch, m.worker_generation, m.rank,
                                m.physical_device_identity,
                                m.process_manifest_identity,
                                h.process_identity, h.pidfd_identity,
                                h.control_identity,
                                m.physical_device_uuid_commitment,
                                m.startup_device_ordinal,
                                m.startup_deadline_ns},
                               challenge.challenge_identity};
}

}  // namespace pih
