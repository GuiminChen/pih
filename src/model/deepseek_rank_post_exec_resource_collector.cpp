#include "pih/model/deepseek_rank_post_exec_resource_collector.h"

#include <utility>

namespace pih {
namespace {

bool nonzero(const Sha256Digest& value) noexcept {
  for (const auto byte : value.bytes) {
    if (byte != std::byte{0}) return true;
  }
  return false;
}

Status validate_identity(
    const DeepSeekRankPostExecResourceIdentity& identity) {
  if (identity.engine_epoch == 0 || identity.worker_generation == 0 ||
      identity.rank >= 4 || identity.process_identity == 0 ||
      identity.challenge_identity == 0 ||
      !nonzero(identity.capacity_plan_instance_root) ||
      !nonzero(identity.os_resource_envelope_root)) {
    return Status::InvalidArgument(
        "DeepSeek rank post-exec resource identity is invalid");
  }
  return Status::Ok();
}

Status validate_snapshot(
    const DeepSeekRankPostExecResourceIdentity& identity,
    const DeepSeekRankPostExecResourceSnapshot& snapshot) {
  if (snapshot.process_identity != identity.process_identity ||
      snapshot.task_count == 0 || snapshot.open_fd_count == 0 ||
      snapshot.vma_count == 0 || snapshot.rlimit_nofile_soft == 0 ||
      snapshot.rlimit_nofile_hard < snapshot.rlimit_nofile_soft ||
      snapshot.rlimit_nofile_hard > snapshot.fs_nr_open ||
      snapshot.vm_max_map_count == 0 || !snapshot.non_dumpable) {
    return Status::FailedPrecondition(
        "DeepSeek rank post-exec resource snapshot is invalid");
  }
  return Status::Ok();
}

}  // namespace

Result<StableDeepSeekRankPostExecResourceCollector>
StableDeepSeekRankPostExecResourceCollector::Create(
    DeepSeekRankPostExecResourceProbe& probe,
    std::uint32_t maximum_samples) {
  if (maximum_samples < 2 || maximum_samples > 8) {
    return Status::InvalidArgument(
        "DeepSeek rank post-exec stabilization bound is invalid");
  }
  return StableDeepSeekRankPostExecResourceCollector(
      probe, maximum_samples);
}

Result<DeepSeekRankPostExecResourceObservation>
StableDeepSeekRankPostExecResourceCollector::collect(
    const DeepSeekRankPostExecResourceIdentity& identity) {
  auto status = validate_identity(identity);
  if (!status.ok()) return status;
  auto previous = probe_->sample();
  if (!previous.ok()) return previous.status();
  status = validate_snapshot(identity, *previous);
  if (!status.ok()) return status;
  for (std::uint32_t sample = 1; sample < maximum_samples_; ++sample) {
    auto current = probe_->sample();
    if (!current.ok()) return current.status();
    status = validate_snapshot(identity, *current);
    if (!status.ok()) return status;
    if (*previous != *current) {
      previous = std::move(current);
      continue;
    }
    return DeepSeekRankPostExecResourceObservation{
        identity.engine_epoch,
        identity.worker_generation,
        identity.rank,
        current->process_identity,
        identity.challenge_identity,
        identity.capacity_plan_instance_root,
        identity.os_resource_envelope_root,
        current->task_count,
        current->open_fd_count,
        current->scm_rights_inflight_fd_count,
        current->vma_count,
        current->rlimit_nofile_soft,
        current->rlimit_nofile_hard,
        current->fs_nr_open,
        current->vm_max_map_count,
        current->non_dumpable};
  }
  return Status::Unavailable(
      "DeepSeek rank post-exec resources did not stabilize");
}

}  // namespace pih
