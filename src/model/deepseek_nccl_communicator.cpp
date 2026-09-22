#include "pih/model/deepseek_nccl_communicator.h"

namespace pih {

Result<DeepSeekNcclReleaseConfig> DeepSeekNcclReleaseConfig::Create(
    DeepSeekGpuArchitecture architecture, std::uint32_t nccl_release,
    std::uint32_t config_struct_bytes) {
  if (static_cast<std::uint8_t>(architecture) >
          static_cast<std::uint8_t>(DeepSeekGpuArchitecture::kSm90) ||
      nccl_release != 23102 || config_struct_bytes == 0) {
    return Status::InvalidArgument("DeepSeek NCCL release config is invalid");
  }
  DeepSeekNcclReleaseConfig config;
  config.nccl_release = nccl_release;
  config.config_struct_bytes = config_struct_bytes;
  config.cga_cluster_size =
      architecture == DeepSeekGpuArchitecture::kSm90 ? 4 : 0;
  return config;
}

Result<DeepSeekNcclCommunicator> DeepSeekNcclCommunicator::Create(
    DeepSeekNcclCommunicatorManifest manifest) {
  if (manifest.engine_epoch == 0 || manifest.communicator_generation == 0 ||
      manifest.bootstrap_lease_id == 0 ||
      manifest.bootstrap_commitment_id == 0 ||
      manifest.device_identity == 0 || manifest.context_identity == 0 ||
      manifest.config_identity == 0 || manifest.communicator_local_rank > 1 ||
      manifest.local_global_rank == manifest.peer_global_rank) {
    return Status::InvalidArgument("DeepSeek NCCL communicator manifest is invalid");
  }
  const bool lower = manifest.communicator_local_rank == 0;
  if ((lower && (manifest.local_global_rank != manifest.edge_id ||
                 manifest.peer_global_rank != manifest.edge_id + 1)) ||
      (!lower && (manifest.local_global_rank != manifest.edge_id + 1 ||
                  manifest.peer_global_rank != manifest.edge_id))) {
    return Status::InvalidArgument("DeepSeek NCCL communicator edge mapping is invalid");
  }
  return DeepSeekNcclCommunicator(manifest);
}

Status DeepSeekNcclCommunicator::fail(Status status) {
  state_ = DeepSeekNcclCommunicatorState::kFailed;
  return status.ok() ? Status::Internal("DeepSeek NCCL communicator failed") : status;
}

Status DeepSeekNcclCommunicator::accept_bootstrap(
    bool both_endpoints_validated) {
  if (state_ != DeepSeekNcclCommunicatorState::kAbsent) {
    return Status::FailedPrecondition("DeepSeek NCCL bootstrap state is invalid");
  }
  if (!both_endpoints_validated) {
    return fail(Status::InvalidArgument(
        "DeepSeek NCCL endpoint bootstrap acknowledgement mismatched"));
  }
  state_ = DeepSeekNcclCommunicatorState::kBothEndpointsValidated;
  return Status::Ok();
}

Status DeepSeekNcclCommunicator::begin_init(
    DeepSeekNcclCommunicatorDriver& driver) {
  if (state_ != DeepSeekNcclCommunicatorState::kBothEndpointsValidated) {
    return Status::FailedPrecondition("DeepSeek NCCL communicator is not validated");
  }
  init_attempted_ = true;
  auto result = driver.init_rank_config(
      manifest_.bootstrap_lease_id, 2, manifest_.communicator_local_rank);
  if (!result.ok()) return fail(result.status());
  if (*result == DeepSeekNcclAsyncStatus::kError) {
    return fail(Status::Internal("DeepSeek NCCL communicator init failed"));
  }
  state_ = *result == DeepSeekNcclAsyncStatus::kSuccess
               ? DeepSeekNcclCommunicatorState::kCommunicatorReady
               : DeepSeekNcclCommunicatorState::kInitInProgress;
  return Status::Ok();
}

Status DeepSeekNcclCommunicator::poll_init(
    DeepSeekNcclCommunicatorDriver& driver) {
  if (state_ != DeepSeekNcclCommunicatorState::kInitInProgress) {
    return Status::FailedPrecondition("DeepSeek NCCL init is not pending");
  }
  auto result = driver.async_status();
  if (!result.ok()) return fail(result.status());
  if (*result == DeepSeekNcclAsyncStatus::kError) {
    return fail(Status::Internal("DeepSeek NCCL async init failed"));
  }
  if (*result == DeepSeekNcclAsyncStatus::kInProgress) {
    return Status::Unavailable("DeepSeek NCCL init remains in progress");
  }
  state_ = DeepSeekNcclCommunicatorState::kCommunicatorReady;
  return Status::Ok();
}

Status DeepSeekNcclCommunicator::reconcile(
    DeepSeekNcclCommunicatorDriver& driver) {
  if (state_ != DeepSeekNcclCommunicatorState::kCommunicatorReady) {
    return Status::FailedPrecondition("DeepSeek NCCL communicator is not ready");
  }
  auto count = driver.communicator_count();
  auto rank = driver.communicator_user_rank();
  auto device = driver.device_identity();
  auto context = driver.context_identity();
  if (!count.ok()) return fail(count.status());
  if (!rank.ok()) return fail(rank.status());
  if (!device.ok()) return fail(device.status());
  if (!context.ok()) return fail(context.status());
  if (*count != 2 || *rank != manifest_.communicator_local_rank ||
      *device != manifest_.device_identity ||
      *context != manifest_.context_identity) {
    return fail(Status::Internal("DeepSeek NCCL communicator identity drifted"));
  }
  state_ = DeepSeekNcclCommunicatorState::kReconciled;
  return Status::Ok();
}

Status DeepSeekNcclCommunicator::mark_warmed(bool min_boundary_passed,
                                             bool max_boundary_passed) {
  if (state_ != DeepSeekNcclCommunicatorState::kReconciled) {
    return Status::FailedPrecondition("DeepSeek NCCL communicator is not reconciled");
  }
  if (!min_boundary_passed || !max_boundary_passed) {
    return fail(Status::Internal("DeepSeek NCCL boundary warm-up failed"));
  }
  state_ = DeepSeekNcclCommunicatorState::kWarmed;
  return Status::Ok();
}

Status DeepSeekNcclCommunicator::seal() {
  if (state_ != DeepSeekNcclCommunicatorState::kWarmed) {
    return Status::FailedPrecondition("DeepSeek NCCL communicator is not warmed");
  }
  state_ = DeepSeekNcclCommunicatorState::kSealed;
  return Status::Ok();
}

Status DeepSeekNcclCommunicator::begin_finalize(
    DeepSeekNcclCommunicatorDriver& driver) {
  if (state_ != DeepSeekNcclCommunicatorState::kSealed) {
    return Status::FailedPrecondition("DeepSeek NCCL communicator is not sealed");
  }
  auto result = driver.finalize();
  if (!result.ok()) return fail(result.status());
  if (*result == DeepSeekNcclAsyncStatus::kError) {
    return fail(Status::Internal("DeepSeek NCCL finalize failed"));
  }
  state_ = *result == DeepSeekNcclAsyncStatus::kSuccess
               ? DeepSeekNcclCommunicatorState::kFinalizedSuccess
               : DeepSeekNcclCommunicatorState::kFinalizeInProgress;
  return Status::Ok();
}

Status DeepSeekNcclCommunicator::poll_finalize(
    DeepSeekNcclCommunicatorDriver& driver) {
  if (state_ != DeepSeekNcclCommunicatorState::kFinalizeInProgress) {
    return Status::FailedPrecondition("DeepSeek NCCL finalize is not pending");
  }
  auto result = driver.async_status();
  if (!result.ok()) return fail(result.status());
  if (*result == DeepSeekNcclAsyncStatus::kError) {
    return fail(Status::Internal("DeepSeek NCCL async finalize failed"));
  }
  if (*result == DeepSeekNcclAsyncStatus::kInProgress) {
    return Status::Unavailable("DeepSeek NCCL finalize remains in progress");
  }
  state_ = DeepSeekNcclCommunicatorState::kFinalizedSuccess;
  return Status::Ok();
}

Status DeepSeekNcclCommunicator::destroy(
    DeepSeekNcclCommunicatorDriver& driver) {
  if (state_ != DeepSeekNcclCommunicatorState::kFinalizedSuccess) {
    return Status::FailedPrecondition("DeepSeek NCCL communicator is not finalized");
  }
  const auto status = driver.destroy();
  if (!status.ok()) return fail(status);
  state_ = DeepSeekNcclCommunicatorState::kDestroyed;
  return Status::Ok();
}

Status DeepSeekNcclCommunicator::abort(
    DeepSeekNcclCommunicatorDriver& driver) {
  if (abort_called_) return Status::Ok();
  if (state_ == DeepSeekNcclCommunicatorState::kDestroyed) {
    return Status::FailedPrecondition("destroyed DeepSeek NCCL communicator cannot abort");
  }
  if (!init_attempted_) {
    abort_called_ = true;
    state_ = DeepSeekNcclCommunicatorState::kAborted;
    return Status::Ok();
  }
  const auto status = driver.abort();
  abort_called_ = true;
  if (!status.ok()) return fail(status);
  state_ = DeepSeekNcclCommunicatorState::kAborted;
  return Status::Ok();
}

}  // namespace pih
