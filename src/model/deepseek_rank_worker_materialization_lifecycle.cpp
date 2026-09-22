#include "pih/model/deepseek_rank_worker_materialization_lifecycle.h"

namespace pih {

Result<DeepSeekRankWorkerMaterializationLifecycle>
DeepSeekRankWorkerMaterializationLifecycle::Create(
    DeepSeekRankWorkerStartupProtocol& startup,
    DeepSeekRankWorkerMaterializationApplication& application) {
  return DeepSeekRankWorkerMaterializationLifecycle(startup, application);
}

Status DeepSeekRankWorkerMaterializationLifecycle::fail(Status cause) noexcept {
  state_ = DeepSeekRankWorkerMaterializationState::kFailed;
  return cause.ok()
             ? Status::Internal("DeepSeek rank worker materialization failed")
             : cause;
}

Status DeepSeekRankWorkerMaterializationLifecycle::advance() {
  if (state_ == DeepSeekRankWorkerMaterializationState::kReady) {
    return Status::Ok();
  }
  if (state_ == DeepSeekRankWorkerMaterializationState::kFailed) {
    return Status::FailedPrecondition(
        "DeepSeek rank worker materialization lifecycle is failed");
  }

  Status status = Status::Ok();
  switch (state_) {
    case DeepSeekRankWorkerMaterializationState::kAwaitingResourceBarrier:
      status = startup_->run_resource_barrier();
      if (!status.ok()) {
        return status.code() == StatusCode::kUnavailable ? status
                                                          : fail(status);
      }
      state_ =
          DeepSeekRankWorkerMaterializationState::kAwaitingArtifactTransfer;
      return Status::Ok();

    case DeepSeekRankWorkerMaterializationState::kAwaitingArtifactTransfer:
      status = startup_->run_artifact_transfer();
      if (!status.ok()) {
        return status.code() == StatusCode::kUnavailable ? status
                                                          : fail(status);
      }
      state_ = DeepSeekRankWorkerMaterializationState::kMaterializing;
      return Status::Ok();

    case DeepSeekRankWorkerMaterializationState::kMaterializing:
      status = application_->materialize();
      if (!status.ok()) return fail(status);
      state_ = DeepSeekRankWorkerMaterializationState::kSendingCompletion;
      return Status::Ok();

    case DeepSeekRankWorkerMaterializationState::kSendingCompletion:
      if (!completion_begun_) {
        status = application_->begin_completion();
        if (!status.ok()) return fail(status);
        completion_begun_ = true;
      }
      status = startup_->run_materialization_completion();
      if (!status.ok()) {
        return status.code() == StatusCode::kUnavailable ? status
                                                          : fail(status);
      }
      if (!startup_->materialization_completion_sent()) {
        return fail(Status::Internal(
            "DeepSeek rank worker completion returned before send"));
      }
      state_ = DeepSeekRankWorkerMaterializationState::kReady;
      return Status::Ok();

    case DeepSeekRankWorkerMaterializationState::kReady:
    case DeepSeekRankWorkerMaterializationState::kFailed:
      break;
  }
  return fail(Status::Internal(
      "DeepSeek rank worker materialization state is invalid"));
}

}  // namespace pih
