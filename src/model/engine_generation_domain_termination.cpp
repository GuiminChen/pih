#include "pih/model/engine_generation_domain_termination.h"

namespace pih {

Result<EngineGenerationDomainTermination>
EngineGenerationDomainTermination::Create(
    EngineGenerationDomainDriver& driver) {
  return EngineGenerationDomainTermination(driver);
}

Status EngineGenerationDomainTermination::force_kill() {
  if (force_kill_issued_)
    return Status::FailedPrecondition(
        "engine generation domain kill was already issued");
  const auto status = driver_->force_kill_domain();
  if (!status.ok()) return status;
  force_kill_issued_ = true;
  return Status::Ok();
}

Result<bool> EngineGenerationDomainTermination::poll_empty() {
  return driver_->domain_empty();
}

}  // namespace pih
