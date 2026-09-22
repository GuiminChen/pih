#include "pih/model/deepseek_nccl_edge_endpoint.h"

#include <new>
#include <utility>

namespace pih {

DeepSeekNcclEdgeEndpoint& DeepSeekNcclEdgeEndpoint::operator=(
    DeepSeekNcclEdgeEndpoint&& other) noexcept {
  if (this != &other) {
    this->~DeepSeekNcclEdgeEndpoint();
    ::new (static_cast<void*>(this))
        DeepSeekNcclEdgeEndpoint(std::move(other));
  }
  return *this;
}

Result<DeepSeekNcclEdgeEndpoint> DeepSeekNcclEdgeEndpoint::Create(
    DeepSeekNcclCApi& api,
    std::unique_ptr<DeepSeekNcclBootstrapLease> bootstrap,
    DeepSeekNcclReleaseConfig config,
    DeepSeekNcclCommunicatorManifest manifest) {
  if (bootstrap == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek NCCL edge endpoint requires bootstrap lease ownership");
  }
  auto communicator = DeepSeekNcclCommunicator::Create(manifest);
  if (!communicator.ok()) return communicator.status();
  auto driver = DeepSeekNcclApiDriver::Create(
      api, *bootstrap, std::move(config), manifest.engine_epoch,
      manifest.edge_id, manifest.bootstrap_lease_id, manifest.device_identity,
      manifest.context_identity);
  if (!driver.ok()) return driver.status();
  return DeepSeekNcclEdgeEndpoint(
      std::move(bootstrap),
      std::make_unique<DeepSeekNcclApiDriver>(std::move(*driver)),
      std::make_unique<DeepSeekNcclCommunicator>(std::move(*communicator)));
}

DeepSeekNcclEdgeEndpoint::~DeepSeekNcclEdgeEndpoint() {
  if (driver_ == nullptr || communicator_ == nullptr) return;
  const auto current = communicator_->state();
  if (current != DeepSeekNcclCommunicatorState::kDestroyed &&
      current != DeepSeekNcclCommunicatorState::kAborted) {
    (void)communicator_->abort(*driver_);
  }
}

Status DeepSeekNcclEdgeEndpoint::accept_bootstrap(
    bool both_endpoints_validated) {
  return communicator_->accept_bootstrap(both_endpoints_validated);
}

Status DeepSeekNcclEdgeEndpoint::begin_init() {
  return communicator_->begin_init(*driver_);
}

Status DeepSeekNcclEdgeEndpoint::poll_init() {
  return communicator_->poll_init(*driver_);
}

Status DeepSeekNcclEdgeEndpoint::reconcile() {
  return communicator_->reconcile(*driver_);
}

Status DeepSeekNcclEdgeEndpoint::mark_warmed(bool min_boundary_passed,
                                              bool max_boundary_passed) {
  return communicator_->mark_warmed(min_boundary_passed, max_boundary_passed);
}

Status DeepSeekNcclEdgeEndpoint::seal() { return communicator_->seal(); }

Status DeepSeekNcclEdgeEndpoint::begin_finalize() {
  return communicator_->begin_finalize(*driver_);
}

Status DeepSeekNcclEdgeEndpoint::poll_finalize() {
  return communicator_->poll_finalize(*driver_);
}

Status DeepSeekNcclEdgeEndpoint::destroy() {
  return communicator_->destroy(*driver_);
}

Status DeepSeekNcclEdgeEndpoint::abort() {
  return communicator_->abort(*driver_);
}

DeepSeekBoundaryTransportDriver* DeepSeekNcclEdgeEndpoint::transport()
    noexcept {
  return communicator_->state() == DeepSeekNcclCommunicatorState::kSealed
             ? driver_.get()
             : nullptr;
}

DeepSeekBoundaryTransportDriver*
DeepSeekNcclEdgeEndpoint::warmup_transport() noexcept {
  return communicator_->state() == DeepSeekNcclCommunicatorState::kReconciled
             ? driver_.get()
             : nullptr;
}

bool DeepSeekNcclEdgeEndpoint::bootstrap_zeroized() const noexcept {
  return bootstrap_->zeroized();
}

DeepSeekNcclCommunicatorState DeepSeekNcclEdgeEndpoint::state()
    const noexcept {
  return communicator_->state();
}

const DeepSeekNcclCommunicatorManifest& DeepSeekNcclEdgeEndpoint::manifest()
    const noexcept {
  return communicator_->manifest();
}

}  // namespace pih
