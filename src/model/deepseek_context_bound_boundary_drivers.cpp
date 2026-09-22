#include "pih/model/deepseek_context_bound_boundary_drivers.h"

namespace pih {

Result<DeepSeekContextBoundCompletionDriver>
DeepSeekContextBoundCompletionDriver::Create(
    std::uintptr_t context_identity, CompletionEventDriver& events,
    CompletionLastErrorProbe& last_error,
    DeepSeekNcclContextActivator& activator) {
  if (context_identity == 0) {
    return Status::InvalidArgument(
        "DeepSeek context-bound completion context is null");
  }
  return DeepSeekContextBoundCompletionDriver(
      context_identity, events, last_error, activator);
}

Status DeepSeekContextBoundCompletionDriver::activate() {
  return activator_->activate(context_identity_);
}

Status DeepSeekContextBoundCompletionDriver::record(
    DriverEventHandle event, DriverStreamHandle stream) {
  auto status = activate();
  if (!status.ok()) return status;
  return events_->record(event, stream);
}

Result<CudaEventQueryResult> DeepSeekContextBoundCompletionDriver::query(
    DriverEventHandle event) {
  auto status = activate();
  if (!status.ok()) return status;
  return events_->query(event);
}

Status DeepSeekContextBoundCompletionDriver::require_clean_last_error() {
  auto status = activate();
  if (!status.ok()) return status;
  return last_error_->require_clean_last_error();
}

Result<DeepSeekContextBoundWarmupPayloadOperations>
DeepSeekContextBoundWarmupPayloadOperations::Create(
    std::uintptr_t context_identity,
    DeepSeekNcclWarmupPayloadOperations& payload,
    DeepSeekNcclContextActivator& activator) {
  if (context_identity == 0) {
    return Status::InvalidArgument(
        "DeepSeek context-bound warm-up payload context is null");
  }
  return DeepSeekContextBoundWarmupPayloadOperations(
      context_identity, payload, activator);
}

Status DeepSeekContextBoundWarmupPayloadOperations::activate() {
  return activator_->activate(context_identity_);
}

Status DeepSeekContextBoundWarmupPayloadOperations::prepare(
    DeepSeekNcclRole role, void* device_buffer, std::uint64_t bytes,
    DriverStreamHandle stream, std::uint64_t pattern_identity) {
  auto status = activate();
  if (!status.ok()) return status;
  return payload_->prepare(role, device_buffer, bytes, stream,
                           pattern_identity);
}

Result<Sha256Digest> DeepSeekContextBoundWarmupPayloadOperations::digest(
    const void* device_buffer, std::uint64_t bytes) {
  auto status = activate();
  if (!status.ok()) return status;
  return payload_->digest(device_buffer, bytes);
}

}  // namespace pih
