#include "engram_reduce.h"
#include "worker_communicator.h"

#include <cstring>
#include <limits>
#include <string>

namespace pih::deepseek_v41 {
namespace {
Status CoreStatus(const pih_status_v1& wire) {
  if (!pih_status_is_valid_v1(&wire))
    return Status::Internal("Collective transport returned invalid status");
  const std::string message(wire.message,
      ::strnlen(wire.message, sizeof(wire.message)));
  switch (wire.code) {
    case PIH_STATUS_OK_V1: return Status::Ok();
    case PIH_STATUS_INVALID_ARGUMENT_V1: return Status::InvalidArgument(message);
    case PIH_STATUS_FAILED_PRECONDITION_V1: return Status::FailedPrecondition(message);
    case PIH_STATUS_RESOURCE_EXHAUSTED_V1: return Status::ResourceExhausted(message);
    case PIH_STATUS_UNAVAILABLE_V1: return Status::Unavailable(message);
    case PIH_STATUS_DEADLINE_EXCEEDED_V1: return Status::DeadlineExceeded(message);
    case PIH_STATUS_INTERNAL_V1: return Status::Internal(message);
  }
  return Status::Internal("Collective transport status code invalid");
}
pih_transport_communicator_v1* Handle(std::uintptr_t value) noexcept {
  return reinterpret_cast<pih_transport_communicator_v1*>(value);
}
uint32_t ElementBytes(uint32_t datatype) noexcept {
  return datatype == PIH_TRANSPORT_BF16_V1 ? 2U : 4U;
}
template<class Launch>
Status ValidateReduction(const Launch& x, std::uintptr_t communicator,
                         uint32_t datatype) {
  const auto element = ElementBytes(datatype);
  if (!communicator || !x.stream || x.world_size <= 1 || x.world_size > 128 ||
      x.rank >= x.world_size || !x.output.address ||
      x.output.address % element || !x.output.bytes || x.output.bytes % element ||
      x.output.bytes / element > 4096ULL * 1048576 ||
      x.output.bytes > std::numeric_limits<std::uintptr_t>::max() - x.output.address ||
      x.output.bytes / element > std::numeric_limits<std::size_t>::max())
    return Status::InvalidArgument("Reduction requires bounded aligned device storage");
  const auto* api = CollectiveTransportApi();
  if (!api) return Status::FailedPrecondition("Collective transport not bound");
  return CoreStatus(api->validate_buffer(api->context, Handle(communicator),
      x.output.address, x.output.bytes, datatype, x.world_size, x.rank));
}
Result<EngramReductionState> SubmitReduce(EngramDeviceRegion output,
    std::uintptr_t stream, std::uintptr_t communicator, uint32_t datatype) {
  const auto* api = CollectiveTransportApi();
  if (!api) return Status::FailedPrecondition("Collective transport not bound");
  uint32_t transition = PIH_TRANSPORT_PENDING_V1;
  const auto status = CoreStatus(api->all_reduce_sum(api->context, Handle(communicator),
      output.address, output.bytes, datatype, stream, &transition));
  if (!status.ok()) return status;
  if (transition != PIH_TRANSPORT_PENDING_V1 && transition != PIH_TRANSPORT_READY_V1)
    return Status::Internal("Collective transport transition invalid");
  return transition == PIH_TRANSPORT_READY_V1
      ? EngramReductionState::kEnqueued : EngramReductionState::kPending;
}
}

EngramReduction::EngramReduction(EngramReduction&& other) noexcept
    : communicator_(other.communicator_), state_(other.state_) {
  other.communicator_ = 0;
  other.state_ = EngramReductionState::kFailed;
}
Status ValidateEngramReduction(const EngramLookupLaunch& x,
    std::uintptr_t communicator) {
  const auto layout = ValidateEngramLookup(x);
  if (!layout.ok()) return layout;
  return ValidateBf16Reduction({x.output, x.stream, x.world_size, x.rank}, communicator);
}
Status ValidateBf16Reduction(const Bf16ReductionLaunch& x,
    std::uintptr_t communicator) {
  return ValidateReduction(x, communicator, PIH_TRANSPORT_BF16_V1);
}
Status ValidateFp32Reduction(const Fp32ReductionLaunch& x,
    std::uintptr_t communicator) {
  return ValidateReduction(x, communicator, PIH_TRANSPORT_FP32_V1);
}
Status ValidateFp32Gather(const Fp32GatherLaunch& x,
    std::uintptr_t communicator) {
  const auto output = ValidateFp32Reduction(
      {x.output, x.stream, x.world_size, x.rank}, communicator);
  if (!output.ok()) return output;
  const auto input = ValidateFp32Reduction(
      {x.input, x.stream, x.world_size, x.rank}, communicator);
  if (!input.ok()) return input;
  if (x.output.bytes / x.world_size != x.input.bytes ||
      x.output.bytes % x.world_size ||
      (x.input.address < x.output.address + x.output.bytes &&
       x.output.address < x.input.address + x.input.bytes))
    return Status::InvalidArgument("FP32 gather requires disjoint rank-major output");
  return Status::Ok();
}
Result<EngramReduction> EngramReduction::Submit(const EngramLookupLaunch& x,
    std::uintptr_t communicator) {
  const auto validation = ValidateEngramReduction(x, communicator);
  if (!validation.ok()) return validation;
  return SubmitBf16({x.output, x.stream, x.world_size, x.rank}, communicator);
}
Result<EngramReduction> EngramReduction::SubmitBf16(const Bf16ReductionLaunch& x,
    std::uintptr_t communicator) {
  const auto validation = ValidateBf16Reduction(x, communicator);
  if (!validation.ok()) return validation;
  auto submitted = SubmitReduce(x.output, x.stream, communicator, PIH_TRANSPORT_BF16_V1);
  if (!submitted.ok()) return submitted.status();
  EngramReduction operation;
  operation.communicator_ = communicator;
  operation.state_ = *submitted;
  return operation;
}
Result<EngramReduction> EngramReduction::SubmitFp32(const Fp32ReductionLaunch& x,
    std::uintptr_t communicator) {
  const auto validation = ValidateFp32Reduction(x, communicator);
  if (!validation.ok()) return validation;
  auto submitted = SubmitReduce(x.output, x.stream, communicator, PIH_TRANSPORT_FP32_V1);
  if (!submitted.ok()) return submitted.status();
  EngramReduction operation;
  operation.communicator_ = communicator;
  operation.state_ = *submitted;
  return operation;
}
Result<EngramReductionState> EngramReduction::PollEnqueued() {
  if (state_ == EngramReductionState::kFailed || !communicator_)
    return Status::FailedPrecondition("Engram reduction failed or was moved from");
  // Keep polling even after enqueue: callers use this as the communicator
  // health check before and after observing CUDA completion. Enqueue is not
  // collective completion and must not hide a subsequent asynchronous error.
  const auto* api = CollectiveTransportApi();
  if (!api) { state_ = EngramReductionState::kFailed;
    return Status::FailedPrecondition("Collective transport not bound"); }
  uint32_t transition = PIH_TRANSPORT_PENDING_V1;
  const auto status = CoreStatus(api->poll_collective(api->context,
      Handle(communicator_), &transition));
  if (!status.ok() || (transition != PIH_TRANSPORT_PENDING_V1 &&
                       transition != PIH_TRANSPORT_READY_V1)) {
    state_ = EngramReductionState::kFailed;
    return status.ok() ? Status::Internal("Collective transition invalid") : status;
  }
  state_ = transition == PIH_TRANSPORT_READY_V1
      ? EngramReductionState::kEnqueued : EngramReductionState::kPending;
  return state_;
}
Result<EngramReduction> EngramReduction::SubmitGatherFp32(
    const Fp32GatherLaunch& x, std::uintptr_t communicator) {
  const auto validation = ValidateFp32Gather(x, communicator);
  if (!validation.ok()) return validation;
  const auto* api = CollectiveTransportApi();
  if (!api) return Status::FailedPrecondition("Collective transport not bound");
  uint32_t transition = PIH_TRANSPORT_PENDING_V1;
  const auto status = CoreStatus(api->all_gather(api->context, Handle(communicator),
      x.input.address, x.input.bytes, x.output.address, x.output.bytes,
      PIH_TRANSPORT_FP32_V1, x.stream, &transition));
  if (!status.ok()) return status;
  if (transition != PIH_TRANSPORT_PENDING_V1 && transition != PIH_TRANSPORT_READY_V1)
    return Status::Internal("Collective transport transition invalid");
  EngramReduction operation;
  operation.communicator_ = communicator;
  operation.state_ = transition == PIH_TRANSPORT_READY_V1
      ? EngramReductionState::kEnqueued : EngramReductionState::kPending;
  return operation;
}
}  // namespace pih::deepseek_v41
