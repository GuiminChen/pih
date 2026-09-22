#include "pih/backend/cuda/typed_copy_plan.h"

#include <limits>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

bool valid_purpose(CudaCopyPurpose purpose) {
  return purpose >= CudaCopyPurpose::kInput &&
         purpose <= CudaCopyPurpose::kSameRankMove;
}

Result<std::uintptr_t> checked_address(const CudaCopyEndpoint& endpoint,
                                       std::uint64_t bytes) {
  if (endpoint.allocation_base == 0 || endpoint.allocation_bytes == 0 ||
      endpoint.owner_id == 0 || endpoint.generation == 0 ||
      endpoint.rank == UINT32_MAX || endpoint.device_or_numa < 0) {
    return Status::InvalidArgument("CUDA copy endpoint identity is invalid");
  }
  auto end = checked_add_u64(endpoint.offset, bytes);
  if (!end.ok()) return end.status();
  if (end.value() > endpoint.allocation_bytes) {
    return Status::InvalidArgument("CUDA copy endpoint range exceeds owner");
  }
  if (endpoint.offset > std::numeric_limits<std::uintptr_t>::max() -
                            endpoint.allocation_base) {
    return Status::ResourceExhausted("CUDA copy endpoint address overflows");
  }
  return endpoint.allocation_base +
         static_cast<std::uintptr_t>(endpoint.offset);
}

bool direction_matches(CudaCopyKind kind, CudaCopyMemoryType source,
                       CudaCopyMemoryType destination) {
  switch (kind) {
    case CudaCopyKind::kHostToDevice:
      return source == CudaCopyMemoryType::kRegisteredPinnedHost &&
             destination == CudaCopyMemoryType::kDevice;
    case CudaCopyKind::kDeviceToHost:
      return source == CudaCopyMemoryType::kDevice &&
             destination == CudaCopyMemoryType::kRegisteredPinnedHost;
    case CudaCopyKind::kDeviceToDevice:
      return source == CudaCopyMemoryType::kDevice &&
             destination == CudaCopyMemoryType::kDevice;
  }
  return false;
}

bool purpose_matches(CudaCopyKind kind, CudaCopyPurpose purpose) {
  switch (kind) {
    case CudaCopyKind::kHostToDevice:
      return purpose == CudaCopyPurpose::kInput ||
             purpose == CudaCopyPurpose::kWeight;
    case CudaCopyKind::kDeviceToHost:
      return purpose == CudaCopyPurpose::kResult ||
             purpose == CudaCopyPurpose::kDiagnostic;
    case CudaCopyKind::kDeviceToDevice:
      return purpose == CudaCopyPurpose::kSameRankMove;
  }
  return false;
}

}  // namespace

Result<CudaTypedCopyPlan> CudaTypedCopyPlan::Create(
    std::uint64_t plan_id, CudaCopyPurpose purpose, CudaCopyKind kind,
    CudaCopyEndpoint source, CudaCopyEndpoint destination,
    std::uint64_t bytes, std::uint64_t required_alignment,
    std::uintptr_t primary_context_identity, DriverStreamHandle stream,
    std::uint64_t completion_event_generation) {
  if (plan_id == 0 || !valid_purpose(purpose) ||
      !direction_matches(kind, source.memory_type, destination.memory_type) ||
      !purpose_matches(kind, purpose) ||
      source.rank != destination.rank || source.owner_id == destination.owner_id ||
      required_alignment == 0 ||
      (required_alignment & (required_alignment - 1)) != 0) {
    return Status::InvalidArgument("CUDA typed copy identity is invalid");
  }
  auto source_address = checked_address(source, bytes);
  auto destination_address = checked_address(destination, bytes);
  if (!source_address.ok()) return source_address.status();
  if (!destination_address.ok()) return destination_address.status();
  if (source_address.value() % required_alignment != 0 ||
      destination_address.value() % required_alignment != 0) {
    return Status::InvalidArgument("CUDA copy address alignment is invalid");
  }
  if (kind == CudaCopyKind::kDeviceToDevice) {
    if (source.device_or_numa != destination.device_or_numa) {
      return Status::InvalidArgument("raw cross-device CUDA copy is forbidden");
    }
    auto source_end = checked_add_u64(source_address.value(), bytes);
    auto destination_end = checked_add_u64(destination_address.value(), bytes);
    if (!source_end.ok()) return source_end.status();
    if (!destination_end.ok()) return destination_end.status();
    if (bytes != 0 && source_address.value() < destination_end.value() &&
        destination_address.value() < source_end.value()) {
      return Status::InvalidArgument("same-device CUDA copy spans overlap");
    }
  } else {
    const auto device = kind == CudaCopyKind::kHostToDevice
                            ? destination.device_or_numa
                            : source.device_or_numa;
    if (device < 0) {
      return Status::InvalidArgument("CUDA copy device is invalid");
    }
  }
  if (bytes == 0) {
    if (stream != 0 || completion_event_generation != 0 ||
        primary_context_identity != 0) {
      return Status::InvalidArgument(
          "zero-byte CUDA copy must be a manifest no-op");
    }
  } else if (stream == 0 || completion_event_generation == 0 ||
             primary_context_identity == 0) {
    return Status::InvalidArgument(
        "nonempty CUDA copy requires context, stream, and completion");
  }
  return CudaTypedCopyPlan(plan_id, purpose, kind, source_address.value(),
                           source.owner_id, source.generation,
                           destination_address.value(), destination.owner_id,
                           destination.generation, bytes,
                           primary_context_identity, stream,
                           completion_event_generation);
}

Status CudaTypedCopyPlan::submit(TypedCopyDriver& driver) {
  if (submitted_) {
    return Status::FailedPrecondition("CUDA typed copy plan was already submitted");
  }
  submitted_ = true;
  if (bytes_ == 0) return Status::Ok();
  if (driver.context_identity() != context_identity_) {
    return Status::FailedPrecondition(
        "CUDA copy driver context differs from sealed plan");
  }
  return driver.copy(kind_, destination_address_, source_address_, bytes_,
                     stream_);
}

}  // namespace pih
