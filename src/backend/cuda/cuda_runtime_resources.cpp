#include "pih/backend/cuda/cuda_runtime_resources.h"

#include <utility>

namespace pih {

Result<CudaRuntimeResources> CudaRuntimeResources::Create(
    std::int32_t device_ordinal, std::uint32_t rank,
    std::uint64_t worker_generation, std::uint32_t context_flags,
    CudaRuntimeResourceDriver& driver) {
  if (device_ordinal < 0 || rank == UINT32_MAX || worker_generation == 0) {
    return Status::InvalidArgument("CUDA runtime resource identity is invalid");
  }
  auto context =
      driver.retain_primary_context(device_ordinal, context_flags);
  if (!context.ok()) return context.status();
  if (*context == 0) {
    driver.release_primary_context(device_ordinal, *context);
    return Status::FailedPrecondition("CUDA primary context handle is null");
  }

  CudaRuntimeResources resources(
      {device_ordinal, rank, worker_generation, context_flags, *context, 0, 0,
       0, 0, 0, 0, 0, 0},
      driver);
  const Status bound = driver.bind_runtime(device_ordinal, *context);
  if (!bound.ok()) return bound;
  auto stream = driver.create_nonblocking_stream(*context);
  if (!stream.ok()) return stream.status();
  if (*stream == 0) {
    return Status::FailedPrecondition("CUDA nonblocking stream handle is null");
  }
  resources.identity_.stream = *stream;
  auto scrub_stream = driver.create_nonblocking_stream(*context);
  if (!scrub_stream.ok()) return scrub_stream.status();
  if (*scrub_stream == 0) {
    return Status::FailedPrecondition(
        "CUDA scrub stream handle is null");
  }
  resources.identity_.scrub_stream = *scrub_stream;
  auto diagnostic_stream = driver.create_nonblocking_stream(*context);
  if (!diagnostic_stream.ok()) return diagnostic_stream.status();
  if (*diagnostic_stream == 0) {
    return Status::FailedPrecondition(
        "CUDA diagnostic stream handle is null");
  }
  resources.identity_.diagnostic_stream = *diagnostic_stream;
  auto deepseek_paging_stream = driver.create_nonblocking_stream(*context);
  if (!deepseek_paging_stream.ok()) return deepseek_paging_stream.status();
  if (*deepseek_paging_stream == 0) {
    return Status::FailedPrecondition(
        "CUDA DeepSeek paging stream handle is null");
  }
  resources.identity_.deepseek_paging_stream = *deepseek_paging_stream;
  auto event = driver.create_disable_timing_event(*context);
  if (!event.ok()) return event.status();
  if (*event == 0) {
    return Status::FailedPrecondition("CUDA dependency event handle is null");
  }
  resources.identity_.event = *event;
  auto scrub_event = driver.create_disable_timing_event(*context);
  if (!scrub_event.ok()) return scrub_event.status();
  if (*scrub_event == 0) {
    return Status::FailedPrecondition("CUDA scrub event handle is null");
  }
  resources.identity_.scrub_event = *scrub_event;
  auto diagnostic_event = driver.create_disable_timing_event(*context);
  if (!diagnostic_event.ok()) return diagnostic_event.status();
  if (*diagnostic_event == 0) {
    return Status::FailedPrecondition(
        "CUDA diagnostic event handle is null");
  }
  resources.identity_.diagnostic_event = *diagnostic_event;
  auto deepseek_expert_event = driver.create_disable_timing_event(*context);
  if (!deepseek_expert_event.ok()) return deepseek_expert_event.status();
  if (*deepseek_expert_event == 0) {
    return Status::FailedPrecondition(
        "CUDA DeepSeek expert event handle is null");
  }
  resources.identity_.deepseek_expert_event = *deepseek_expert_event;
  return resources;
}

CudaRuntimeResources::~CudaRuntimeResources() { reset(); }

CudaRuntimeResources::CudaRuntimeResources(
    CudaRuntimeResources&& other) noexcept
    : identity_(std::exchange(other.identity_, {})),
      driver_(std::exchange(other.driver_, nullptr)) {}

CudaRuntimeResources& CudaRuntimeResources::operator=(
    CudaRuntimeResources&& other) noexcept {
  if (this != &other) {
    reset();
    identity_ = std::exchange(other.identity_, {});
    driver_ = std::exchange(other.driver_, nullptr);
  }
  return *this;
}

void CudaRuntimeResources::reset() noexcept {
  if (driver_ == nullptr) return;
  if (identity_.deepseek_expert_event != 0) {
    driver_->destroy_event(identity_.deepseek_expert_event);
  }
  if (identity_.diagnostic_event != 0) {
    driver_->destroy_event(identity_.diagnostic_event);
  }
  if (identity_.scrub_event != 0) {
    driver_->destroy_event(identity_.scrub_event);
  }
  if (identity_.event != 0) driver_->destroy_event(identity_.event);
  if (identity_.deepseek_paging_stream != 0) {
    driver_->destroy_stream(identity_.deepseek_paging_stream);
  }
  if (identity_.diagnostic_stream != 0) {
    driver_->destroy_stream(identity_.diagnostic_stream);
  }
  if (identity_.scrub_stream != 0) {
    driver_->destroy_stream(identity_.scrub_stream);
  }
  if (identity_.stream != 0) driver_->destroy_stream(identity_.stream);
  if (identity_.context != 0) {
    driver_->release_primary_context(identity_.device_ordinal,
                                     identity_.context);
  }
  identity_ = {};
  driver_ = nullptr;
}

}  // namespace pih
