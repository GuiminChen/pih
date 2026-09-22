#include "runtime_resources.h"
#include <utility>

namespace pih::qwen_plugin {
Result<RuntimeResources> RuntimeResources::Create(std::int32_t ordinal,
    std::uint32_t rank, std::uint64_t generation, std::uint32_t flags,
    CudaRuntimeResourceDriver& driver) {
  if (ordinal < 0 || rank == UINT32_MAX || generation == 0)
    return Status::InvalidArgument("Qwen runtime resource identity invalid");
  RuntimeResources result(driver);
  auto& id = result.identity_;
  id.device_ordinal = ordinal;
  id.rank = rank;
  id.worker_generation = generation;
  id.context_flags = flags;
  auto context = driver.retain_primary_context(ordinal, flags);
  if (!context.ok()) return context.status();
  if (!*context) return Status::FailedPrecondition("Qwen primary context is null");
  id.context = *context;
  auto bound = driver.bind_runtime(ordinal, id.context);
  if (!bound.ok()) return bound;
  for (auto* stream : {&id.stream, &id.scrub_stream, &id.diagnostic_stream}) {
    auto created = driver.create_nonblocking_stream(id.context);
    if (!created.ok()) return created.status();
    if (!*created) return Status::FailedPrecondition("Qwen stream is null");
    *stream = *created;
  }
  for (auto* event : {&id.event, &id.scrub_event, &id.diagnostic_event}) {
    auto created = driver.create_disable_timing_event(id.context);
    if (!created.ok()) return created.status();
    if (!*created) return Status::FailedPrecondition("Qwen event is null");
    *event = *created;
  }
  return result;
}
RuntimeResources::~RuntimeResources() { Reset(); }
RuntimeResources::RuntimeResources(RuntimeResources&& other) noexcept
    : identity_(std::exchange(other.identity_, {})),
      driver_(std::exchange(other.driver_, nullptr)) {}
RuntimeResources& RuntimeResources::operator=(RuntimeResources&& other) noexcept {
  if (this != &other) {
    Reset();
    identity_ = std::exchange(other.identity_, {});
    driver_ = std::exchange(other.driver_, nullptr);
  }
  return *this;
}
void RuntimeResources::Reset() noexcept {
  if (!driver_) return;
  for (auto event : {identity_.diagnostic_event, identity_.scrub_event, identity_.event})
    if (event) driver_->destroy_event(event);
  for (auto stream : {identity_.diagnostic_stream, identity_.scrub_stream, identity_.stream})
    if (stream) driver_->destroy_stream(stream);
  if (identity_.context)
    driver_->release_primary_context(identity_.device_ordinal, identity_.context);
  identity_ = {};
  driver_ = nullptr;
}
}  // namespace pih::qwen_plugin
