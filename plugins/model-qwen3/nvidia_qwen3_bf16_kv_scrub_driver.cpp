#include "pih/model/nvidia_qwen3_bf16_kv_scrub_driver.h"

#include <chrono>
#include <limits>
#include <thread>

#include "async_binding.h"

namespace pih {
Result<NvidiaQwenBf16KvScrubDriver>
NvidiaQwenBf16KvScrubDriver::Create(
    const pih_nvidia_cuda_async_api_v1& async, std::uintptr_t context_identity,
    std::uint64_t timeout_ns) {
  if (!qwen_plugin::ValidAsyncApi(async) || !context_identity || !timeout_ns)
    return Status::InvalidArgument("Qwen KV scrub capability identity is invalid");
  return NvidiaQwenBf16KvScrubDriver(async, context_identity, timeout_ns);
}

Status NvidiaQwenBf16KvScrubDriver::clear_and_wait(
    std::uintptr_t destination, std::uint64_t bytes,
    DriverStreamHandle stream, DriverEventHandle event) {
  if (!destination || !bytes || !stream || !event ||
      bytes > std::numeric_limits<std::size_t>::max() ||
      bytes - 1 > std::numeric_limits<std::uintptr_t>::max() - destination)
    return Status::InvalidArgument("Qwen KV scrub submission is invalid");

  qwen_plugin::CapabilityEventDriver events(*async_, context_identity_);
  auto status = events.require_clean_last_error();
  if (!status.ok()) return status;
  status = qwen_plugin::MemoryStatus(async_->memset_async(
      async_->context, context_identity_, destination, 0, bytes, stream));
  if (!status.ok()) return status;
  status = events.record(event, stream);
  if (!status.ok()) return status;

  const auto started = std::chrono::steady_clock::now();
  for (;;) {
    const auto result = events.query(event);
    if (!result.ok()) return result.status();
    if (*result == CudaEventQueryResult::kSuccess)
      return events.require_clean_last_error();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    if (elapsed < 0 || static_cast<std::uint64_t>(elapsed) >= timeout_ns_)
      return Status::Unavailable("Qwen KV scrub completion timed out");
    std::this_thread::yield();
  }
}
}  // namespace pih
