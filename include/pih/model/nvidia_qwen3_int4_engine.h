#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"
#include "pih/contracts/nvidia_cuda_memory_v1.h"
#include "pih/contracts/nvidia_cuda_resources_v1.h"
#include "pih/contracts/nvidia_cuda_async_v1.h"
#include "pih/model/nvidia_qwen_packed_runtime.h"
#include "pih/scheduler/controller_request_arena.h"

namespace pih {
class NvidiaQwen3Int4Engine final {
 public:
  static Result<std::unique_ptr<NvidiaQwen3Int4Engine>> LoadPinnedSnapshot(
      const pih_nvidia_cuda_memory_api_v1& memory,
      const pih_nvidia_cuda_resources_api_v1& resources,
      const pih_nvidia_cuda_async_api_v1& async,
      const std::filesystem::path& cubin_root,
      std::int32_t device_ordinal,
      std::span<const std::byte> artifact_bytes,
      std::string_view config_json,
      Sha256Digest expected_artifact_digest,
      std::uint32_t provider_prepared_sm);
  ~NvidiaQwen3Int4Engine();
  NvidiaQwen3Int4Engine(const NvidiaQwen3Int4Engine&)=delete;
  NvidiaQwen3Int4Engine& operator=(const NvidiaQwen3Int4Engine&)=delete;
  Result<std::uint64_t> submit_packed(
      std::span<const std::int64_t> prompt,
      std::uint32_t maximum_new_tokens,
      const ControllerRequestSampling& sampling);
  Status cancel_packed(std::uint64_t request_generation);
  Result<NvidiaQwenPackedStep> drive_packed();
  Result<std::optional<NvidiaQwenPackedOutputEvent>> try_take_packed_event();
  Status acknowledge_packed_output(std::uint64_t plan_sequence);
  Status drain_packed(std::uint64_t request_generation);
  Status close();
 private:
  static Result<std::unique_ptr<NvidiaQwen3Int4Engine>> LoadImpl(
      const pih_nvidia_cuda_memory_api_v1& memory,
    const pih_nvidia_cuda_resources_api_v1& resources,
    const pih_nvidia_cuda_async_api_v1& async,
      const std::filesystem::path& cubin_root,std::int32_t device_ordinal,
      Sha256Digest expected_artifact_digest,
      std::span<const std::byte> artifact_bytes,
      std::string_view config_json,
      std::uint32_t provider_prepared_sm);
  struct State;
  explicit NvidiaQwen3Int4Engine(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};
}
