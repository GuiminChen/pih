#pragma once

#include <span>
#include <vector>

#include "pih/core/canonical_hash.h"
#include "pih/model/deepseek_expert_transfer_driver.h"

namespace pih {

enum class DeepSeekDsparkStateComponentKind : std::uint8_t {
  kFixed = 1,
  kRatio4Main = 2,
  kRatio4Index = 3,
  kRatio128 = 4,
};

struct DeepSeekDsparkDeviceStateComponent final {
  DeepSeekDsparkStateComponentKind kind{};
  std::uint32_t logical_page = 0;
  std::uintptr_t device_address = 0;
  std::uint64_t bytes = 0;
};

struct DeepSeekDsparkGpuStateDigestSubmission final {
  std::uint32_t rank = 0;
  std::uint64_t plan_sequence = 0;
  std::uint64_t old_generation = 0;
  std::uint64_t new_generation = 0;
  std::uint32_t retained_record_count = 0;
  std::span<const DeepSeekDsparkDeviceStateComponent> components;
  std::uintptr_t device_digest_workspace = 0;
  std::uint64_t device_digest_workspace_bytes = 0;
  std::span<Sha256Digest> host_component_digests;
  std::uintptr_t device_error_flag_u32 = 0;
  std::uint32_t* host_error_flag = nullptr;
  std::uintptr_t stream = 0;
  std::uintptr_t completion_event = 0;
};

class DeepSeekDsparkGpuStateDigestOperations {
 public:
  virtual ~DeepSeekDsparkGpuStateDigestOperations() = default;
  // Hashes each component on device and asynchronously copies only the
  // resulting 32-byte digests and error word to pinned host storage.
  virtual Status launch_component_sha256(
      const DeepSeekDsparkGpuStateDigestSubmission& submission) = 0;
  virtual Result<DeepSeekExpertAsyncStatus> query_event(
      std::uintptr_t completion_event) = 0;
};

class DeepSeekDsparkGpuStateDigestPoll final {
 public:
  [[nodiscard]] DeepSeekExpertAsyncStatus status() const noexcept {
    return status_;
  }
  [[nodiscard]] const Sha256Digest& local_state_hash() const noexcept {
    return local_state_hash_;
  }
  [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::uint64_t plan_sequence() const noexcept {
    return plan_sequence_;
  }
  [[nodiscard]] std::uint64_t old_generation() const noexcept {
    return old_generation_;
  }
  [[nodiscard]] std::uint64_t new_generation() const noexcept {
    return new_generation_;
  }
  [[nodiscard]] std::uint32_t retained_record_count() const noexcept {
    return retained_record_count_;
  }

 private:
  friend class DeepSeekDsparkGpuStateDigestCoordinator;
  DeepSeekDsparkGpuStateDigestPoll(
      DeepSeekExpertAsyncStatus status, Sha256Digest local_state_hash,
      std::uint32_t rank, std::uint64_t plan_sequence,
      std::uint64_t old_generation, std::uint64_t new_generation,
      std::uint32_t retained_record_count)
      : status_(status), local_state_hash_(local_state_hash), rank_(rank),
        plan_sequence_(plan_sequence), old_generation_(old_generation),
        new_generation_(new_generation),
        retained_record_count_(retained_record_count) {}

  DeepSeekExpertAsyncStatus status_ = DeepSeekExpertAsyncStatus::kInProgress;
  Sha256Digest local_state_hash_;
  std::uint32_t rank_ = 0;
  std::uint64_t plan_sequence_ = 0;
  std::uint64_t old_generation_ = 0;
  std::uint64_t new_generation_ = 0;
  std::uint32_t retained_record_count_ = 0;
};

class DeepSeekDsparkGpuStateDigestCoordinator final {
 public:
  static Result<DeepSeekDsparkGpuStateDigestCoordinator> Create(
      std::uint32_t maximum_components,
      DeepSeekDsparkGpuStateDigestOperations& operations);
  Status launch(const DeepSeekDsparkGpuStateDigestSubmission& submission);
  Result<DeepSeekDsparkGpuStateDigestPoll> poll();

 private:
  Status poison(Status status);
  std::uint32_t maximum_components_ = 0;
  DeepSeekDsparkGpuStateDigestOperations* operations_ = nullptr;
  DeepSeekDsparkGpuStateDigestSubmission active_;
  bool inflight_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
