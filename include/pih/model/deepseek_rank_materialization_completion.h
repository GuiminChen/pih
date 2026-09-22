#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

#include "pih/model/runtime_profile_payload.h"
#include "pih/model/deepseek_rank_materialization_grant.h"

namespace pih {

class DeepSeekRankEngineResources;

inline constexpr std::string_view kDeepSeekRankMaterializationCompletionAbi =
    "pih_deepseek_rank_materialization_completion_v1";
inline constexpr std::string_view
    kDeepSeekRankMaterializationCompletionFrameAbi =
        "pih_deepseek_rank_materialization_completion_frame_v1";
inline constexpr std::size_t
    kDeepSeekRankMaterializationCompletionFrameBytes = 656;

// Independent runtime observations joined to the sealed in-process resource
// graph. Allocation roots are supplied by the instrumented CUDA/pinned
// allocators and must commit to the exact byte/generation facts below.
struct DeepSeekRankMaterializationCompletionObservation final {
  std::uint64_t completion_monotonic_ns = 0;
  std::uint64_t resident_selected_page_bytes = 0;
  std::uint64_t major_fault_count = 0;
  std::int32_t cuda_device_ordinal = -1;
  std::uint64_t cuda_resident_weight_allocation_bytes = 0;
  std::uint64_t pinned_staging_allocation_bytes = 0;
  Sha256Digest cuda_allocation_root{};
  Sha256Digest pinned_allocation_root{};
};

struct DeepSeekRankMaterializationCompletionFields final {
  std::uint32_t protocol_version = 0;
  std::uint64_t engine_epoch = 0;
  std::uint64_t worker_generation = 0;
  std::uint32_t world_size = 0;
  std::uint32_t rank = 0;
  std::uint64_t process_manifest_identity = 0;
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  std::uint64_t control_identity = 0;
  std::uint64_t challenge_identity = 0;
  std::int32_t device_ordinal = -1;
  RuntimeProfileGpuFamily gpu_family =
      RuntimeProfileGpuFamily::kRtx4090D24GiB;
  RuntimeProfileResidency residency = RuntimeProfileResidency::kFullResident;
  bool production_eligible = false;
  bool dspark_enabled = false;
  std::uint64_t prefault_completed_monotonic_ns = 0;
  std::uint64_t completion_monotonic_ns = 0;
  std::uint64_t deadline_ns = 0;
  std::uint64_t mapped_interval_bytes = 0;
  std::uint64_t selected_page_union_bytes = 0;
  std::uint64_t resident_selected_page_bytes = 0;
  std::uint64_t prefault_major_fault_count = 0;
  std::uint64_t completion_major_fault_count = 0;
  std::uint64_t fixed_weight_backing_bytes = 0;
  std::uint64_t fixed_weight_payload_bytes = 0;
  std::uint64_t fixed_weight_allocation_generation = 0;
  std::uint64_t pinned_staging_bytes = 0;
  std::uint64_t pinned_staging_allocation_generation = 0;
  std::uint32_t expert_slot_count = 0;
  std::uint32_t staging_extent_count = 0;
  std::uint32_t pager_transfer_reservations = 0;
  Sha256Digest profile_envelope_root{};
  Sha256Digest device_observation_root{};
  Sha256Digest capacity_plan_instance_root{};
  Sha256Digest post_mapping_seal_root{};
  Sha256Digest metadata_transaction_root{};
  Sha256Digest mapping_owner_root{};
  Sha256Digest grant_root{};
  Sha256Digest prefault_layout_root{};
  Sha256Digest prefault_receipt_root{};
  Sha256Digest weight_layout_root{};
  Sha256Digest weight_seal_root{};
  Sha256Digest cuda_allocation_root{};
  Sha256Digest pinned_allocation_root{};
};

Result<Sha256Digest> compile_deepseek_rank_materialization_completion_root(
    const DeepSeekRankMaterializationCompletionFields& fields);
Result<DeepSeekRankMaterializationCompletionFields>
compile_deepseek_rank_materialization_completion(
    const DeepSeekRankEngineResources& resources,
    const DeepSeekRankMaterializationCompletionObservation& observation);
Status validate_deepseek_rank_materialization_completion_worker_identity(
    const DeepSeekRankMaterializationAdmission& admission,
    const DeepSeekRankExecReady& ready);
Result<std::array<
    std::byte, kDeepSeekRankMaterializationCompletionFrameBytes>>
encode_deepseek_rank_materialization_completion(
    const DeepSeekRankMaterializationCompletionFields& fields);
Result<DeepSeekRankMaterializationCompletionFields>
decode_deepseek_rank_materialization_completion(
    std::span<const std::byte> frame);

class DeepSeekRankMaterializationCompletionSenderOperations {
 public:
  virtual ~DeepSeekRankMaterializationCompletionSenderOperations() = default;
  // kUnavailable means no bytes were accepted and the identical frame may be
  // retried.
  virtual Status send_completion(
      std::int32_t control_fd, std::span<const std::byte> frame) = 0;
  virtual Result<std::uint64_t> monotonic_now_ns() = 0;
};

class DeepSeekRankMaterializationCompletionSender final {
 public:
  static Result<DeepSeekRankMaterializationCompletionSender> Create(
      DeepSeekRankMaterializationCompletionFields completion,
      std::int32_t control_fd,
      DeepSeekRankMaterializationCompletionSenderOperations& operations);

  Status advance();
  [[nodiscard]] bool complete() const noexcept { return complete_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::uint64_t deadline_ns() const noexcept {
    return completion_.deadline_ns;
  }

 private:
  DeepSeekRankMaterializationCompletionSender(
      DeepSeekRankMaterializationCompletionFields completion,
      std::array<std::byte,
                 kDeepSeekRankMaterializationCompletionFrameBytes> frame,
      std::int32_t control_fd,
      DeepSeekRankMaterializationCompletionSenderOperations& operations)
      noexcept
      : completion_(std::move(completion)), frame_(std::move(frame)),
        control_fd_(control_fd), operations_(&operations) {}

  DeepSeekRankMaterializationCompletionFields completion_;
  std::array<std::byte,
             kDeepSeekRankMaterializationCompletionFrameBytes> frame_{};
  std::int32_t control_fd_ = -1;
  DeepSeekRankMaterializationCompletionSenderOperations* operations_ =
      nullptr;
  bool complete_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
