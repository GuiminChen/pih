#pragma once

#include <cstdint>
#include <string_view>

#include "pih/model/deepseek_rank_post_mapping_resource_codec.h"

namespace pih {

inline constexpr std::string_view
    kDeepSeekRankPostMappingResourceReportAbi =
        "pih_deepseek_rank_post_mapping_resource_report_v1";

// Shared commitment to the exact rooted authority accepted by a worker and
// the exact rooted observation it sent. This is not an all-rank admission:
// only a later controller grant derived from the sealed coordinator may use
// it to authorize the next startup phase.
class DeepSeekRankPostMappingResourceReport final {
 public:
  static Result<DeepSeekRankPostMappingResourceReport> Compile(
      const DeepSeekRankPostMappingResourceAuthority& authority,
      const DeepSeekRankPostMappingResourceObservation& observation);

  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }
  [[nodiscard]] std::uint64_t worker_generation() const noexcept {
    return worker_generation_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::uint64_t process_identity() const noexcept {
    return process_identity_;
  }
  [[nodiscard]] std::uint64_t authority_deadline_ns() const noexcept {
    return authority_deadline_ns_;
  }
  [[nodiscard]] const Sha256Digest& mapping_owner_root() const noexcept {
    return mapping_owner_root_;
  }
  [[nodiscard]] const Sha256Digest& capacity_plan_instance_root()
      const noexcept {
    return capacity_plan_instance_root_;
  }
  [[nodiscard]] const Sha256Digest& first_resource_seal_root()
      const noexcept {
    return first_resource_seal_root_;
  }
  [[nodiscard]] const Sha256Digest& metadata_transaction_root()
      const noexcept {
    return metadata_transaction_root_;
  }
  [[nodiscard]] const Sha256Digest& authority_frame_root() const noexcept {
    return authority_frame_root_;
  }
  [[nodiscard]] const Sha256Digest& observation_frame_root() const noexcept {
    return observation_frame_root_;
  }
  [[nodiscard]] const Sha256Digest& report_root() const noexcept {
    return report_root_;
  }

 private:
  DeepSeekRankPostMappingResourceReport(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      std::uint32_t world_size, std::uint32_t rank,
      std::uint64_t process_identity, std::uint64_t authority_deadline_ns,
      Sha256Digest mapping_owner_root,
      Sha256Digest capacity_plan_instance_root,
      Sha256Digest first_resource_seal_root,
      Sha256Digest metadata_transaction_root,
      Sha256Digest authority_frame_root,
      Sha256Digest observation_frame_root,
      Sha256Digest report_root) noexcept;

  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
  std::uint32_t world_size_ = 0;
  std::uint32_t rank_ = 0;
  std::uint64_t process_identity_ = 0;
  std::uint64_t authority_deadline_ns_ = 0;
  Sha256Digest mapping_owner_root_{};
  Sha256Digest capacity_plan_instance_root_{};
  Sha256Digest first_resource_seal_root_{};
  Sha256Digest metadata_transaction_root_{};
  Sha256Digest authority_frame_root_{};
  Sha256Digest observation_frame_root_{};
  Sha256Digest report_root_{};
};

}  // namespace pih
