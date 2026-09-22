#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "pih/model/deepseek_rank_control_codec.h"
#include "pih/model/deepseek_rank_post_exec_resource_collector.h"

namespace pih {

inline constexpr std::string_view kDeepSeekRankPostExecResourceExchangeAbi =
    "pih_deepseek_rank_post_exec_resource_exchange_v1";

enum class DeepSeekRankPostExecResourceReporterWaitEvent : std::uint8_t {
  kAuthorityReadable,
  kObservationWritable,
};

Result<DeepSeekRankPostExecResourceAuthority>
compile_deepseek_rank_post_exec_resource_authority(
    const DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessManifest> manifests,
    const DeepSeekRankSpawnResourcePlan& spawn_plan,
    std::span<const DeepSeekRankPostExecResourcePlan> plans,
    std::uint32_t rank, std::uint64_t deadline_ns);

class DeepSeekRankPostExecResourceReporterOperations {
 public:
  virtual ~DeepSeekRankPostExecResourceReporterOperations() = default;
  virtual Result<std::optional<std::vector<std::byte>>> receive_authority(
      std::int32_t control_fd) = 0;
  virtual Result<std::uint64_t> monotonic_now_ns() = 0;
  virtual Status send_observation(
      std::int32_t control_fd, std::span<const std::byte> frame) = 0;
};

class DeepSeekRankPostExecResourceReporter final {
 public:
  static Result<DeepSeekRankPostExecResourceReporter> Create(
      DeepSeekRankProcessManifest manifest, DeepSeekRankExecReady exec_ready,
      std::int32_t control_fd,
      StableDeepSeekRankPostExecResourceCollector& collector,
      DeepSeekRankPostExecResourceReporterOperations& operations);

  Status advance();
  [[nodiscard]] bool reported() const noexcept { return reported_; }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::optional<
      DeepSeekRankPostExecResourceReporterWaitEvent>
  wait_event() const noexcept;
  [[nodiscard]] std::optional<std::uint64_t> deadline_ns()
      const noexcept;

 private:
  DeepSeekRankPostExecResourceReporter(
      DeepSeekRankProcessManifest manifest, DeepSeekRankExecReady exec_ready,
      std::int32_t control_fd,
      StableDeepSeekRankPostExecResourceCollector& collector,
      DeepSeekRankPostExecResourceReporterOperations& operations) noexcept;
  Status fail(Status cause) noexcept;

  DeepSeekRankProcessManifest manifest_;
  DeepSeekRankExecReady exec_ready_;
  std::int32_t control_fd_ = -1;
  StableDeepSeekRankPostExecResourceCollector* collector_ = nullptr;
  DeepSeekRankPostExecResourceReporterOperations* operations_ = nullptr;
  std::optional<DeepSeekRankPostExecResourceAuthority> authority_;
  std::optional<std::array<
      std::byte, kDeepSeekRankPostExecResourceObservationBytes>> frame_;
  bool reported_ = false;
  bool poisoned_ = false;
};

class DeepSeekRankPostExecResourceChannel {
 public:
  virtual ~DeepSeekRankPostExecResourceChannel() = default;
  virtual Status send_authority(
      const DeepSeekRankProcessHandle& handle,
      std::span<const std::byte> frame) = 0;
  virtual Result<std::optional<DeepSeekRankPostExecResourceObservation>>
  poll_observation(const DeepSeekRankProcessHandle& handle) = 0;
};

class DeepSeekRankPostExecResourceCoordinator final {
 public:
  static Result<DeepSeekRankPostExecResourceCoordinator> Create(
      DeepSeekRankProcessSupervisor& supervisor,
      std::span<const DeepSeekRankProcessManifest> manifests,
      DeepSeekRankSpawnResourcePlan spawn_plan,
      std::span<const DeepSeekRankPostExecResourcePlan> plans,
      std::uint64_t deadline_ns,
      DeepSeekRankPostExecResourceChannel& channel);

  Status advance(std::uint64_t now_ns);
  [[nodiscard]] bool sealed() const noexcept { return seal_.has_value(); }
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::size_t receipt_count() const noexcept;
  [[nodiscard]] const DeepSeekRankPostExecResourceSeal* seal()
      const noexcept {
    return seal_ ? &*seal_ : nullptr;
  }

 private:
  DeepSeekRankPostExecResourceCoordinator(
      DeepSeekRankProcessSupervisor& supervisor,
      std::vector<DeepSeekRankProcessManifest> manifests,
      DeepSeekRankSpawnResourcePlan spawn_plan,
      std::vector<DeepSeekRankPostExecResourcePlan> plans,
      std::vector<std::array<
          std::byte, kDeepSeekRankPostExecResourceAuthorityBytes>>
          authority_frames,
      std::uint64_t deadline_ns,
      DeepSeekRankPostExecResourceChannel& channel) noexcept;
  Status fail(Status cause) noexcept;

  DeepSeekRankProcessSupervisor* supervisor_ = nullptr;
  std::vector<DeepSeekRankProcessManifest> manifests_;
  DeepSeekRankSpawnResourcePlan spawn_plan_;
  std::vector<DeepSeekRankPostExecResourcePlan> plans_;
  std::vector<std::array<
      std::byte, kDeepSeekRankPostExecResourceAuthorityBytes>>
      authority_frames_;
  std::uint64_t deadline_ns_ = 0;
  DeepSeekRankPostExecResourceChannel* channel_ = nullptr;
  std::vector<std::optional<DeepSeekRankPostExecResourceReceipt>> receipts_;
  std::optional<DeepSeekRankPostExecResourceSeal> seal_;
  bool authorities_dispatched_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
