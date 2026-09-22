#include "pih/model/deepseek_dspark_gpu_state_digest.h"

namespace pih { namespace {

Result<Sha256Digest> component_leaf(
    const DeepSeekDsparkDeviceStateComponent& component,
    const Sha256Digest& digest) {
  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek:dspark:gpu-state-component:v1", 4);
  if (!hash.ok()) return hash.status();
  auto status = hash->add_u32(1, static_cast<std::uint32_t>(component.kind));
  if (status.ok()) status = hash->add_u32(2, component.logical_page);
  if (status.ok()) status = hash->add_u64(3, component.bytes);
  if (status.ok()) status = hash->add_hash(4, digest);
  return status.ok() ? hash->finalize() : Result<Sha256Digest>(status);
}

Result<Sha256Digest> node_hash(const Sha256Digest& left,
                               const Sha256Digest& right) {
  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek:dspark:gpu-state-node:v1", 2);
  if (!hash.ok()) return hash.status();
  auto status = hash->add_hash(1, left);
  if (status.ok()) status = hash->add_hash(2, right);
  return status.ok() ? hash->finalize() : Result<Sha256Digest>(status);
}

Result<Sha256Digest> promote_hash(const Sha256Digest& child) {
  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek:dspark:gpu-state-promote:v1", 1);
  if (!hash.ok()) return hash.status();
  const auto status = hash->add_hash(1, child);
  return status.ok() ? hash->finalize() : Result<Sha256Digest>(status);
}

Result<Sha256Digest> component_root(
    std::span<const DeepSeekDsparkDeviceStateComponent> components,
    std::span<const Sha256Digest> digests) {
  std::vector<Sha256Digest> level;
  level.reserve(components.size());
  for (std::size_t i = 0; i < components.size(); ++i) {
    auto leaf = component_leaf(components[i], digests[i]);
    if (!leaf.ok()) return leaf.status();
    level.push_back(*leaf);
  }
  while (level.size() > 1) {
    std::vector<Sha256Digest> next;
    next.reserve((level.size() + 1) / 2);
    for (std::size_t i = 0; i < level.size(); i += 2) {
      auto value = i + 1 < level.size()
                       ? node_hash(level[i], level[i + 1])
                       : promote_hash(level[i]);
      if (!value.ok()) return value.status();
      next.push_back(*value);
    }
    level = std::move(next);
  }
  return level.front();
}

bool valid_order(std::span<const DeepSeekDsparkDeviceStateComponent> values) {
  std::uint32_t previous_kind = 0;
  std::uint32_t previous_page = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto kind = static_cast<std::uint32_t>(values[i].kind);
    if (kind < 1 || kind > 4 || (i != 0 &&
        (kind < previous_kind ||
         (kind == previous_kind && values[i].logical_page <= previous_page)))) {
      return false;
    }
    previous_kind = kind;
    previous_page = values[i].logical_page;
  }
  return true;
}

}  // namespace

Result<DeepSeekDsparkGpuStateDigestCoordinator>
DeepSeekDsparkGpuStateDigestCoordinator::Create(
    std::uint32_t maximum_components,
    DeepSeekDsparkGpuStateDigestOperations& operations) {
  if (maximum_components == 0 || maximum_components > 65536) {
    return Status::InvalidArgument("DeepSeek GPU state digest capacity is invalid");
  }
  DeepSeekDsparkGpuStateDigestCoordinator value;
  value.maximum_components_ = maximum_components;
  value.operations_ = &operations;
  return value;
}

Status DeepSeekDsparkGpuStateDigestCoordinator::poison(Status status) {
  poisoned_ = true;
  inflight_ = false;
  return status.ok() ? Status::Internal("DeepSeek GPU state digest poisoned")
                     : status;
}

Status DeepSeekDsparkGpuStateDigestCoordinator::launch(
    const DeepSeekDsparkGpuStateDigestSubmission& value) {
  if (inflight_ || poisoned_ || value.rank >= 4 ||
      value.plan_sequence == 0 || value.old_generation == 0 ||
      value.new_generation != value.old_generation + 1 ||
      value.components.empty() ||
      value.components.size() > maximum_components_ ||
      value.host_component_digests.size() != value.components.size() ||
      !valid_order(value.components) || value.device_digest_workspace == 0 ||
      value.device_digest_workspace_bytes <
          value.components.size() * sizeof(Sha256Digest) ||
      value.device_error_flag_u32 == 0 || value.host_error_flag == nullptr ||
      value.stream == 0 || value.completion_event == 0) {
    return Status::FailedPrecondition(
        "DeepSeek GPU state digest submission is invalid");
  }
  for (const auto& component : value.components) {
    if (component.device_address == 0 || component.bytes == 0) {
      return Status::FailedPrecondition(
          "DeepSeek GPU state digest component is invalid");
    }
  }
  *value.host_error_flag = 0;
  const auto status = operations_->launch_component_sha256(value);
  if (!status.ok()) return poison(status);
  active_ = value;
  inflight_ = true;
  return Status::Ok();
}

Result<DeepSeekDsparkGpuStateDigestPoll>
DeepSeekDsparkGpuStateDigestCoordinator::poll() {
  if (!inflight_ || poisoned_) {
    return Status::FailedPrecondition("DeepSeek GPU state digest is not pollable");
  }
  auto event = operations_->query_event(active_.completion_event);
  if (!event.ok()) return poison(event.status());
  if (*event == DeepSeekExpertAsyncStatus::kInProgress) {
    return DeepSeekDsparkGpuStateDigestPoll{
        *event, {}, active_.rank, active_.plan_sequence,
        active_.old_generation, active_.new_generation,
        active_.retained_record_count};
  }
  if (*event == DeepSeekExpertAsyncStatus::kError ||
      *active_.host_error_flag != 0) {
    (void)poison(Status::Internal("DeepSeek GPU state digest failed"));
    return DeepSeekDsparkGpuStateDigestPoll{
        DeepSeekExpertAsyncStatus::kError, {}, active_.rank,
        active_.plan_sequence, active_.old_generation,
        active_.new_generation, active_.retained_record_count};
  }
  if (*event != DeepSeekExpertAsyncStatus::kSuccess) {
    return poison(Status::Internal(
        "DeepSeek GPU state digest received invalid event state"));
  }
  auto root = component_root(active_.components,
                             active_.host_component_digests);
  if (!root.ok()) return poison(root.status());
  auto hash = CanonicalHashBuilder::Create(
      "pih:deepseek:dspark:gpu-state-root:v1", 7);
  if (!hash.ok()) return poison(hash.status());
  auto status = hash->add_u32(1, active_.rank);
  if (status.ok()) status = hash->add_u64(2, active_.plan_sequence);
  if (status.ok()) status = hash->add_u64(3, active_.old_generation);
  if (status.ok()) status = hash->add_u64(4, active_.new_generation);
  if (status.ok()) status = hash->add_u32(5, active_.retained_record_count);
  if (status.ok()) status = hash->add_u32(
      6, static_cast<std::uint32_t>(active_.components.size()));
  if (status.ok()) status = hash->add_hash(7, *root);
  if (!status.ok()) return poison(status);
  auto final = hash->finalize();
  if (!final.ok()) return poison(final.status());
  inflight_ = false;
  return DeepSeekDsparkGpuStateDigestPoll{
      DeepSeekExpertAsyncStatus::kSuccess, *final, active_.rank,
      active_.plan_sequence, active_.old_generation,
      active_.new_generation, active_.retained_record_count};
}

}  // namespace pih
