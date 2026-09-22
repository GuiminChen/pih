#include "worker_artifacts.h"
#include <ctime>
#include <new>

namespace pih::deepseek_v41 {
namespace {
Status Deadline(std::uint64_t deadline) {
  timespec now{};
  if (::clock_gettime(CLOCK_MONOTONIC, &now) || now.tv_sec < 0)
    return Status::Unavailable("Cannot observe worker monotonic startup clock");
  if (static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL + static_cast<std::uint64_t>(now.tv_nsec) >= deadline)
    return Status::DeadlineExceeded("Worker artifact admission deadline expired");
  return Status::Ok();
}
std::string_view Text(const std::vector<std::byte>& bytes) {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}
}
Result<std::unique_ptr<WorkerArtifacts>> WorkerArtifacts::Open(const WorkerBootstrap& bootstrap) {
  auto valid = bootstrap.Validate(); if (!valid.ok()) return valid;
  valid = Deadline(bootstrap.startup_ns); if (!valid.ok()) return valid;
  try {
    const std::filesystem::path root(bootstrap.artifact_directory);
    auto config_bytes = ReadAuthenticatedMetadata(root / "hf_config.json", bootstrap.config_sha256, FlashConfig::kMaximumBytes);
    if (!config_bytes.ok()) return config_bytes.status();
    auto config = FlashConfig::Parse(Text(*config_bytes)); if (!config.ok()) return config.status();
    if (config->config_sha256() != bootstrap.config_sha256)
      return Status::FailedPrecondition("Worker configuration identity changed after parsing");
    valid = Deadline(bootstrap.startup_ns); if (!valid.ok()) return valid;
    auto map = ReadAuthenticatedMetadata(root / "compressed-token-map.bin", bootstrap.map_sha256,
        FlashConfig::kVocabularySize * 4ULL);
    if (!map.ok()) return map.status();
    auto hashes = EngramHashState::Create(*config, *map, bootstrap.map_sha256); if (!hashes.ok()) return hashes.status();
    auto lock = ReadAuthenticatedMetadata(bootstrap.plugin_lock, bootstrap.plugin_lock_sha256, 1U << 20);
    if (!lock.ok()) return lock.status();
    valid = Deadline(bootstrap.startup_ns); if (!valid.ok()) return valid;
    auto owner = std::unique_ptr<WorkerArtifacts>(new WorkerArtifacts(std::move(*config), std::move(*hashes)));
    owner->lock_.assign(Text(*lock));
    auto weights = BackboneWeightFiles::OpenManifest(root, bootstrap.weight_manifest_sha256, owner->config_,
        bootstrap.world, bootstrap.rank, bootstrap.device_budget);
    if (!weights.ok()) return weights.status();
    owner->weights_ = std::move(*weights);
    valid = Deadline(bootstrap.startup_ns); if (!valid.ok()) return valid;
    return owner;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Worker artifact admission allocation failed"); }
}
}  // namespace pih::deepseek_v41
