#include "pih/model/deepseek_rank_worker_arguments.h"

#include <array>
#include <charconv>
#include <limits>

#include "pih/model/deepseek_rank_artifact_metadata_blob.h"

namespace pih {
namespace {

constexpr std::array<std::string_view, 13> kNames{{
    "rank", "world-size", "engine-epoch", "worker-generation",
    "device-identity", "startup-device-ordinal", "process-manifest",
    "startup-deadline-ns", "controller-pid", "controller-pidfd",
    "control-fd", "dspark-enabled", "metadata-reassembly-bytes"}};

Result<std::uint64_t> number(std::string_view value) {
  if (value.empty() || (value.size() > 1 && value.front() == '0'))
    return Status::InvalidArgument("DeepSeek worker identity number is not canonical");
  std::uint64_t result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
    return Status::InvalidArgument("DeepSeek worker identity number is invalid");
  return result;
}

}  // namespace

Result<DeepSeekRankWorkerArguments> DeepSeekRankWorkerArguments::Parse(
    std::span<const std::string_view> arguments) {
  DeepSeekRankWorkerArguments result;
  std::array<std::uint64_t, kNames.size()> values{};
  std::array<bool, kNames.size()> seen{};
  for (const auto argument : arguments) {
    constexpr std::string_view prefix = "--pih-";
    if (!argument.starts_with(prefix)) {
      result.application_arguments.emplace_back(argument); continue;
    }
    const auto equals = argument.find('=', prefix.size());
    if (equals == std::string_view::npos)
      return Status::InvalidArgument("DeepSeek worker reserved argument has no value");
    const auto name = argument.substr(prefix.size(), equals - prefix.size());
    std::size_t index = kNames.size();
    for (std::size_t i = 0; i < kNames.size(); ++i)
      if (kNames[i] == name) { index = i; break; }
    if (index == kNames.size() || seen[index])
      return Status::InvalidArgument("DeepSeek worker reserved argument is unknown or repeated");
    auto parsed = number(argument.substr(equals + 1));
    if (!parsed.ok()) return parsed.status();
    seen[index] = true; values[index] = *parsed;
  }
  for (const bool present : seen)
    if (!present) return Status::InvalidArgument("DeepSeek worker identity argument is missing");
  if (values[0] > std::numeric_limits<std::uint32_t>::max() ||
      values[1] > std::numeric_limits<std::uint32_t>::max() ||
      values[5] > std::numeric_limits<std::int32_t>::max() ||
      values[9] > std::numeric_limits<std::int32_t>::max() ||
      values[10] > std::numeric_limits<std::int32_t>::max())
    return Status::InvalidArgument("DeepSeek worker identity argument overflows its field");
  result.manifest = {values[2], values[3], static_cast<std::uint32_t>(values[1]),
                     static_cast<std::uint32_t>(values[0]), values[4], values[6],
                     {}, static_cast<std::int32_t>(values[5]), values[7]};
  result.controller_process_identity = values[8];
  result.controller_pidfd = static_cast<std::int32_t>(values[9]);
  result.control_fd = static_cast<std::int32_t>(values[10]);
  result.expected_dspark_enabled = values[11] == 1;
  result.maximum_metadata_reassembly_bytes = values[12];
  if (result.manifest.engine_epoch == 0 || result.manifest.worker_generation == 0 ||
      result.manifest.world_size == 0 || result.manifest.world_size > 4 ||
      result.manifest.rank >= result.manifest.world_size ||
      result.manifest.physical_device_identity == 0 ||
      result.manifest.process_manifest_identity == 0 ||
      result.manifest.physical_device_uuid_commitment != Sha256Digest{} ||
      result.manifest.startup_device_ordinal < 0 ||
      result.manifest.startup_deadline_ns == 0 ||
      result.controller_process_identity == 0 || result.controller_pidfd < 0 ||
      result.control_fd < 0 || values[11] > 1 ||
      result.maximum_metadata_reassembly_bytes == 0 ||
      result.maximum_metadata_reassembly_bytes >
          kDeepSeekRankArtifactMetadataBlobMaximumBytes)
    return Status::InvalidArgument("DeepSeek worker argument identity is invalid");
  return result;
}

}  // namespace pih
