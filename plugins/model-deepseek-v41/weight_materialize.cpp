#include "weight_materialize.h"
#include "weight_output_layout.h"
#include "weight_manifest.h"
#include "engram_hash.h"
#include "pih/core/canonical_json.h"
#include <algorithm>
#include <cerrno>
#include <dirent.h>
#include <fcntl.h>
#include <memory>
#include <new>
#include <sys/stat.h>
#include <unistd.h>

namespace pih::deepseek_v41 {
namespace {
struct Member final {
  std::string name;
  int fd = -1;
  std::uint64_t expected = 0, written = 0;
  Sha256 hash;
  ~Member() { if (fd >= 0) ::close(fd); }
};
Status EmptyPrivateDirectory(int fd) {
  struct stat st{};
  if (fd < 0 || ::fstat(fd, &st) || !S_ISDIR(st.st_mode) ||
      st.st_uid != ::geteuid() || (st.st_mode & 077) || !(st.st_mode & S_IWUSR))
    return Status::FailedPrecondition("V4.1 staging must be an owned private writable directory");
  const int scan = ::openat(fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (scan < 0) return Status::Unavailable("Cannot scan V4.1 staging directory");
  DIR* directory = ::fdopendir(scan);
  if (!directory) { ::close(scan); return Status::Unavailable("Cannot enumerate V4.1 staging directory"); }
  bool empty = true;
  errno = 0;
  while (const auto* entry = ::readdir(directory)) {
    const std::string_view name(entry->d_name);
    if (name != "." && name != "..") { empty = false; break; }
  }
  const int error = errno;
  ::closedir(directory);
  if (!empty || error) return Status::FailedPrecondition("V4.1 staging must be empty and readable");
  return Status::Ok();
}
Status AppendAt(Member& file, std::uint64_t offset, std::span<const std::byte> bytes) {
  if (offset != file.written || offset > file.expected || bytes.size() > file.expected - offset)
    return Status::Internal("V4.1 materializer received noncontiguous or excess output");
  const auto original = bytes;
  while (!bytes.empty()) {
    const auto count = ::pwrite(file.fd, bytes.data(), bytes.size(), static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return Status::Unavailable("V4.1 staging write failed; retain partial directory");
    offset += static_cast<std::size_t>(count);
    bytes = bytes.subspan(static_cast<std::size_t>(count));
  }
  auto status = file.hash.update(original);
  if (!status.ok()) return status;
  file.written = offset;
  return Status::Ok();
}
Result<Sha256Digest> Finish(int directory, Member& file, std::span<std::byte> workspace) {
  if (file.written != file.expected || ::fchmod(file.fd, 0400) || ::fsync(file.fd))
    return Status::Unavailable("V4.1 staging member incomplete or cannot be synced read-only");
  auto expected = file.hash.finalize();
  if (!expected.ok()) return expected.status();
  Sha256 observed;
  for (std::uint64_t offset = 0; offset < file.expected;) {
    auto bytes = workspace.first(static_cast<std::size_t>(std::min<std::uint64_t>(workspace.size(), file.expected - offset)));
    std::size_t done = 0;
    while (done < bytes.size()) {
      const auto count = ::pread(file.fd, bytes.data() + done, bytes.size() - done, static_cast<off_t>(offset + done));
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) return Status::Unavailable("V4.1 staging readback failed");
      done += static_cast<std::size_t>(count);
    }
    auto status = observed.update(bytes); if (!status.ok()) return status;
    offset += bytes.size();
  }
  auto digest = observed.finalize();
  if (!digest.ok()) return digest.status();
  struct stat held{}, named{};
  if (*digest != *expected || ::fstat(file.fd, &held) ||
      ::fstatat(directory, file.name.c_str(), &named, AT_SYMLINK_NOFOLLOW) ||
      !S_ISREG(held.st_mode) || held.st_nlink != 1 || (held.st_mode & 0222) ||
      held.st_size < 0 || static_cast<std::uint64_t>(held.st_size) != file.expected ||
      held.st_dev != named.st_dev || held.st_ino != named.st_ino)
    return Status::FailedPrecondition("V4.1 staging member changed or readback differs");
  return *digest;
}
Status CreateMember(int directory, Member& file) {
  file.fd = ::openat(directory, file.name.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (file.fd < 0) return Status::FailedPrecondition("Cannot exclusively create V4.1 staging member");
  return Status::Ok();
}
}
static Result<MaterializedWeightRank> MaterializeRank(
    int directory, const FlashConfig& config, std::uint32_t world,
    std::uint32_t rank, std::uint64_t device_budget,
    const std::function<Status(const BackboneWeightOutputLayout&, const WeightShardWriter&,
        std::span<std::byte>)>& emit, const std::function<Status()>& revalidate) {
  auto status = revalidate(); if (!status.ok()) return status;
  status = EmptyPrivateDirectory(directory); if (!status.ok()) return status;
  auto layout = BackboneWeightOutputLayout::Create(config, world, rank, device_budget);
  if (!layout.ok()) return layout.status();
  try {
    std::vector<std::byte> workspace(1024 * 1024);
    std::vector<std::unique_ptr<Member>> files;
    files.reserve(layout->shards().size());
    for (const auto& shard : layout->shards()) {
      auto file = std::make_unique<Member>();
      file->name = shard.name; file->expected = shard.file_bytes;
      status = CreateMember(directory, *file); if (!status.ok()) return status;
      files.push_back(std::move(file));
    }
    status = emit(*layout,
        [&](std::string_view name, std::uint64_t offset, std::span<const std::byte> bytes) {
          for (auto& file : files) if (file->name == name) return AppendAt(*file, offset, bytes);
          return Status::Internal("V4.1 output names an unplanned member");
        }, workspace);
    if (!status.ok()) return status;
    status = revalidate(); if (!status.ok()) return status;
    MaterializedWeightRank receipt;
    JsonValue::Array shards;
    for (auto& file : files) {
      auto hash = Finish(directory, *file, workspace); if (!hash.ok()) return hash.status();
      shards.emplace_back(JsonValue::Object{
          {"name", JsonValue(file->name)}, {"bytes", JsonValue(static_cast<std::int64_t>(file->expected))},
          {"sha256", JsonValue(hash->hex())}});
      receipt.file_bytes += file->expected;
      ++receipt.shard_count;
    }
    auto json = canonical_ascii_json(JsonValue(JsonValue::Object{
        {"schema", JsonValue(std::string("pih.deepseek-v41.weights.v1"))},
        {"config_sha256", JsonValue(config.config_sha256().hex())},
        {"world_size", JsonValue(static_cast<std::int64_t>(world))},
        {"rank", JsonValue(static_cast<std::int64_t>(rank))}, {"shards", JsonValue(std::move(shards))}}),
        kWeightManifestMaximumBytes);
    if (!json.ok()) return json.status();
    const auto bytes = std::as_bytes(std::span(*json));
    auto digest = sha256(bytes); if (!digest.ok()) return digest.status();
    auto parsed = ParseWeightManifest(*json, *digest, config, world, rank);
    if (!parsed.ok()) return parsed.status();
    Member manifest; manifest.name = "weights.manifest.json"; manifest.expected = bytes.size();
    status = CreateMember(directory, manifest); if (!status.ok()) return status;
    status = AppendAt(manifest, 0, bytes); if (!status.ok()) return status;
    auto hash = Finish(directory, manifest, workspace); if (!hash.ok()) return hash.status();
    if (*hash != *digest) return Status::Internal("V4.1 manifest serialization hash differs");
    status = revalidate(); if (!status.ok()) return status;
    if (::fsync(directory)) return Status::Unavailable("Cannot sync V4.1 staging directory");
    receipt.manifest_sha256 = *hash;
    receipt.file_bytes += manifest.expected;
    return receipt;
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 materialization allocation failed; retain partial directory");
  } catch (...) {
    return Status::Internal("V4.1 materialization failed; retain partial directory");
  }
}
Result<MaterializedWeightRank> MaterializeCanonicalWeightRank(
    int directory, const BackboneWeightFiles& source, const FlashConfig& config,
    std::uint32_t world, std::uint32_t rank, std::uint64_t budget) {
  if (source.catalog().world_size() != 1 || source.catalog().rank() != 0 ||
      source.catalog().config_sha256() != config.config_sha256())
    return Status::InvalidArgument("V4.1 materialization requires matching canonical TP1 source");
  try {
    return MaterializeRank(directory, config, world, rank, budget,
        [&](const BackboneWeightOutputLayout&, const WeightShardWriter& write, std::span<std::byte> workspace) {
          std::vector<RuntimeWeight> metadata;
          metadata.reserve(source.catalog().weights().size());
          for (const auto& item : source.catalog().weights()) metadata.push_back(item.weight.tensor);
          auto written = WriteCanonicalBackboneRank(config, world, rank, budget, metadata,
              [&](std::string_view name, std::uint64_t offset, std::span<std::byte> bytes) {
                return source.Read(name, offset, bytes);
              }, write, workspace);
          return written.ok() ? Status::Ok() : written.status();
        }, [&] { return source.Revalidate(); });
  } catch (...) {
    return Status::Internal("V4.1 canonical materialization failed; retain partial directory");
  }
}
Result<MaterializedWeightRank> MaterializeSourceBackbone(int directory,
    const WeightSourceCheckpoint& source, std::span<const std::string> excluded,
    std::uint64_t budget) {
  auto status = source.ValidateBackboneConversion(excluded); if (!status.ok()) return status;
  try {
    // Freeze the caller's disposition before beginning writes, and serialize in
    // stable order rather than relying on caller ordering in provenance.
    std::vector<std::string> exclusions(excluded.begin(), excluded.end());
    std::sort(exclusions.begin(), exclusions.end());
    std::size_t exclusion_bytes = 0;
    for (const auto& name : exclusions) {
      // Normalized inventory names are ASCII letters/digits/underscore/dot,
      // so no JSON escape expansion is possible. Reserve space for fixed fields.
      if (name.size() + 3 > 16 * 1024 * 1024 - 4096 - exclusion_bytes)
        return Status::InvalidArgument("V4.1 exclusion provenance exceeds 16 MiB metadata budget");
      exclusion_bytes += name.size() + 3;
    }
    auto receipt = MaterializeRank(directory, source.config(), 1, 0, budget,
        [&](const BackboneWeightOutputLayout& layout, const WeightShardWriter& write, std::span<std::byte> workspace) {
          for (const auto& shard : layout.shards()) {
            for (std::size_t offset = 0; offset < shard.prefix.size();) {
              const auto count = std::min(workspace.size(), shard.prefix.size() - offset);
              auto written = write(shard.name, offset, std::span<const std::byte>(shard.prefix).subspan(offset, count));
              if (!written.ok()) return written;
              offset += count;
            }
          }
          return source.ConvertBackbone(exclusions,
              [&](const RuntimeWeight& tensor, std::uint64_t offset, std::span<const std::byte> bytes) {
                const auto* location = layout.catalog().Find(tensor.name);
                if (!location || offset > location->weight.tensor.bytes ||
                    bytes.size() > location->weight.tensor.bytes - offset)
                  return Status::Internal("V4.1 conversion output differs from planned runtime tensor");
                return write(layout.shards()[location->shard].name, location->file_offset + offset, bytes);
              });
        }, [&] { return source.Revalidate(); });
    if (!receipt.ok()) return receipt.status();
    JsonValue::Array omitted;
    for (const auto& name : exclusions) omitted.emplace_back(name);
    auto json = canonical_ascii_json(JsonValue(JsonValue::Object{
        {"schema", JsonValue(std::string("pih.deepseek-v41.conversion-provenance.v1"))},
        {"conversion", JsonValue(std::string("canonical-text-backbone-v1"))},
        {"source_expectations_sha256", JsonValue(source.expectation_digest().hex())},
        {"config_sha256", JsonValue(source.config().config_sha256().hex())},
        {"weights_manifest_sha256", JsonValue(receipt->manifest_sha256.hex())},
        {"excluded_source_names", JsonValue(std::move(omitted))}}), 16 * 1024 * 1024);
    if (!json.ok()) return json.status();
    Member provenance; provenance.name = "conversion.provenance.json"; provenance.expected = json->size();
    status = CreateMember(directory, provenance); if (!status.ok()) return status;
    std::vector<std::byte> workspace(1024 * 1024);
    const auto bytes = std::as_bytes(std::span(json->data(), json->size()));
    for (std::size_t offset = 0; offset < bytes.size();) {
      const auto count = std::min(workspace.size(), bytes.size() - offset);
      status = AppendAt(provenance, offset, bytes.subspan(offset, count)); if (!status.ok()) return status;
      offset += count;
    }
    auto hash = Finish(directory, provenance, workspace); if (!hash.ok()) return hash.status();
    status = source.Revalidate(); if (!status.ok()) return status;
    if (::fsync(directory)) return Status::Unavailable("Cannot sync V4.1 conversion provenance directory");
    receipt->provenance_sha256 = *hash;
    receipt->file_bytes += provenance.expected;
    return std::move(*receipt);
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 source materialization allocation failed; retain partial directory");
  } catch (...) {
    return Status::Internal("V4.1 source materialization failed; retain partial directory");
  }
}
Result<std::uint64_t> MaterializeRuntimeMetadata(int directory, std::string_view config_json,
    const Sha256Digest& expected_config, std::span<const std::byte> map, const Sha256Digest& expected_map) {
  if (config_json.empty() || config_json.size() > FlashConfig::kMaximumBytes || expected_config == Sha256Digest{})
    return Status::InvalidArgument("V4.1 runtime metadata configuration bound invalid");
  try {
    const auto config_bytes = std::as_bytes(std::span(config_json.data(), config_json.size()));
    auto config_hash = sha256(config_bytes); if (!config_hash.ok()) return config_hash.status();
    if (*config_hash != expected_config) return Status::FailedPrecondition("V4.1 runtime configuration differs from rank binding");
    auto config = FlashConfig::Parse(config_json); if (!config.ok()) return config.status();
    auto admitted_map = EngramHashState::Create(*config, map, expected_map);
    if (!admitted_map.ok()) return admitted_map.status();
    struct stat root{};
    if (directory < 0 || ::fstat(directory, &root) || !S_ISDIR(root.st_mode) ||
        root.st_uid != ::geteuid() || (root.st_mode & 077) || !(root.st_mode & S_IWUSR))
      return Status::FailedPrecondition("V4.1 runtime metadata requires private writable staging");
    for (const auto* name : {"hf_config.json", "compressed-token-map.bin"}) {
      struct stat member{};
      if (::fstatat(directory, name, &member, AT_SYMLINK_NOFOLLOW) == 0 || errno != ENOENT)
        return Status::FailedPrecondition("V4.1 runtime metadata already exists or cannot be inspected");
    }
    std::vector<std::byte> workspace(1024 * 1024);
    auto save = [&](const char* name, std::span<const std::byte> bytes, const Sha256Digest& expected) -> Status {
      Member member; member.name = name; member.expected = bytes.size();
      auto status = CreateMember(directory, member); if (!status.ok()) return status;
      status = AppendAt(member, 0, bytes); if (!status.ok()) return status;
      auto hash = Finish(directory, member, workspace); if (!hash.ok()) return hash.status();
      return *hash == expected ? Status::Ok() : Status::FailedPrecondition("V4.1 runtime metadata readback differs");
    };
    auto status = save("hf_config.json", config_bytes, expected_config); if (!status.ok()) return status;
    status = save("compressed-token-map.bin", map, expected_map); if (!status.ok()) return status;
    if (::fsync(directory)) return Status::Unavailable("Cannot sync V4.1 runtime metadata directory");
    return static_cast<std::uint64_t>(config_bytes.size() + map.size());
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 runtime metadata allocation failed; retain partial staging");
  } catch (...) {
    return Status::Internal("V4.1 runtime metadata assembly failed; retain partial staging");
  }
}
}  // namespace pih::deepseek_v41
