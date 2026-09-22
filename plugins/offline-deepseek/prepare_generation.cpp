#include "prepare_generation.h"
#include "disposition_records.h"
#include "source_inventory.h"
#include <cerrno>
#include <dirent.h>
#include <fcntl.h>
#include <memory>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

namespace pih::offline_deepseek {
Status ValidatePreparationStaging(int fd) {
  struct stat directory{};
  if (::fstat(fd, &directory) != 0 || !S_ISDIR(directory.st_mode) ||
      directory.st_uid != ::geteuid() || (directory.st_mode & (S_IWGRP | S_IWOTH)) != 0)
    return Status::InvalidArgument("preparation requires an owned private staging directory");
  const auto scan_fd = ::openat(fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (scan_fd < 0) return Status::Unavailable("cannot inspect preparation staging");
  auto* stream = ::fdopendir(scan_fd);
  if (!stream) { ::close(scan_fd); return Status::Unavailable("cannot enumerate preparation staging"); }
  std::unique_ptr<DIR, decltype(&::closedir)> owned(stream, &::closedir);
  errno = 0;
  while (const auto* entry = ::readdir(stream)) {
    const std::string_view name(entry->d_name);
    if (name != "." && name != "..")
      return Status::FailedPrecondition("preparation staging must be empty");
    errno = 0;
  }
  if (errno != 0) return Status::Unavailable("preparation staging enumeration failed");
  return Status::Ok();
}
namespace {
Status ValidateAuthority(const SourcePreparationAuthority& authority) {
  if (authority.model_digest == Sha256Digest{} || authority.semantic_root == Sha256Digest{} ||
      authority.inventory_root == Sha256Digest{} || authority.expected_payload_root == Sha256Digest{} ||
      authority.converter_identity_root == Sha256Digest{})
    return Status::InvalidArgument("preparation authority roots must be nonzero");
  return Status::Ok();
}
bool Same(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode &&
      a.st_size == b.st_size && a.st_mtim.tv_sec == b.st_mtim.tv_sec &&
      a.st_mtim.tv_nsec == b.st_mtim.tv_nsec && a.st_ctim.tv_sec == b.st_ctim.tv_sec &&
      a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
}  // namespace
Result<CopiedGeneration> PreparePp1Generation(int staging_directory_fd,
    std::span<const int> admitted_source_descriptors,
    std::span<const SourcePayloadShardInput> shards,
    const SourcePreparationAuthority& authority) {
  if (admitted_source_descriptors.size() != 50 || shards.size() != 48)
    return Status::InvalidArgument("preparation descriptor cardinality invalid");
  auto status = ValidateAuthority(authority);
  if (!status.ok()) return status;
  status = ValidatePreparationStaging(staging_directory_fd);
  if (!status.ok()) return status;
  std::set<int> descriptors;
  std::set<std::pair<dev_t, ino_t>> identities;
  std::vector<struct stat> initial(admitted_source_descriptors.size());
  for (std::size_t i = 0; i < initial.size(); ++i) {
    const auto fd = admitted_source_descriptors[i];
    if (fd < 0 || !descriptors.insert(fd).second || ::fstat(fd, &initial[i]) != 0 ||
        !S_ISREG(initial[i].st_mode) || initial[i].st_size <= 0 ||
        !identities.emplace(initial[i].st_dev, initial[i].st_ino).second)
      return Status::InvalidArgument("preparation source descriptor set invalid or aliased");
  }
  std::set<int> shard_descriptors;
  for (const auto& shard : shards)
    if (!descriptors.contains(shard.descriptor) || !shard_descriptors.insert(shard.descriptor).second)
      return Status::InvalidArgument("preparation shard is absent from source set or duplicated");
  auto payload = ObserveSourcePayloadClosure(shards, authority.model_digest,
      authority.semantic_root, authority.inventory_root);
  if (!payload.ok()) return payload.status();
  if (payload->closure_root != authority.expected_payload_root)
    return Status::FailedPrecondition("observed payload closure differs from admitted source root");
  auto disposition = BuildPp1Disposition(payload->tensors, authority.inventory_root, payload->closure_root);
  if (!disposition.ok()) return disposition.status();
  status = ValidatePreparationStaging(staging_directory_fd);
  if (!status.ok()) return status;
  auto copied = CopyGenerationShards(staging_directory_fd, admitted_source_descriptors,
      disposition->records.selected, disposition->records.layout_authorities,
      disposition->records.selected_dispositions, disposition->layout_authority,
      authority.converter_identity_root);
  if (!copied.ok()) return copied.status();
  status = WriteGenerationMetadata(staging_directory_fd, *copied);
  if (!status.ok()) return status;
  for (std::size_t i = 0; i < initial.size(); ++i) {
    struct stat observed{};
    if (::fstat(admitted_source_descriptors[i], &observed) != 0 || !Same(initial[i], observed))
      return Status::FailedPrecondition("source changed across preparation; staging files retained");
  }
  return std::move(*copied);
}
Result<CopiedGeneration> PreparePp1Generation(int staging_directory_fd,
    const SourceArtifact& source,
    const SourcePreparationAuthority& authority) {
  if (authority.model_digest != source.model_digest())
    return Status::InvalidArgument("preparation model authority differs from admitted source");
  auto status = ValidateAuthority(authority);
  if (!status.ok()) return status;
  status = ValidatePreparationStaging(staging_directory_fd);
  if (!status.ok()) return status;
  auto inventory = CompileSourceInventory(source);
  if (!inventory.ok()) return inventory.status();
  if (inventory->plan.semantic_root != authority.semantic_root || inventory->inventory_root != authority.inventory_root)
    return Status::FailedPrecondition("preparation semantic/inventory root differs from actual source");
  status = source.Revalidate();
  if (!status.ok()) return status;
  auto copied = PreparePp1Generation(staging_directory_fd, source.descriptors(), inventory->plan.shards, authority);
  if (!copied.ok()) return copied.status();
  status = source.Revalidate();
  if (!status.ok()) return status;
  return std::move(*copied);
}
}  // namespace pih::offline_deepseek
