#include "generation_promote.h"
#include <cerrno>
#include <cstdio>
#include <dirent.h>
#include <set>
#include <memory>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace pih::offline_deepseek {
namespace {
struct Fd final {
  int value = -1;
  explicit Fd(int fd) : value(fd) {}
  Fd(Fd&& other) noexcept : value(std::exchange(other.value, -1)) {}
  ~Fd() { if (value >= 0) ::close(value); }
};
Result<Fd> OpenDirectory(const std::filesystem::path& path) {
  if (!path.is_absolute()) return Status::InvalidArgument("promotion path must be absolute");
  Fd result(::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (result.value < 0) return Status::Unavailable("cannot open filesystem root");
  for (const auto& part : path.relative_path()) {
    const auto name = part.string();
    if (name.empty() || name == "." || name == ".." || name.find('\0') != name.npos)
      return Status::InvalidArgument("promotion path is not canonical");
    const auto next = ::openat(result.value, name.c_str(),
        O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (next < 0) return Status::FailedPrecondition("cannot open promotion directory without links");
    ::close(result.value);
    result.value = next;
  }
  return result;
}
bool Identity(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && S_ISDIR(b.st_mode);
}
bool Controlled(const struct stat& directory) {
  return S_ISDIR(directory.st_mode) && directory.st_uid == ::geteuid() &&
      (directory.st_mode & (S_IWGRP | S_IWOTH)) == 0;
}
std::set<std::string> ExpectedMembers() {
  std::set<std::string> expected{"pih.manifest.json", "pih.runtime-records.json",
      "model.safetensors.index.json", "model-endpoint.safetensors"};
  for (unsigned i = 0; i < 43; ++i)
    expected.insert("model-layers-" + std::to_string(i) + ".safetensors");
  return expected;
}
Status ExactMembers(int fd) {
  auto expected = ExpectedMembers();
  const auto scan_fd = ::openat(fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (scan_fd < 0) return Status::Unavailable("cannot inspect promotion member set");
  auto* directory = ::fdopendir(scan_fd);
  if (!directory) { ::close(scan_fd); return Status::Unavailable("cannot enumerate promotion directory"); }
  std::unique_ptr<DIR, decltype(&::closedir)> owned_directory(directory, &::closedir);
  bool valid = true;
  errno = 0;
  while (auto* entry = ::readdir(directory)) {
    const std::string_view name(entry->d_name);
    if (name != "." && name != ".." && expected.erase(std::string(name)) != 1) {
      valid = false;
      break;
    }
    errno = 0;
  }
  const auto scan_error = errno;
  owned_directory.reset();
  if (!valid || scan_error != 0 || !expected.empty())
    return Status::FailedPrecondition("promotion directory has extra or missing members");
  return Status::Ok();
}
Status SealReadOnly(int directory_fd) {
  struct Member final {
    std::string name;
    Fd descriptor;
    struct stat initial;
  };
  struct stat directory{};
  if (::fstat(directory_fd, &directory) != 0 || !Controlled(directory))
    return Status::FailedPrecondition("generation directory no longer controlled before sealing");
  std::vector<Member> members;
  members.reserve(47);
  std::set<std::pair<dev_t, ino_t>> identities;
  // Admit and retain every descriptor before changing any permission, so an
  // invalid late member does not cause predictable partial sealing.
  for (const auto& name : ExpectedMembers()) {
    Fd file(::openat(directory_fd, name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
    struct stat info{};
    if (file.value < 0 || ::fstat(file.value, &info) != 0 || !S_ISREG(info.st_mode) ||
        info.st_uid != ::geteuid() || info.st_nlink != 1 || info.st_dev != directory.st_dev ||
        !identities.emplace(info.st_dev, info.st_ino).second)
      return Status::FailedPrecondition("sealing requires distinct owned regular files without hardlinks");
    members.push_back({name, std::move(file), info});
  }
  for (const auto& member : members) {
    if (::fchmod(member.descriptor.value, 0444) != 0 || ::fsync(member.descriptor.value) != 0)
      return Status::Unavailable("generation moved but member sealing failed; permissions may be partial");
  }
  if (::fchmod(directory_fd, 0555) != 0 || ::fsync(directory_fd) != 0)
    return Status::Unavailable("generation moved but directory sealing failed; inspect permissions");
  for (const auto& member : members) {
    struct stat observed{}, named{};
    if (::fstat(member.descriptor.value, &observed) != 0 ||
        ::fstatat(directory_fd, member.name.c_str(), &named, AT_SYMLINK_NOFOLLOW) != 0 ||
        !S_ISREG(named.st_mode) || observed.st_dev != member.initial.st_dev ||
        observed.st_ino != member.initial.st_ino || named.st_dev != observed.st_dev ||
        named.st_ino != observed.st_ino || (observed.st_mode & 07777) != 0444 ||
        (named.st_mode & 07777) != 0444 || observed.st_nlink != 1 || named.st_nlink != 1 ||
        observed.st_size != member.initial.st_size || named.st_size != observed.st_size)
      return Status::FailedPrecondition("generation member identity or mode changed during sealing");
  }
  struct stat sealed{};
  if (::fstat(directory_fd, &sealed) != 0 || !Identity(directory, sealed) ||
      (sealed.st_mode & 07777) != 0555)
    return Status::FailedPrecondition("generation directory identity or mode changed during sealing");
  return ExactMembers(directory_fd);
}
}  // namespace
GenerationPromotion PromoteVerifiedGeneration(const std::filesystem::path& staging,
    const std::filesystem::path& generations_directory, const Sha256Digest& expected_root,
    const VerificationProgress& progress) {
  GenerationPromotion outcome;
  try {
    outcome.destination = generations_directory / ("sha256-" + expected_root.hex());
    auto fail = [&](Status status) { outcome.status = std::move(status); return outcome; };
    const auto source_name = staging.filename().string();
    if (source_name.empty() || source_name == "." || source_name == ".." ||
        staging == outcome.destination || expected_root == Sha256Digest{})
      return fail(Status::InvalidArgument("invalid promotion staging path or root"));
    auto source_parent = OpenDirectory(staging.parent_path());
    if (!source_parent.ok()) return fail(source_parent.status());
    auto destination_parent = OpenDirectory(generations_directory);
    if (!destination_parent.ok()) return fail(destination_parent.status());
    auto source = OpenDirectory(staging);
    if (!source.ok()) return fail(source.status());
    struct stat before{}, from{}, to{};
    if (::fstat(source->value, &before) != 0 || ::fstat(source_parent->value, &from) != 0 ||
        ::fstat(destination_parent->value, &to) != 0 || !Controlled(before) ||
        !Controlled(from) || !Controlled(to) || before.st_dev != to.st_dev || from.st_dev != to.st_dev)
      return fail(Status::FailedPrecondition("promotion requires controlled directories on one filesystem"));
    const auto name = outcome.destination.filename().string();
    struct stat existing{};
    if (::fstatat(destination_parent->value, name.c_str(), &existing, AT_SYMLINK_NOFOLLOW) == 0 || errno != ENOENT)
      return fail(Status::FailedPrecondition("promotion destination exists or cannot be inspected"));
    auto verified = VerifyGeneration(staging, expected_root, progress);
    if (!verified.ok()) return fail(verified.status());
    auto members = ExactMembers(source->value);
    if (!members.ok()) return fail(members);
    struct stat named{};
    if (verified->directory_device != static_cast<std::uint64_t>(before.st_dev) ||
        verified->directory_inode != static_cast<std::uint64_t>(before.st_ino) ||
        ::fstatat(source_parent->value, source_name.c_str(), &named, AT_SYMLINK_NOFOLLOW) != 0 ||
        !Identity(before, named))
      return fail(Status::FailedPrecondition("verified staging directory identity changed"));
    if (::fsync(source->value) != 0)
      return fail(Status::Unavailable("staging sync failed before promotion"));
    // Linux no-replace is mandatory; ordinary rename would permit overwriting.
    constexpr unsigned kRenameNoReplace = 1;
    if (::renameat2(source_parent->value, source_name.c_str(), destination_parent->value,
                    name.c_str(), kRenameNoReplace) != 0)
      return fail(Status::FailedPrecondition("atomic no-replace promotion failed; no overwrite fallback"));
    outcome.renamed = true;
    if (::fsync(source_parent->value) != 0 || ::fsync(destination_parent->value) != 0)
      return fail(Status::Unavailable("generation moved but parent sync failed; inspect destination"));
    outcome.directories_synced = true;
    auto sealed = SealReadOnly(source->value);
    if (!sealed.ok()) return fail(sealed);
    outcome.read_only_sealed = true;
    if (::fsync(destination_parent->value) != 0)
      return fail(Status::Unavailable("generation sealed but final parent sync failed"));
    auto final = VerifyGeneration(outcome.destination, expected_root, progress);
    if (!final.ok()) return fail(final.status());
    members = ExactMembers(source->value);
    if (!members.ok()) return fail(members);
    if (final->directory_device != static_cast<std::uint64_t>(before.st_dev) ||
        final->directory_inode != static_cast<std::uint64_t>(before.st_ino))
      return fail(Status::FailedPrecondition("generation moved but final directory identity differs"));
    return outcome;
  } catch (const std::exception& error) {
    outcome.status = Status::FailedPrecondition(error.what());
    return outcome;
  }
}
}  // namespace pih::offline_deepseek
