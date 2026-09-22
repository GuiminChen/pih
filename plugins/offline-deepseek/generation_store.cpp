#include "generation_store.h"
#include "identity_copy.h"
#include "pih/core/bounded_json.h"
#include "pih/core/canonical_json.h"
#include <array>
#include <cerrno>
#include <dirent.h>
#include <fcntl.h>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace pih::offline_deepseek {
namespace {
constexpr std::array<const char*, 4> kDirectories{".staging", "generations", "receipts", "pointers"};
template<class T> T Require(Result<T> result) {
  if (!result.ok()) throw std::runtime_error(std::string(result.status().message()));
  return std::move(*result);
}
void Check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Fd final {
  int value = -1;
  explicit Fd(int fd) : value(fd) { Check(fd >= 0, "cannot open generation store path without links"); }
  Fd(Fd&& other) noexcept : value(std::exchange(other.value, -1)) {}
  ~Fd() { if (value >= 0) ::close(value); }
};
Fd OpenDirectory(const std::filesystem::path& path) {
  Check(path.is_absolute(), "store path must be absolute");
  Fd result(::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  for (const auto& part : path.relative_path()) {
    const auto name = part.string();
    Check(!name.empty() && name != "." && name != ".." && name.find('\0') == name.npos,
        "store path must be canonical");
    Fd next(::openat(result.value, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    std::swap(result.value, next.value);
  }
  return result;
}
struct stat Controlled(int fd, bool directory) {
  struct stat result{};
  Check(::fstat(fd, &result) == 0 && (directory ? S_ISDIR(result.st_mode) : S_ISREG(result.st_mode)) &&
      result.st_uid == ::geteuid() && (result.st_mode & (S_IWGRP | S_IWOTH)) == 0,
      "store member must be owned and not group/other writable");
  return result;
}
bool Identity(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode && a.st_uid == b.st_uid;
}
bool Stable(const struct stat& a, const struct stat& b) {
  return Identity(a, b) && a.st_size == b.st_size && a.st_nlink == b.st_nlink &&
      a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
      a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
std::set<std::string> Names(int fd) {
  Fd scan(::openat(fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  auto* stream = ::fdopendir(scan.value);
  Check(stream != nullptr, "cannot enumerate generation store");
  scan.value = -1;
  std::unique_ptr<DIR, decltype(&::closedir)> owned(stream, &::closedir);
  std::set<std::string> names;
  errno = 0;
  while (const auto* entry = ::readdir(stream)) {
    const std::string_view name(entry->d_name);
    if (name != "." && name != "..") {
      Check(names.size() < 65536, "generation store entry count exceeds limit");
      Check(names.emplace(name).second, "duplicate generation store entry");
    }
    errno = 0;
  }
  Check(errno == 0, "generation store enumeration failed");
  return names;
}
std::string Manifest() {
  const auto text = [](std::string_view value) { return JsonValue(std::string(value)); };
  JsonValue::Object fields{
      {"schema", text("pih.deepseek_v4_flash_0731_generation_store.v1")},
      {"publication_abi", text("content_addressed_directory_generation_v1")},
      {"support_state", text("hardware_evidence_open")},
      {"directories", JsonValue(JsonValue::Object{{"staging", text(".staging")},
          {"generations", text("generations")}, {"receipts", text("receipts")}, {"pointers", text("pointers")}})}};
  const auto body = Require(canonical_ascii_json(JsonValue(fields), 64U << 10));
  fields.emplace_back("store_manifest_root", text(Require(sha256(std::as_bytes(std::span(body)))).hex()));
  return Require(canonical_ascii_json(JsonValue(std::move(fields)), 64U << 10));
}
void DigestName(std::string_view value) {
  const auto digest = Require(Sha256Digest::ParseHex(value));
  Check(digest != Sha256Digest{} && digest.hex() == value, "store entry digest name invalid");
}
}  // namespace
struct GenerationStore::Impl final {
  std::filesystem::path path;
  Fd root;
  std::array<int, 4> directories{-1, -1, -1, -1};
  int manifest = -1;
  struct stat root_identity{};
  std::array<struct stat, 4> directory_identities{};
  struct stat manifest_identity{};
  explicit Impl(std::filesystem::path value, Fd fd) : path(std::move(value)), root(std::move(fd)) {}
  ~Impl() {
    for (const auto fd : directories) if (fd >= 0) ::close(fd);
    if (manifest >= 0) ::close(manifest);
  }
  void Validate() const {
    const auto root_before = Controlled(root.value, true);
    const std::set<std::string> expected{"store.json", ".staging", "generations", "receipts", "pointers"};
    Check(Names(root.value) == expected, "generation store root member set differs");
    Check(Identity(root_identity, Controlled(root.value, true)), "store root identity changed");
    auto current = OpenDirectory(path);
    Check(Identity(root_identity, Controlled(current.value, true)), "store root path replaced");
    for (std::size_t i = 0; i < directories.size(); ++i) {
      const auto before = Controlled(directories[i], true);
      struct stat named{};
      Check(::fstatat(root.value, kDirectories[i], &named, AT_SYMLINK_NOFOLLOW) == 0 &&
          Identity(directory_identities[i], named) &&
          Identity(directory_identities[i], Controlled(directories[i], true)), "store subdirectory replaced");
      for (const auto& name : Names(directories[i])) {
        struct stat entry{};
        Check(::fstatat(directories[i], name.c_str(), &entry, AT_SYMLINK_NOFOLLOW) == 0 &&
            entry.st_dev == root_identity.st_dev && entry.st_uid == ::geteuid() &&
            (entry.st_mode & (S_IWGRP | S_IWOTH)) == 0, "store child identity/ownership invalid");
        if (i < 2) Check(S_ISDIR(entry.st_mode), "store child must be a non-link directory");
        else Check(S_ISREG(entry.st_mode) && entry.st_nlink == 1, "store child must be a singly-linked regular file");
        if (i == 1) {
          Check(name.starts_with("sha256-"), "stored generation name invalid");
          DigestName(std::string_view(name).substr(7));
        } else if (i == 2) {
          Check(name.ends_with(".json"), "stored receipt name invalid");
          DigestName(std::string_view(name).substr(0, name.size() - 5));
        } else if (i == 3) Check(name == "current.json", "store contains uncommitted/unknown pointer");
      }
      Check(Stable(before, Controlled(directories[i], true)), "store directory changed during enumeration");
    }
    const auto expected_manifest = Manifest();
    const auto observed = Controlled(manifest, false);
    Check(Identity(manifest_identity, observed) && observed.st_nlink == 1 &&
        observed.st_size == static_cast<off_t>(expected_manifest.size()), "store manifest identity/size changed");
    std::string bytes(expected_manifest.size(), '\0');
    for (std::size_t offset = 0; offset < bytes.size();) {
      const auto count = ::pread(manifest, bytes.data() + offset, bytes.size() - offset, static_cast<off_t>(offset));
      if (count < 0 && errno == EINTR) continue;
      Check(count > 0, "store manifest read failed");
      offset += static_cast<std::size_t>(count);
    }
    struct stat after{}, named{};
    Check(bytes == expected_manifest && ::fstat(manifest, &after) == 0 &&
        ::fstatat(root.value, "store.json", &named, AT_SYMLINK_NOFOLLOW) == 0 &&
        Identity(observed, after) && Identity(observed, named) &&
        after.st_size == observed.st_size && named.st_size == observed.st_size &&
        after.st_mtim.tv_sec == observed.st_mtim.tv_sec && after.st_mtim.tv_nsec == observed.st_mtim.tv_nsec &&
        after.st_ctim.tv_sec == observed.st_ctim.tv_sec && after.st_ctim.tv_nsec == observed.st_ctim.tv_nsec,
        "store manifest differs from frozen ABI or changed during read");
    Check(Stable(root_before, Controlled(root.value, true)), "store root changed during validation");
    auto final_root = OpenDirectory(path);
    Check(Identity(root_before, Controlled(final_root.value, true)), "store path replaced during validation");
  }
};
GenerationStore::GenerationStore(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GenerationStore::~GenerationStore() = default;
int GenerationStore::root_descriptor() const noexcept { return impl_->root.value; }
int GenerationStore::staging_descriptor() const noexcept { return impl_->directories[0]; }
int GenerationStore::generations_descriptor() const noexcept { return impl_->directories[1]; }
int GenerationStore::receipts_descriptor() const noexcept { return impl_->directories[2]; }
int GenerationStore::pointers_descriptor() const noexcept { return impl_->directories[3]; }
Status GenerationStore::Revalidate() const {
  try { impl_->Validate(); return Status::Ok(); }
  catch (const std::exception& error) { return Status::FailedPrecondition(error.what()); }
}
Result<std::unique_ptr<GenerationStore>> GenerationStore::Open(const std::filesystem::path& path) {
  try {
    auto impl = std::make_unique<Impl>(path, OpenDirectory(path));
    impl->root_identity = Controlled(impl->root.value, true);
    std::set<std::pair<dev_t, ino_t>> identities{{impl->root_identity.st_dev, impl->root_identity.st_ino}};
    for (std::size_t i = 0; i < kDirectories.size(); ++i) {
      impl->directories[i] = ::openat(impl->root.value, kDirectories[i], O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      const auto state = Controlled(impl->directories[i], true);
      Check(state.st_dev == impl->root_identity.st_dev && identities.emplace(state.st_dev, state.st_ino).second,
          "store directories must be distinct and on one filesystem");
      impl->directory_identities[i] = state;
    }
    impl->manifest = ::openat(impl->root.value, "store.json", O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    impl->manifest_identity = Controlled(impl->manifest, false);
    Check(impl->manifest_identity.st_dev == impl->root_identity.st_dev, "store manifest filesystem differs");
    impl->Validate();
    return std::unique_ptr<GenerationStore>(new GenerationStore(std::move(impl)));
  } catch (const std::exception& error) { return Status::FailedPrecondition(error.what()); }
}
Status InitializeGenerationStore(const std::filesystem::path& path) {
  bool created = false;
  try {
    Check(path.is_absolute() && path.has_filename(), "store initialization requires a named absolute path");
    const auto name = path.filename().string();
    Check(name != "." && name != ".." && name.find('\0') == name.npos, "invalid store directory name");
    auto parent = OpenDirectory(path.parent_path());
    const auto parent_identity = Controlled(parent.value, true);
    Check(::mkdirat(parent.value, name.c_str(), 0700) == 0, "store creation failed; existing paths are never reused");
    created = true;
    Fd root(::openat(parent.value, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    const auto root_identity = Controlled(root.value, true);
    Check(root_identity.st_dev == parent_identity.st_dev, "new store filesystem differs from parent");
    for (const auto* member : kDirectories) {
      Check(::mkdirat(root.value, member, 0700) == 0, "store subdirectory creation failed");
      Fd child(::openat(root.value, member, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
      Check(::fsync(child.value) == 0, "store child sync failed");
    }
    const auto manifest = Manifest();
    (void)Require(WriteMetadataMember(root.value, "store.json", manifest,
        Require(sha256(std::as_bytes(std::span(manifest))))));
    Check(::fsync(root.value) == 0 && ::fsync(parent.value) == 0, "store root/parent sync failed");
    auto observed = Require(GenerationStore::Open(path));
    Check(Identity(root_identity, Controlled(observed->root_descriptor(), true)), "initialized store path replaced");
    auto current_parent = OpenDirectory(path.parent_path());
    Check(Identity(parent_identity, Controlled(current_parent.value, true)), "store parent path replaced");
    return Status::Ok();
  } catch (const std::exception& error) {
    return Status::FailedPrecondition(std::string(error.what()) + (created ? "; partial store retained; inspect before retry" : ""));
  }
}
}  // namespace pih::offline_deepseek
