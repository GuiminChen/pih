#include "generation_activate.h"
#include "generation_receipt.h"
#include "generation_store.h"
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace pih::offline_deepseek {
namespace {
template<class T> T Require(Result<T> value) {
  if (!value.ok()) throw std::runtime_error(std::string(value.status().message()));
  return std::move(*value);
}
void Require(Status value) { if (!value.ok()) throw std::runtime_error(std::string(value.message())); }
void Check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Fd final {
  int value;
  explicit Fd(int fd) : value(fd) { Check(fd >= 0, "cannot open activation member without links"); }
  Fd(Fd&& other) noexcept : value(std::exchange(other.value, -1)) {}
  ~Fd() { if (value >= 0) ::close(value); }
};
bool Same(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode &&
      a.st_uid == b.st_uid && a.st_nlink == b.st_nlink && a.st_size == b.st_size &&
      a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
      a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
class Member final {
 public:
  Member(int parent, std::string name, std::string_view expected, bool sealed)
      : parent_(parent), name_(std::move(name)), fd_(::openat(parent, name_.c_str(),
          O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)) {
    struct stat directory{};
    Check(::fstat(fd_.value, &initial_) == 0 && ::fstat(parent_, &directory) == 0 &&
        S_ISREG(initial_.st_mode) && initial_.st_uid == ::geteuid() && initial_.st_nlink == 1 &&
        initial_.st_dev == directory.st_dev && initial_.st_size == static_cast<off_t>(expected.size()) &&
        (initial_.st_mode & 07777) == (sealed ? 0444 : 0600), "activation metadata identity/mode invalid");
    std::string bytes(expected.size(), '\0');
    for (std::size_t offset = 0; offset < bytes.size();) {
      const auto count = ::pread(fd_.value, bytes.data() + offset, bytes.size() - offset, static_cast<off_t>(offset));
      if (count < 0 && errno == EINTR) continue;
      Check(count > 0, "activation metadata read failed");
      offset += static_cast<std::size_t>(count);
    }
    Check(bytes == expected, "activation metadata differs from verified expectation");
    Stable();
  }
  void Stable() const {
    struct stat observed{}, named{};
    Check(::fstat(fd_.value, &observed) == 0 &&
        ::fstatat(parent_, name_.c_str(), &named, AT_SYMLINK_NOFOLLOW) == 0 &&
        Same(initial_, observed) && Same(initial_, named), "activation metadata changed or was replaced");
  }
 private:
  int parent_;
  std::string name_;
  Fd fd_;
  struct stat initial_{};
};
void AbsentCurrent(int parent) {
  struct stat value{};
  Check(::fstatat(parent, "current.json", &value, AT_SYMLINK_NOFOLLOW) != 0 && errno == ENOENT,
      "activation expected no current pointer");
}
void SealedGeneration(int directory) {
  struct stat root{};
  Check(::fstat(directory, &root) == 0 && S_ISDIR(root.st_mode) &&
      root.st_uid == ::geteuid() && (root.st_mode & 07777) == 0555, "activation generation is not sealed");
  std::vector<std::string> names{"model-endpoint.safetensors", "model.safetensors.index.json",
      "pih.runtime-records.json", "pih.manifest.json"};
  for (unsigned i = 0; i < 43; ++i) names.push_back("model-layers-" + std::to_string(i) + ".safetensors");
  for (const auto& name : names) {
    struct stat member{};
    Check(::fstatat(directory, name.c_str(), &member, AT_SYMLINK_NOFOLLOW) == 0 &&
        S_ISREG(member.st_mode) && member.st_uid == ::geteuid() && member.st_nlink == 1 &&
        member.st_dev == root.st_dev && (member.st_mode & 07777) == 0444,
        "activation generation member is not sealed");
  }
}
}  // namespace

GenerationActivation ActivateGeneration(const SourceArtifact& source,
    const SourcePreparationAuthority& authority, const std::filesystem::path& store_path,
    const Sha256Digest& artifact_root, const Sha256Digest& receipt_root, const Sha256Digest& catalog_root,
    std::uint64_t ordinal, const std::optional<Sha256Digest>& expected_previous,
    const VerificationProgress& progress) {
  GenerationActivation result;
  try {
    // Validate scalar/predecessor geometry before expensive source/target I/O.
    (void)Require(EncodeGenerationPointer({artifact_root, receipt_root, catalog_root, ordinal, expected_previous}));
    auto store = Require(GenerationStore::Open(store_path));
    Check(::flock(store->root_descriptor(), LOCK_EX | LOCK_NB) == 0, "generation store is busy or cannot be locked");
    Require(store->Revalidate());
    const auto current = Require(ReadCurrentGenerationPointer(*store));
    Check(current.has_value() == expected_previous.has_value() &&
        (!current || current->pointer_root == *expected_previous), "current pointer differs from expected predecessor");
    const auto next = Require(NextGenerationPointer(artifact_root, receipt_root, catalog_root, ordinal, current));
    result.pointer_root = next.pointer_root;
    std::unique_ptr<Member> old_pointer;
    if (current) old_pointer = std::make_unique<Member>(store->pointers_descriptor(), "current.json", current->json, false);
    const auto generation_name = "sha256-" + artifact_root.hex();
    Fd generation(::openat(store->generations_descriptor(), generation_name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    struct stat initial{};
    Check(::fstat(generation.value, &initial) == 0, "cannot inspect activation generation");
    SealedGeneration(generation.value);
    const auto observed = Require(VerifySourceBoundGeneration(source, authority,
        store_path / "generations" / generation_name, artifact_root, progress));
    Check(observed.directory_device == static_cast<std::uint64_t>(initial.st_dev) &&
        observed.directory_inode == static_cast<std::uint64_t>(initial.st_ino), "activation generation path replaced");
    const auto receipt = Require(EncodeGenerationReceipt(observed));
    Check(receipt.receipt_root == receipt_root, "activation receipt root differs from source-bound verification");
    Member receipt_file(store->receipts_descriptor(), artifact_root.hex() + ".json", receipt.json, true);
    Require(source.Revalidate());
    Require(store->Revalidate());
    SealedGeneration(generation.value);
    struct stat named{};
    Check(::fstatat(store->generations_descriptor(), generation_name.c_str(), &named, AT_SYMLINK_NOFOLLOW) == 0 &&
        Same(initial, named), "activation generation changed across verification");
    if (old_pointer) old_pointer->Stable(); else AbsentCurrent(store->pointers_descriptor());
    const auto temporary = ".current-" + std::to_string(ordinal) + "-" + next.pointer_root.hex() + ".tmp";
    result.temporary_may_exist = true;
    const auto pointer_digest = Require(sha256(std::as_bytes(std::span(next.json))));
    (void)Require(WriteMetadataMember(store->pointers_descriptor(), temporary, next.json, pointer_digest));
    Member temporary_file(store->pointers_descriptor(), temporary, next.json, false);
    receipt_file.Stable();
    if (old_pointer) old_pointer->Stable(); else AbsentCurrent(store->pointers_descriptor());
    temporary_file.Stable();
    if (current) {
      Check(::renameat(store->pointers_descriptor(), temporary.c_str(), store->pointers_descriptor(), "current.json") == 0,
          "atomic current pointer replacement failed");
    } else {
      Check(::renameat2(store->pointers_descriptor(), temporary.c_str(), store->pointers_descriptor(), "current.json", 1) == 0,
          "atomic initial pointer creation failed; no overwrite fallback");
    }
    result.pointer_replaced = true;
    result.temporary_may_exist = false;
    Check(::fsync(store->pointers_descriptor()) == 0 && ::fsync(store->root_descriptor()) == 0,
        "pointer replaced but directory synchronization failed");
    result.directories_synced = true;
    const auto activated = Require(ReadCurrentGenerationPointer(*store));
    Check(activated && activated->json == next.json, "current pointer differs after atomic activation");
    SealedGeneration(generation.value);
    Check(::fstatat(store->generations_descriptor(), generation_name.c_str(), &named, AT_SYMLINK_NOFOLLOW) == 0 &&
        Same(initial, named), "activated generation path changed");
    receipt_file.Stable();
    Require(source.Revalidate());
    Require(store->Revalidate());
    result.activated = true;
  } catch (const std::exception& error) { result.status = Status::FailedPrecondition(error.what()); }
  return result;
}
}  // namespace pih::offline_deepseek
