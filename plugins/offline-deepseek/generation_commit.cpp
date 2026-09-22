#include "generation_commit.h"
#include "generation_receipt.h"
#include "generation_store.h"
#include <cerrno>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/file.h>
#include <unistd.h>
#include <utility>

namespace pih::offline_deepseek {
namespace {
template<class T> T Require(Result<T> value) {
  if (!value.ok()) throw std::runtime_error(std::string(value.status().message()));
  return std::move(*value);
}
void Require(Status value) {
  if (!value.ok()) throw std::runtime_error(std::string(value.message()));
}
void Check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Fd final {
  int value;
  explicit Fd(int fd) : value(fd) { Check(value >= 0, "cannot open commit member without links"); }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  ~Fd() { ::close(value); }
};
void Absent(int parent, const std::string& name) {
  struct stat ignored{};
  errno = 0;
  Check(::fstatat(parent, name.c_str(), &ignored, AT_SYMLINK_NOFOLLOW) != 0 && errno == ENOENT,
      "generation/receipt destination exists or cannot be inspected");
}
bool Identity(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}
bool Matches(const GenerationObservation& observed, const struct stat& identity) {
  return observed.directory_device == static_cast<std::uint64_t>(identity.st_dev) &&
      observed.directory_inode == static_cast<std::uint64_t>(identity.st_ino);
}
void SealReceipt(int parent, const std::string& name, const EncodedGenerationReceipt& expected) {
  Fd file(::openat(parent, name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
  struct stat before{}, directory{};
  Check(::fstat(file.value, &before) == 0 && ::fstat(parent, &directory) == 0 &&
      S_ISREG(before.st_mode) && before.st_uid == ::geteuid() && before.st_nlink == 1 &&
      before.st_dev == directory.st_dev && (before.st_mode & (S_IWGRP | S_IWOTH)) == 0 &&
      before.st_size == static_cast<off_t>(expected.json.size()), "commit receipt identity/size invalid");
  std::string bytes(expected.json.size(), '\0');
  for (std::size_t offset = 0; offset < bytes.size();) {
    const auto count = ::pread(file.value, bytes.data() + offset, bytes.size() - offset, static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    Check(count > 0, "commit receipt reread failed");
    offset += static_cast<std::size_t>(count);
  }
  Check(bytes == expected.json && Require(sha256(std::as_bytes(std::span(bytes)))) == expected.object_sha256,
      "commit receipt bytes differ from verified projection");
  struct stat read_after{}, named{};
  Check(::fstat(file.value, &read_after) == 0 && Identity(before, read_after) &&
      before.st_size == read_after.st_size && before.st_mode == read_after.st_mode &&
      before.st_mtim.tv_sec == read_after.st_mtim.tv_sec && before.st_mtim.tv_nsec == read_after.st_mtim.tv_nsec &&
      before.st_ctim.tv_sec == read_after.st_ctim.tv_sec && before.st_ctim.tv_nsec == read_after.st_ctim.tv_nsec,
      "commit receipt changed during reread");
  Check(::fchmod(file.value, 0444) == 0 && ::fsync(file.value) == 0 && ::fsync(parent) == 0,
      "commit receipt sealing or synchronization failed");
  struct stat after{};
  Check(::fstat(file.value, &after) == 0 &&
      ::fstatat(parent, name.c_str(), &named, AT_SYMLINK_NOFOLLOW) == 0 &&
      Identity(before, after) && Identity(after, named) && S_ISREG(named.st_mode) &&
      (after.st_mode & 07777) == 0444 && named.st_mode == after.st_mode &&
      after.st_nlink == 1 && named.st_nlink == 1 && after.st_uid == ::geteuid() &&
      after.st_size == before.st_size && named.st_size == after.st_size &&
      after.st_mtim.tv_sec == before.st_mtim.tv_sec && after.st_mtim.tv_nsec == before.st_mtim.tv_nsec,
      "sealed commit receipt changed or was replaced");
}
}  // namespace

GenerationCommit CommitGeneration(const SourceArtifact& source,
    const SourcePreparationAuthority& authority, const std::filesystem::path& store_path,
    std::string_view staging_name, const Sha256Digest& artifact_root, const VerificationProgress& progress) {
  GenerationCommit result;
  try {
    Check(artifact_root != Sha256Digest{} && !staging_name.empty() && staging_name.size() <= 255 &&
        staging_name != "." && staging_name != ".." && staging_name.find_first_of("/\\") == staging_name.npos &&
        staging_name.find('\0') == staging_name.npos, "commit artifact root or staging name invalid");
    auto store = Require(GenerationStore::Open(store_path));
    Check(::flock(store->root_descriptor(), LOCK_EX | LOCK_NB) == 0, "generation store is busy or cannot be locked");
    Require(store->Revalidate());
    const std::string name(staging_name);
    const auto generation_name = "sha256-" + artifact_root.hex();
    const auto receipt_name = artifact_root.hex() + ".json";
    Absent(store->generations_descriptor(), generation_name);
    Absent(store->receipts_descriptor(), receipt_name);
    Fd staging(::openat(store->staging_descriptor(), name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    struct stat initial{}, parent{};
    Check(::fstat(staging.value, &initial) == 0 && ::fstat(store->root_descriptor(), &parent) == 0 &&
        S_ISDIR(initial.st_mode) && initial.st_uid == ::geteuid() &&
        (initial.st_mode & (S_IWGRP | S_IWOTH)) == 0 && initial.st_dev == parent.st_dev,
        "commit staging must be owned, controlled and on the store filesystem");
    const auto path = store_path / ".staging" / name;
    const auto observed = Require(VerifySourceBoundGeneration(source, authority, path, artifact_root, progress));
    Check(Matches(observed, initial), "commit staging identity differs from verified source-bound directory");
    const auto receipt = Require(EncodeGenerationReceipt(observed));
    result.receipt_root = receipt.receipt_root;
    result.verification_projection_sha256 = observed.verification_projection_sha256;
    Require(store->Revalidate());
    Require(source.Revalidate());
    Absent(store->generations_descriptor(), generation_name);
    Absent(store->receipts_descriptor(), receipt_name);
    struct stat named{};
    Check(::fstatat(store->staging_descriptor(), name.c_str(), &named, AT_SYMLINK_NOFOLLOW) == 0 &&
        Identity(initial, named) && S_ISDIR(named.st_mode), "commit staging replaced before promotion");
    result.promotion = PromoteVerifiedGeneration(path, store_path / "generations", artifact_root, progress);
    Require(result.promotion.status);
    Check(result.promotion.renamed && result.promotion.directories_synced && result.promotion.read_only_sealed,
        "commit promotion did not complete all durability/sealing phases");
    Require(store->Revalidate());
    const auto final = Require(VerifySourceBoundGeneration(source, authority, result.promotion.destination,
        artifact_root, progress));
    Check(Matches(final, initial) && final.verification_projection_json == observed.verification_projection_json &&
        final.verification_projection_sha256 == observed.verification_projection_sha256,
        "committed directory differs from source-bound staging observation");
    Require(ValidateGenerationReceipt(receipt.json, final));
    Require(source.Revalidate());
    Require(store->Revalidate());
    Absent(store->receipts_descriptor(), receipt_name);
    // Set before the first possible creation. A failed exclusive write may leave
    // an empty/partial file; its existence is intentionally never rolled back.
    result.receipt_may_exist = true;
    (void)Require(WriteMetadataMember(store->receipts_descriptor(), receipt_name,
        receipt.json, receipt.object_sha256));
    SealReceipt(store->receipts_descriptor(), receipt_name, receipt);
    Require(store->Revalidate());
    Require(source.Revalidate());
    result.receipt_committed = true;
  } catch (const std::exception& error) {
    result.status = Status::FailedPrecondition(error.what());
  }
  return result;
}
}  // namespace pih::offline_deepseek
