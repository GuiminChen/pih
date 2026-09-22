#include "active_generation.h"
#include "generation_store.h"
#include <cerrno>
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
  explicit Fd(int fd) : value(fd) { Check(fd >= 0, "cannot open active generation member without links"); }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  ~Fd() { ::close(value); }
};
bool Same(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode &&
      a.st_uid == b.st_uid && a.st_nlink == b.st_nlink && a.st_size == b.st_size &&
      a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
      a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
}  // namespace
Result<ActiveGenerationObservation> ObserveActiveGeneration(const std::filesystem::path& store_path,
    const Sha256Digest& expected_pointer_root, const Sha256Digest& expected_catalog_root,
    const VerificationProgress& progress) {
  try {
    Check(expected_pointer_root != Sha256Digest{} && expected_catalog_root != Sha256Digest{},
        "active generation requires expected pointer/catalog roots");
    auto store = Require(GenerationStore::Open(store_path));
    Check(::flock(store->root_descriptor(), LOCK_SH | LOCK_NB) == 0, "active generation store is busy");
    Require(store->Revalidate());
    auto pointer = Require(ReadCurrentGenerationPointer(*store));
    Check(pointer && pointer->pointer_root == expected_pointer_root &&
        pointer->input.catalog_root == expected_catalog_root, "active pointer/catalog differs from pinned expectation");
    const auto receipt_name = pointer->input.artifact_root.hex() + ".json";
    Fd receipt(::openat(store->receipts_descriptor(), receipt_name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
    struct stat before{}, parent{};
    Check(::fstat(receipt.value, &before) == 0 && ::fstat(store->receipts_descriptor(), &parent) == 0 &&
        S_ISREG(before.st_mode) && before.st_uid == ::geteuid() && before.st_nlink == 1 &&
        before.st_dev == parent.st_dev && (before.st_mode & 07777) == 0444 && before.st_size > 0 &&
        before.st_size <= (2LL << 20), "active receipt must be sealed within its byte limit");
    std::string bytes(static_cast<std::size_t>(before.st_size), '\0');
    for (std::size_t offset = 0; offset < bytes.size();) {
      const auto count = ::pread(receipt.value, bytes.data() + offset, bytes.size() - offset, static_cast<off_t>(offset));
      if (count < 0 && errno == EINTR) continue;
      Check(count > 0, "active receipt read failed");
      offset += static_cast<std::size_t>(count);
    }
    const auto generation_name = "sha256-" + pointer->input.artifact_root.hex();
    Fd directory(::openat(store->generations_descriptor(), generation_name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    struct stat initial{};
    Check(::fstat(directory.value, &initial) == 0 && S_ISDIR(initial.st_mode) && initial.st_dev == parent.st_dev,
        "active generation directory identity invalid");
    auto generation = Require(VerifyReceiptBoundGeneration(store_path / "generations" / generation_name,
        bytes, pointer->input.artifact_root, pointer->input.receipt_root, progress));
    Check(generation.receipt_binding_verified && generation.directory_device == static_cast<std::uint64_t>(initial.st_dev) &&
        generation.directory_inode == static_cast<std::uint64_t>(initial.st_ino), "active generation path replaced");
    Require(store->Revalidate());
    const auto current = Require(ReadCurrentGenerationPointer(*store));
    Check(current && current->json == pointer->json, "active pointer changed across generation verification");
    struct stat after{}, named{}, directory_after{}, directory_named{};
    Check(::fstat(receipt.value, &after) == 0 &&
        ::fstatat(store->receipts_descriptor(), receipt_name.c_str(), &named, AT_SYMLINK_NOFOLLOW) == 0 &&
        Same(before, after) && Same(before, named), "active receipt changed across verification");
    Check(::fstat(directory.value, &directory_after) == 0 &&
        ::fstatat(store->generations_descriptor(), generation_name.c_str(), &directory_named, AT_SYMLINK_NOFOLLOW) == 0 &&
        Same(initial, directory_after) && Same(initial, directory_named), "active generation changed across verification");
    return ActiveGenerationObservation{std::move(*pointer), std::move(generation)};
  } catch (const std::exception& error) { return Status::FailedPrecondition(error.what()); }
}
}  // namespace pih::offline_deepseek
