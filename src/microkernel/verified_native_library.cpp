#include "microkernel/verified_native_library.h"

#include <array>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>

#include "pih/core/sha256.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <dlfcn.h>
#if defined(__linux__)
#include <linux/memfd.h>
#include <sys/syscall.h>
#endif
#endif

namespace pih::microkernel {
namespace {

bool IsAbsolutePath(const std::string& path) {
#if defined(_WIN32)
  return path.size() >= 3 &&
         ((path[0] >= 'A' && path[0] <= 'Z') ||
          (path[0] >= 'a' && path[0] <= 'z')) &&
         path[1] == ':' && (path[2] == '\\' || path[2] == '/');
#else
  return !path.empty() && path.front() == '/';
#endif
}

#if !defined(_WIN32)

class FileDescriptor final {
 public:
  explicit FileDescriptor(int value) noexcept : value_(value) {}
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&& other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  ~FileDescriptor() {
    if (value_ >= 0) (void)::close(value_);
  }
  [[nodiscard]] int get() const noexcept { return value_; }

 private:
  int value_;
};

#if defined(__linux__)
FileDescriptor VerifiedSnapshot(int descriptor, std::string_view expected_hex,
                                const char* failure) {
  const auto expected = pih::Sha256Digest::ParseHex(expected_hex);
  if (!expected.ok() || *expected == pih::Sha256Digest{})
    throw std::invalid_argument(failure);
  struct stat identity {};
  if (::fstat(descriptor, &identity) != 0 || !S_ISREG(identity.st_mode) ||
      identity.st_size <= 0) throw std::runtime_error(failure);
  const FileDescriptor snapshot(static_cast<int>(::syscall(
      SYS_memfd_create, "pih-native-plugin", MFD_CLOEXEC | MFD_ALLOW_SEALING | MFD_EXEC)));
  if (snapshot.get() < 0) throw std::runtime_error("native_library_snapshot_unavailable");
  pih::Sha256 digest;
  std::array<std::byte, 1024 * 1024> buffer{};
  off_t offset = 0;
  while (offset < identity.st_size) {
    const auto wanted = static_cast<std::size_t>(std::min<off_t>(
        identity.st_size - offset, static_cast<off_t>(buffer.size())));
    ssize_t count = ::pread(descriptor, buffer.data(), wanted, offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) throw std::runtime_error(failure);
    const auto status = digest.update(
        std::span<const std::byte>(buffer.data(),
                                  static_cast<std::size_t>(count)));
    if (!status.ok()) throw std::runtime_error(failure);
    std::size_t written = 0;
    while (written < static_cast<std::size_t>(count)) {
      const auto bytes = ::pwrite(snapshot.get(), buffer.data() + written,
          static_cast<std::size_t>(count) - written, offset + written);
      if (bytes < 0 && errno == EINTR) continue;
      if (bytes <= 0) throw std::runtime_error("native_library_snapshot_write_failed");
      written += static_cast<std::size_t>(bytes);
    }
    offset += count;
  }
  const auto observed = digest.finalize();
  if (!observed.ok() || *observed != *expected) {
    throw std::runtime_error(failure);
  }
  constexpr int seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
  if (::fchmod(snapshot.get(), 0500) != 0 ||
      ::fcntl(snapshot.get(), F_ADD_SEALS, seals) != 0 ||
      ::fcntl(snapshot.get(), F_GET_SEALS) != seals)
    throw std::runtime_error("native_library_snapshot_seal_failed");
  // Return a separate descriptor while the local owner closes its copy.
  const int owned = ::fcntl(snapshot.get(), F_DUPFD_CLOEXEC, 3);
  if (owned < 0) throw std::runtime_error("native_library_snapshot_duplicate_failed");
  return FileDescriptor(owned);
}
#endif

FileDescriptor DuplicateWithUniqueProcessNumber(int source,
                                                const char* failure) {
  // dlopen caches native objects by loader-visible names. A closed descriptor
  // number must therefore never be reused in another /proc/self/fd path during
  // this process, even though the descriptor itself is only needed through
  // the eager load call.
  static std::mutex mutex;
  static int next_descriptor = 3;
  std::lock_guard lock(mutex);
  if (next_descriptor == std::numeric_limits<int>::max()) {
    throw std::runtime_error(failure);
  }
  const int duplicated =
      ::fcntl(source, F_DUPFD_CLOEXEC, next_descriptor);
  if (duplicated < 0 || duplicated == std::numeric_limits<int>::max()) {
    if (duplicated >= 0) (void)::close(duplicated);
    throw std::runtime_error(failure);
  }
  next_descriptor = duplicated + 1;
  return FileDescriptor(duplicated);
}

#endif

}  // namespace

void* OpenVerifiedNativeLibrary(const std::string& absolute_path,
                                std::string_view expected_sha256_hex,
                                const char* load_failure) {
  if (!IsAbsolutePath(absolute_path)) {
    throw std::invalid_argument("native_library_path_must_be_absolute");
  }
#if defined(_WIN32)
  if (!expected_sha256_hex.empty()) {
    throw std::runtime_error("verified_native_library_requires_linux");
  }
  const auto handle = LoadLibraryExA(absolute_path.c_str(), nullptr,
                                     LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                         LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (handle == nullptr) throw std::runtime_error(load_failure);
  return reinterpret_cast<void*>(handle);
#else
  if (expected_sha256_hex.empty()) {
    void* handle = dlopen(absolute_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) throw std::runtime_error(load_failure);
    return handle;
  }
#if !defined(__linux__)
  throw std::runtime_error("verified_native_library_requires_linux");
#else
  const FileDescriptor source_descriptor(
      ::open(absolute_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
  if (source_descriptor.get() < 0) throw std::runtime_error(load_failure);
  const auto snapshot = VerifiedSnapshot(source_descriptor.get(),
      expected_sha256_hex, "native_library_digest_mismatch");
  const auto descriptor = DuplicateWithUniqueProcessNumber(
      snapshot.get(), load_failure);
  // File-descriptor pinning alone does not prevent another writer modifying
  // the source after hashing. Map only the sealed, authenticated snapshot.
  const auto descriptor_path =
      std::string("/proc/self/fd/") + std::to_string(descriptor.get());
  void* handle = dlopen(descriptor_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) throw std::runtime_error(load_failure);
  return handle;
#endif
#endif
}

}  // namespace pih::microkernel
