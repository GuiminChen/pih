#include "pih/io/controller_file_lease.h"

#include "pih/io/descriptor_mapped_range.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>

#include "pih/core/checked_math.h"

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/ioctl.h>
#if __has_include(<linux/fsverity.h>)
#include <linux/fsverity.h>
#define PIH_HAS_FSVERITY_UAPI 1
#endif
#endif

namespace pih {

Status validate_fsverity_sha256_measurement(
    const Sha256Digest& expected, std::uint16_t measured_algorithm,
    std::span<const std::byte> measured_digest) {
  constexpr std::uint16_t kFsVeritySha256Algorithm = 1;
  if (measured_algorithm != kFsVeritySha256Algorithm ||
      measured_digest.size() != expected.bytes.size()) {
    return Status::FailedPrecondition(
        "fs-verity measurement is not SHA-256");
  }
  if (!std::equal(expected.bytes.begin(), expected.bytes.end(),
                  measured_digest.begin())) {
    return Status::FailedPrecondition(
        "fs-verity measurement differs from expected digest");
  }
  return Status::Ok();
}

struct ArtifactDescriptorImpl final {
#ifdef _WIN32
  HANDLE handle = INVALID_HANDLE_VALUE;
#else
  int handle = -1;
#endif
  ArtifactFileIdentity identity;

  ~ArtifactDescriptorImpl() {
#ifdef _WIN32
    if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
    if (handle >= 0) close(handle);
#endif
  }
};

struct ArtifactDirectoryImpl final {
#ifdef _WIN32
  HANDLE handle = INVALID_HANDLE_VALUE;
  std::filesystem::path canonical_path;
#else
  int handle = -1;
#endif

  ~ArtifactDirectoryImpl() {
#ifdef _WIN32
    if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
    if (handle >= 0) close(handle);
#endif
  }
};

struct DescriptorMappedRange::Impl final {
  void* mapping_base = nullptr;
  std::size_t mapping_bytes = 0;
#ifdef _WIN32
  HANDLE mapping = nullptr;
#endif

  ~Impl() {
#ifdef _WIN32
    if (mapping_base != nullptr) UnmapViewOfFile(mapping_base);
    if (mapping != nullptr) CloseHandle(mapping);
#else
    if (mapping_base != nullptr) munmap(mapping_base, mapping_bytes);
#endif
  }
};

namespace {

Result<ArtifactFileIdentity> query_identity(
    const ArtifactDescriptorImpl& impl) {
  ArtifactFileIdentity identity;
#ifdef _WIN32
  BY_HANDLE_FILE_INFORMATION info{};
  if (!GetFileInformationByHandle(impl.handle, &info) ||
      (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
      (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
    return Status::Unavailable("cannot query artifact descriptor identity");
  }
  identity.file_bytes =
      (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32U) |
      info.nFileSizeLow;
  identity.filesystem_identity = info.dwVolumeSerialNumber;
  identity.file_identity =
      (static_cast<std::uint64_t>(info.nFileIndexHigh) << 32U) |
      info.nFileIndexLow;
  const auto ticks =
      (static_cast<std::uint64_t>(info.ftLastWriteTime.dwHighDateTime) << 32U) |
      info.ftLastWriteTime.dwLowDateTime;
  identity.data_mtime_seconds = static_cast<std::int64_t>(ticks / 10'000'000ULL);
  identity.data_mtime_nanoseconds =
      static_cast<std::uint32_t>((ticks % 10'000'000ULL) * 100ULL);
#else
  struct stat info {};
  if (fstat(impl.handle, &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_size < 0) {
    return Status::Unavailable("cannot query artifact descriptor identity");
  }
  identity.file_bytes = static_cast<std::uint64_t>(info.st_size);
  identity.filesystem_identity = static_cast<std::uint64_t>(info.st_dev);
  identity.file_identity = static_cast<std::uint64_t>(info.st_ino);
  identity.data_mtime_seconds = info.st_mtim.tv_sec;
  identity.data_mtime_nanoseconds =
      static_cast<std::uint32_t>(info.st_mtim.tv_nsec);
#endif
  return identity;
}

bool valid_basename(std::string_view name) {
  if (name.empty() || name == "." || name == ".." || name.size() > 255)
    return false;
  for (const unsigned char value : name) {
    const bool allowed =
        (value >= 'a' && value <= 'z') ||
        (value >= 'A' && value <= 'Z') ||
        (value >= '0' && value <= '9') || value == '.' || value == '_' ||
        value == '-';
    if (!allowed) return false;
  }
  return true;
}

Status read_from(const ArtifactDescriptorImpl& impl, std::uint64_t offset,
                 std::span<std::byte> output) {
  auto end = checked_add_u64(offset, output.size());
  if (!end.ok() || *end > impl.identity.file_bytes) {
    return Status::InvalidArgument("artifact descriptor read is out of bounds");
  }
  std::size_t completed = 0;
  while (completed < output.size()) {
#ifdef _WIN32
    const auto position = offset + completed;
    OVERLAPPED overlapped{};
    overlapped.Offset = static_cast<DWORD>(position & 0xffffffffULL);
    overlapped.OffsetHigh = static_cast<DWORD>(position >> 32U);
    const auto remaining = output.size() - completed;
    const DWORD request = static_cast<DWORD>(std::min<std::size_t>(
        remaining, std::numeric_limits<DWORD>::max()));
    DWORD received = 0;
    const BOOL immediate = ReadFile(impl.handle, output.data() + completed,
                                    request, &received, &overlapped);
    if (!immediate && GetLastError() == ERROR_IO_PENDING) {
      if (!GetOverlappedResult(impl.handle, &overlapped, &received, TRUE)) {
        return Status::Unavailable("artifact descriptor read failed");
      }
    } else if (!immediate) {
      return Status::Unavailable("artifact descriptor read failed");
    }
    if (received == 0) {
      return Status::Unavailable("artifact descriptor read failed");
    }
    completed += received;
#else
    const auto remaining = output.size() - completed;
    const auto request = std::min<std::size_t>(
        remaining, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
    const auto received = pread(impl.handle, output.data() + completed, request,
                                static_cast<off_t>(offset + completed));
    if (received < 0 && errno == EINTR) continue;
    if (received <= 0)
      return Status::Unavailable("artifact descriptor read failed");
    completed += static_cast<std::size_t>(received);
#endif
  }
  return Status::Ok();
}

Result<std::unique_ptr<ArtifactDescriptorImpl>> duplicate_impl(
    const ArtifactDescriptorImpl& source) {
  auto result = std::make_unique<ArtifactDescriptorImpl>();
#ifdef _WIN32
  if (!DuplicateHandle(GetCurrentProcess(), source.handle, GetCurrentProcess(),
                       &result->handle, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
    return Status::ResourceExhausted("cannot duplicate artifact descriptor");
  }
#else
  result->handle = fcntl(source.handle, F_DUPFD_CLOEXEC, 0);
  if (result->handle < 0)
    return Status::ResourceExhausted("cannot duplicate artifact descriptor");
#endif
  result->identity = source.identity;
  return result;
}

}  // namespace

Result<ControllerFileLease> ControllerFileLease::OpenBeneath(
    const std::filesystem::path& trusted_root,
    std::string_view member_basename, std::uint64_t maximum_bytes,
    ArtifactImmutabilityMode immutability_mode) {
  if (trusted_root.empty() || !valid_basename(member_basename) ||
      maximum_bytes == 0) {
    return Status::InvalidArgument("artifact lease request is invalid");
  }
  // Verity claims require a separately verified kernel receipt. Until that
  // receipt is supplied, a normal descriptor is development-only.
  if (immutability_mode != ArtifactImmutabilityMode::kUncalibrated) {
    return Status::FailedPrecondition(
        "artifact verity mode requires an enforcement receipt");
  }
  auto authority = ArtifactDirectoryAuthority::Open(trusted_root);
  if (!authority.ok()) return authority.status();
  return OpenBeneath(*authority, member_basename, maximum_bytes,
                     immutability_mode);
}

Result<ArtifactDirectoryAuthority> ArtifactDirectoryAuthority::Open(
    const std::filesystem::path& trusted_root) {
  if (trusted_root.empty())
    return Status::InvalidArgument("artifact root is invalid");
  auto impl = std::make_unique<ArtifactDirectoryImpl>();
#ifdef _WIN32
  std::error_code error;
  impl->canonical_path = std::filesystem::canonical(trusted_root, error);
  if (error) return Status::Unavailable("cannot resolve artifact root");
  impl->handle = CreateFileW(
      impl->canonical_path.c_str(), FILE_READ_ATTRIBUTES,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
      nullptr);
  if (impl->handle == INVALID_HANDLE_VALUE)
    return Status::Unavailable("cannot open artifact root");
#else
  impl->handle = open(trusted_root.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
  if (impl->handle < 0) return Status::Unavailable("cannot open artifact root");
#endif
  return ArtifactDirectoryAuthority(std::move(impl));
}

ArtifactDirectoryAuthority::~ArtifactDirectoryAuthority() = default;
ArtifactDirectoryAuthority::ArtifactDirectoryAuthority(
    std::unique_ptr<ArtifactDirectoryImpl> impl)
    : impl_(std::move(impl)) {}
ArtifactDirectoryAuthority::ArtifactDirectoryAuthority(
    ArtifactDirectoryAuthority&&) noexcept = default;
ArtifactDirectoryAuthority& ArtifactDirectoryAuthority::operator=(
    ArtifactDirectoryAuthority&&) noexcept = default;

Result<ControllerFileLease> ControllerFileLease::OpenBeneath(
    const ArtifactDirectoryAuthority& trusted_root,
    std::string_view member_basename, std::uint64_t maximum_bytes,
    ArtifactImmutabilityMode immutability_mode) {
  if (trusted_root.impl_ == nullptr || !valid_basename(member_basename) ||
      maximum_bytes == 0) {
    return Status::InvalidArgument("artifact lease request is invalid");
  }
  if (immutability_mode != ArtifactImmutabilityMode::kUncalibrated) {
    return Status::FailedPrecondition(
        "artifact verity mode requires an enforcement receipt");
  }
  auto impl = std::make_unique<ArtifactDescriptorImpl>();
#ifdef _WIN32
  const auto path = trusted_root.impl_->canonical_path /
                    std::filesystem::path(std::string(member_basename));
  impl->handle = CreateFileW(
      path.c_str(), GENERIC_READ,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT |
                         FILE_FLAG_OVERLAPPED,
      nullptr);
  if (impl->handle == INVALID_HANDLE_VALUE)
    return Status::Unavailable("cannot open artifact descriptor");
#else
  impl->handle = openat(trusted_root.impl_->handle,
                        std::string(member_basename).c_str(),
                        O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (impl->handle < 0)
    return Status::Unavailable("cannot open artifact descriptor");
#endif
  auto identity = query_identity(*impl);
  if (!identity.ok()) return identity.status();
  impl->identity = *identity;
  if (impl->identity.file_bytes > maximum_bytes) {
    return Status::ResourceExhausted("artifact descriptor exceeds byte budget");
  }
  return ControllerFileLease(std::move(impl), immutability_mode, false);
}

Result<ControllerFileLease> ControllerFileLease::OpenBeneathFsVerity(
    const std::filesystem::path& trusted_root,
    std::string_view member_basename, std::uint64_t maximum_bytes,
    const Sha256Digest& expected_digest) {
  auto lease = OpenBeneath(trusted_root, member_basename, maximum_bytes,
                           ArtifactImmutabilityMode::kUncalibrated);
  if (!lease.ok()) return lease.status();
#if defined(_WIN32) || !defined(PIH_HAS_FSVERITY_UAPI)
  return Status::FailedPrecondition(
      "fs-verity measurement is unavailable on this platform");
#else
  constexpr std::size_t kMaximumDigestBytes = 64;
  alignas(fsverity_digest)
      std::array<unsigned char,
                 sizeof(fsverity_digest) + kMaximumDigestBytes> buffer{};
  auto* measurement =
      reinterpret_cast<fsverity_digest*>(buffer.data());
  measurement->digest_size = kMaximumDigestBytes;
  if (ioctl(lease->impl_->handle, FS_IOC_MEASURE_VERITY, measurement) != 0) {
    return Status::FailedPrecondition(
        "artifact descriptor is not enforced by fs-verity");
  }
  if (measurement->digest_size > kMaximumDigestBytes) {
    return Status::FailedPrecondition(
        "fs-verity measurement exceeds digest budget");
  }
  const auto* digest_begin = reinterpret_cast<const std::byte*>(
      buffer.data() + sizeof(fsverity_digest));
  const auto measured = std::span<const std::byte>(
      digest_begin, measurement->digest_size);
  const auto validated = validate_fsverity_sha256_measurement(
      expected_digest, measurement->digest_algorithm, measured);
  if (!validated.ok()) return validated;
  lease->immutability_mode_ = ArtifactImmutabilityMode::kFsVerity;
  lease->production_eligible_ = true;
  lease->enforced_digest_ = expected_digest;
  return lease;
#endif
}

Result<ControllerFileLease>
ControllerFileLease::OpenBeneathDmVeritySnapshot(
    const std::filesystem::path& trusted_root,
    std::string_view member_basename, std::uint64_t maximum_bytes,
    const DmVeritySupervisorReceipt& supervisor_receipt,
    const Sha256Digest& expected_root_digest,
    const Sha256Digest& expected_table_digest,
    const Sha256Digest& expected_supervisor_attestation_digest,
    std::uint64_t integrity_reserve_bytes) {
  auto lease = OpenBeneath(trusted_root, member_basename, maximum_bytes,
                           ArtifactImmutabilityMode::kUncalibrated);
  if (!lease.ok()) return lease.status();
#ifdef _WIN32
  return Status::FailedPrecondition(
      "dm-verity snapshot leases require the Linux production runtime");
#else
  auto capacity = validate_dm_verity_supervisor_receipt(
      supervisor_receipt, expected_root_digest, expected_table_digest,
      expected_supervisor_attestation_digest,
      lease->identity().filesystem_identity, integrity_reserve_bytes);
  if (!capacity.ok()) return capacity.status();
  lease->immutability_mode_ = ArtifactImmutabilityMode::kDmVeritySnapshot;
  lease->production_eligible_ = true;
  lease->enforced_digest_ = expected_root_digest;
  lease->integrity_table_digest_ = expected_table_digest;
  lease->integrity_owner_bytes_ = capacity->total_integrity_bytes;
  lease->supervisor_attestation_digest_ =
      expected_supervisor_attestation_digest;
  lease->integrity_reserve_bytes_ = integrity_reserve_bytes;
  return lease;
#endif
}

ControllerFileLease::~ControllerFileLease() = default;
ControllerFileLease::ControllerFileLease(
    std::unique_ptr<ArtifactDescriptorImpl> impl,
    ArtifactImmutabilityMode mode, bool production_eligible)
    : impl_(std::move(impl)),
      immutability_mode_(mode),
      production_eligible_(production_eligible) {}
ControllerFileLease::ControllerFileLease(ControllerFileLease&&) noexcept =
    default;
ControllerFileLease& ControllerFileLease::operator=(
    ControllerFileLease&&) noexcept = default;

const ArtifactFileIdentity& ControllerFileLease::identity() const noexcept {
  return impl_->identity;
}

Status ControllerFileLease::read_exact(
    std::uint64_t offset, std::span<std::byte> output) const {
  return read_from(*impl_, offset, output);
}

Status ControllerFileLease::poll_identity_unchanged() const {
  if (impl_ == nullptr)
    return Status::FailedPrecondition("artifact lease is not initialized");
  auto current = query_identity(*impl_);
  if (!current.ok()) return current.status();
  if (*current != impl_->identity) {
    return Status::FailedPrecondition(
        "artifact descriptor identity mutated during engine epoch");
  }
  return Status::Ok();
}

Status ControllerFileLease::poll_integrity_unchanged(
    const DmVeritySupervisorReceipt* current_dm_receipt) const {
  const auto identity = poll_identity_unchanged();
  if (!identity.ok()) return identity;
  if (immutability_mode_ == ArtifactImmutabilityMode::kUncalibrated) {
    if (current_dm_receipt != nullptr)
      return Status::InvalidArgument(
          "dm-verity receipt was supplied to an uncalibrated lease");
    return Status::Ok();
  }
  if (immutability_mode_ == ArtifactImmutabilityMode::kFsVerity) {
    if (current_dm_receipt != nullptr)
      return Status::InvalidArgument(
          "dm-verity receipt was supplied to an fs-verity lease");
#if defined(_WIN32) || !defined(PIH_HAS_FSVERITY_UAPI)
    return Status::FailedPrecondition(
        "fs-verity polling is unavailable on this platform");
#else
    constexpr std::size_t kMaximumDigestBytes = 64;
    alignas(fsverity_digest)
        std::array<unsigned char,
                   sizeof(fsverity_digest) + kMaximumDigestBytes> buffer{};
    auto* measurement = reinterpret_cast<fsverity_digest*>(buffer.data());
    measurement->digest_size = kMaximumDigestBytes;
    if (ioctl(impl_->handle, FS_IOC_MEASURE_VERITY, measurement) != 0 ||
        measurement->digest_size > kMaximumDigestBytes) {
      return Status::FailedPrecondition(
          "fs-verity enforcement changed during engine epoch");
    }
    const auto* digest_begin = reinterpret_cast<const std::byte*>(
        buffer.data() + sizeof(fsverity_digest));
    return validate_fsverity_sha256_measurement(
        enforced_digest_, measurement->digest_algorithm,
        {digest_begin, measurement->digest_size});
#endif
  }
  if (current_dm_receipt == nullptr) {
    return Status::FailedPrecondition(
        "dm-verity epoch poll requires a fresh supervisor receipt");
  }
  auto capacity = validate_dm_verity_supervisor_receipt(
      *current_dm_receipt, enforced_digest_, integrity_table_digest_,
      supervisor_attestation_digest_, impl_->identity.filesystem_identity,
      integrity_reserve_bytes_);
  if (!capacity.ok()) return capacity.status();
  if (capacity->total_integrity_bytes != integrity_owner_bytes_) {
    return Status::FailedPrecondition(
        "dm-verity integrity owner changed during engine epoch");
  }
  return Status::Ok();
}

Result<ArtifactWorkerDescriptor>
ControllerFileLease::duplicate_for_worker() const {
  auto duplicate = duplicate_impl(*impl_);
  if (!duplicate.ok()) return duplicate.status();
  return ArtifactWorkerDescriptor(std::move(*duplicate));
}

ArtifactWorkerDescriptor::~ArtifactWorkerDescriptor() = default;
ArtifactWorkerDescriptor::ArtifactWorkerDescriptor(
    std::unique_ptr<ArtifactDescriptorImpl> impl)
    : impl_(std::move(impl)) {}
ArtifactWorkerDescriptor::ArtifactWorkerDescriptor(
    ArtifactWorkerDescriptor&&) noexcept = default;
ArtifactWorkerDescriptor& ArtifactWorkerDescriptor::operator=(
    ArtifactWorkerDescriptor&&) noexcept = default;

const ArtifactFileIdentity& ArtifactWorkerDescriptor::identity() const noexcept {
  return impl_->identity;
}

Status ArtifactWorkerDescriptor::read_exact(
    std::uint64_t offset, std::span<std::byte> output) const {
  return read_from(*impl_, offset, output);
}

#ifdef __linux__
Result<ArtifactWorkerDescriptor>
ArtifactWorkerDescriptor::AdoptLinuxFileDescriptor(int* descriptor) {
  if (descriptor == nullptr || *descriptor < 0 ||
      ::fcntl(*descriptor, F_GETFD) < 0) {
    return Status::InvalidArgument(
        "Linux artifact descriptor adoption input is invalid");
  }
  auto impl = std::make_unique<ArtifactDescriptorImpl>();
  impl->handle = std::exchange(*descriptor, -1);
  auto identity = query_identity(*impl);
  if (!identity.ok()) return identity.status();
  impl->identity = *identity;
  return ArtifactWorkerDescriptor(std::move(impl));
}

int ArtifactWorkerDescriptor::linux_file_descriptor_for_transfer()
    const noexcept {
  return impl_ == nullptr ? -1 : impl_->handle;
}
#endif

Result<DescriptorMappedRange> DescriptorMappedRange::MapReadOnly(
    const ArtifactWorkerDescriptor& descriptor, std::uint64_t file_offset,
    std::uint64_t bytes) {
  if (descriptor.impl_ == nullptr || bytes == 0) {
    return Status::InvalidArgument("descriptor mapping request is invalid");
  }
  auto end = checked_add_u64(file_offset, bytes);
  if (!end.ok() || *end > descriptor.impl_->identity.file_bytes) {
    return Status::InvalidArgument("descriptor mapping range is out of bounds");
  }
  std::uint64_t alignment = 0;
#ifdef _WIN32
  SYSTEM_INFO system_info{};
  GetSystemInfo(&system_info);
  alignment = system_info.dwAllocationGranularity;
#else
  const auto page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0)
    return Status::Internal("cannot determine descriptor mapping page size");
  alignment = static_cast<std::uint64_t>(page_size);
#endif
  const auto aligned_offset = (file_offset / alignment) * alignment;
  const auto delta = file_offset - aligned_offset;
  auto mapping_bytes = checked_add_u64(delta, bytes);
  if (!mapping_bytes.ok() ||
      *mapping_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted(
        "descriptor mapping exceeds process address space");
  }
  auto impl = std::make_unique<Impl>();
  impl->mapping_bytes = static_cast<std::size_t>(*mapping_bytes);
#ifdef _WIN32
  impl->mapping = CreateFileMappingW(descriptor.impl_->handle, nullptr,
                                     PAGE_READONLY, 0, 0, nullptr);
  if (impl->mapping == nullptr)
    return Status::ResourceExhausted("cannot create descriptor file mapping");
  impl->mapping_base = MapViewOfFile(
      impl->mapping, FILE_MAP_READ, static_cast<DWORD>(aligned_offset >> 32U),
      static_cast<DWORD>(aligned_offset & 0xffffffffULL), impl->mapping_bytes);
  if (impl->mapping_base == nullptr)
    return Status::ResourceExhausted("cannot map descriptor file range");
#else
  impl->mapping_base = mmap(nullptr, impl->mapping_bytes, PROT_READ, MAP_SHARED,
                            descriptor.impl_->handle,
                            static_cast<off_t>(aligned_offset));
  if (impl->mapping_base == MAP_FAILED) {
    impl->mapping_base = nullptr;
    return Status::ResourceExhausted("cannot map descriptor file range");
  }
#ifdef MADV_NOHUGEPAGE
  if (madvise(impl->mapping_base, impl->mapping_bytes, MADV_NOHUGEPAGE) != 0) {
    return Status::FailedPrecondition(
        "descriptor mapping cannot enforce no-huge-page policy");
  }
#endif
#endif
  const auto* data = static_cast<const std::byte*>(impl->mapping_base) + delta;
  return DescriptorMappedRange(std::move(impl), data, file_offset, bytes);
}

DescriptorMappedRange::DescriptorMappedRange(
    std::unique_ptr<Impl> impl, const std::byte* data,
    std::uint64_t file_offset, std::uint64_t size_bytes)
    : impl_(std::move(impl)),
      data_(data),
      file_offset_(file_offset),
      size_bytes_(size_bytes) {}

DescriptorMappedRange::~DescriptorMappedRange() = default;
DescriptorMappedRange::DescriptorMappedRange(
    DescriptorMappedRange&& other) noexcept
    : impl_(std::move(other.impl_)),
      data_(std::exchange(other.data_, nullptr)),
      file_offset_(std::exchange(other.file_offset_, 0)),
      size_bytes_(std::exchange(other.size_bytes_, 0)) {}

DescriptorMappedRange& DescriptorMappedRange::operator=(
    DescriptorMappedRange&& other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
    data_ = std::exchange(other.data_, nullptr);
    file_offset_ = std::exchange(other.file_offset_, 0);
    size_bytes_ = std::exchange(other.size_bytes_, 0);
  }
  return *this;
}

Result<std::span<const std::byte>> DescriptorMappedRange::slice(
    std::uint64_t offset, std::uint64_t bytes) const {
  auto end = checked_add_u64(offset, bytes);
  if (!end.ok() || *end > size_bytes_ ||
      bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::InvalidArgument("descriptor mapped slice is out of bounds");
  }
  return std::span<const std::byte>(
      data_ + static_cast<std::size_t>(offset),
      static_cast<std::size_t>(bytes));
}

}  // namespace pih
