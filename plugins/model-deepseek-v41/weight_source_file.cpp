#include "weight_source_file.h"
#include "weight_catalog.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <new>
#include <optional>
#include <sys/stat.h>
#include <unistd.h>

namespace pih::deepseek_v41 {
namespace {
bool Same(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
      a.st_mode == b.st_mode && a.st_nlink == b.st_nlink && a.st_uid == b.st_uid && a.st_gid == b.st_gid &&
      a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
      a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
Status ReadExact(int fd, std::uint64_t offset, std::span<std::byte> bytes) {
  while (!bytes.empty()) {
    const auto count = ::pread(fd, bytes.data(), bytes.size(), static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return Status::Unavailable("V4.1 source shard read failed or truncated");
    offset += static_cast<std::size_t>(count);
    bytes = bytes.subspan(static_cast<std::size_t>(count));
  }
  return Status::Ok();
}
}
struct WeightSourceFile::Impl final {
  int directory = -1, file = -1;
  std::string member;
  std::uint64_t bytes = 0;
  struct stat identity{};
  std::optional<SafetensorsHeader> header;
  ~Impl() { if (file >= 0) ::close(file); if (directory >= 0) ::close(directory); }
  Status Check() const {
    struct stat held{}, named{};
    if (::fstat(file, &held) || ::fstatat(directory, member.c_str(), &named, AT_SYMLINK_NOFOLLOW) ||
        !Same(identity, held) || !Same(held, named))
      return Status::FailedPrecondition("V4.1 source shard changed or was replaced");
    return Status::Ok();
  }
};
WeightSourceFile::WeightSourceFile(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
WeightSourceFile::~WeightSourceFile() = default;
const SafetensorsHeader& WeightSourceFile::header() const noexcept { return *impl_->header; }
std::string_view WeightSourceFile::member_name() const noexcept { return impl_->member; }
Status WeightSourceFile::Revalidate() const { return impl_->Check(); }
Status WeightSourceFile::Read(std::uint64_t offset, std::span<std::byte> bytes) const {
  if (bytes.empty() || bytes.size() > 1024 * 1024 || offset > impl_->bytes || bytes.size() > impl_->bytes - offset)
    return Status::InvalidArgument("V4.1 source read outside admitted file or buffer bounds");
  auto status = impl_->Check(); if (!status.ok()) return status;
  status = ReadExact(impl_->file, offset, bytes); if (!status.ok()) return status;
  return impl_->Check();
}
Result<std::unique_ptr<WeightSourceFile>> WeightSourceFile::Open(int directory,
    std::string_view member, std::uint64_t bytes, const Sha256Digest& digest) {
  if (directory < 0 || !IsWeightShardMember(member) || bytes < 10 || bytes > (512ULL << 30) ||
      digest == Sha256Digest{})
    return Status::InvalidArgument("V4.1 source shard requires bounded size/name and trusted digest");
  try {
    auto impl = std::make_unique<Impl>();
    impl->directory = ::fcntl(directory, F_DUPFD_CLOEXEC, 3);
    struct stat root{};
    if (impl->directory < 0 || ::fstat(impl->directory, &root) || !S_ISDIR(root.st_mode))
      return Status::FailedPrecondition("V4.1 source parent is not an accessible directory");
    impl->member = member; impl->bytes = bytes;
    impl->file = ::openat(impl->directory, impl->member.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    if (impl->file < 0 || ::fstat(impl->file, &impl->identity) || !S_ISREG(impl->identity.st_mode) ||
        impl->identity.st_nlink != 1 || (impl->identity.st_mode & 0222) || impl->identity.st_size < 0 ||
        static_cast<std::uint64_t>(impl->identity.st_size) != bytes)
      return Status::FailedPrecondition("V4.1 source shard must be exact-size read-only single-link regular storage");
    auto status = impl->Check(); if (!status.ok()) return status;
    std::vector<std::byte> workspace(1024 * 1024);
    Sha256 hash;
    for (std::uint64_t offset = 0; offset < bytes;) {
      auto chunk = std::span(workspace).first(static_cast<std::size_t>(std::min<std::uint64_t>(workspace.size(), bytes - offset)));
      status = ReadExact(impl->file, offset, chunk); if (!status.ok()) return status;
      status = hash.update(chunk); if (!status.ok()) return status;
      offset += chunk.size();
    }
    auto observed = hash.finalize(); if (!observed.ok()) return observed.status();
    if (*observed != digest) return Status::FailedPrecondition("V4.1 source shard differs from trusted digest");
    status = impl->Check(); if (!status.ok()) return status;
    std::array<std::byte, 8> length{};
    status = ReadExact(impl->file, 0, length); if (!status.ok()) return status;
    std::uint64_t header_bytes = 0;
    for (unsigned i = 0; i < 8; ++i) header_bytes |= std::to_integer<std::uint64_t>(length[i]) << (8 * i);
    if (!header_bytes || header_bytes > SafetensorsHeader::kMaxHeaderBytes || header_bytes > bytes - 8)
      return Status::InvalidArgument("V4.1 source safetensors header exceeds bounds");
    std::vector<std::byte> prefix(static_cast<std::size_t>(header_bytes + 8));
    status = ReadExact(impl->file, 0, prefix); if (!status.ok()) return status;
    status = impl->Check(); if (!status.ok()) return status;
    auto parsed = SafetensorsHeader::ParsePrefix(prefix, bytes);
    if (!parsed.ok()) return parsed.status();
    impl->header.emplace(std::move(*parsed));
    return std::unique_ptr<WeightSourceFile>(new WeightSourceFile(std::move(impl)));
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 source shard admission allocation failed");
  }
}
Status ConvertWoASourceFiles(std::uint32_t layer,
    const WeightSourceFile& weights, std::string_view weight_name,
    const WeightSourceFile& scales, std::string_view scale_name,
    const WoATensorWriter& write) {
  try {
  auto binding = WoASourceBinding::Create(layer, weights.header(), weight_name, scales.header(), scale_name);
  if (!binding.ok()) return binding.status();
  auto status = weights.Revalidate(); if (!status.ok()) return status;
  status = scales.Revalidate(); if (!status.ok()) return status;
  status = binding->Convert(
      [&](std::uint64_t offset, std::span<std::byte> bytes) { return weights.Read(offset, bytes); },
      [&](std::uint64_t offset, std::span<std::byte> bytes) { return scales.Read(offset, bytes); }, write);
  if (!status.ok()) return status;
  status = weights.Revalidate(); if (!status.ok()) return status;
  return scales.Revalidate();
  } catch (const std::bad_alloc&) {
    return Status::ResourceExhausted("V4.1 source conversion allocation failed; discard unpublished output");
  } catch (...) {
    return Status::Internal("V4.1 source conversion failed; discard unpublished output");
  }
}
}  // namespace pih::deepseek_v41
