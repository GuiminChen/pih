#include "weight_files.h"
#include "weight_manifest.h"
#include "pih/model/safetensors_header.h"
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <new>
#include <optional>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace pih::deepseek_v41 {
namespace {
struct Fd {
  int value = -1;
  explicit Fd(int fd) : value(fd) {}
  ~Fd() { if (value >= 0) ::close(value); }
  Fd(const Fd&) = delete;
};
Result<int> Directory(const std::filesystem::path& path) {
  if (!path.is_absolute()) return Status::InvalidArgument("V4.1 weight directory must be absolute");
  Fd fd(::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (fd.value < 0) return Status::Unavailable("Cannot open filesystem root");
  for (const auto& part : path.relative_path()) {
    const auto name = part.string();
    if (name.empty() || name == "." || name == ".." || name.find('\0') != name.npos)
      return Status::InvalidArgument("Noncanonical V4.1 weight directory");
    Fd next(::openat(fd.value, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    if (next.value < 0) return Status::FailedPrecondition("Cannot open weight directory without links");
    std::swap(fd.value, next.value);
  }
  return std::exchange(fd.value, -1);
}
bool Same(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
      a.st_mode == b.st_mode && a.st_uid == b.st_uid && a.st_gid == b.st_gid &&
      a.st_nlink == b.st_nlink && a.st_mtim.tv_sec == b.st_mtim.tv_sec &&
      a.st_mtim.tv_nsec == b.st_mtim.tv_nsec && a.st_ctim.tv_sec == b.st_ctim.tv_sec &&
      a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
Status ReadAt(int fd, std::uint64_t offset, std::span<std::byte> out) {
  while (!out.empty()) {
    const auto count = ::pread(fd, out.data(), out.size(), static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return Status::Unavailable("V4.1 weight read failed or truncated");
    offset += static_cast<std::size_t>(count); out = out.subspan(static_cast<std::size_t>(count));
  }
  return Status::Ok();
}
}
struct BackboneWeightFiles::Impl {
  std::filesystem::path path;
  int root = -1;
  struct stat root_identity{};
  std::vector<int> descriptors;
  std::vector<struct stat> states;
  std::vector<ExpectedWeightShard> expected;
  std::optional<BackboneWeightCatalog> catalog;
  ~Impl() { for (const auto fd : descriptors) ::close(fd); if (root >= 0) ::close(root); }
  Status Check(std::size_t i) const {
    struct stat observed{}, named{};
    if (::fstat(descriptors[i], &observed) ||
        ::fstatat(root, expected[i].name.c_str(), &named, AT_SYMLINK_NOFOLLOW) ||
        !Same(states[i], observed) || !Same(observed, named))
      return Status::FailedPrecondition("V4.1 weight shard changed or was replaced");
    return Status::Ok();
  }
};
BackboneWeightFiles::BackboneWeightFiles(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
BackboneWeightFiles::~BackboneWeightFiles() = default;
Result<std::vector<std::byte>> ReadAuthenticatedMetadata(const std::filesystem::path& path,
    const Sha256Digest& expected, std::uint64_t maximum) {
  if (!path.is_absolute() || expected == Sha256Digest{} || !maximum || maximum > (16U << 20))
    return Status::InvalidArgument("Metadata path, digest or read bound invalid");
  try {
    const auto name = path.filename().string();
    if (name.empty() || name == "." || name == ".." || name.find('\0') != name.npos)
      return Status::InvalidArgument("Metadata member name invalid");
    auto root = Directory(path.parent_path()); if (!root.ok()) return root.status();
    Fd held(*root);
    Fd fd(::openat(*root, name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
    struct stat before{}, after{};
    if (fd.value < 0 || ::fstat(fd.value, &before) || !S_ISREG(before.st_mode) ||
        before.st_nlink != 1 || (before.st_mode & 0222) || before.st_size <= 0 ||
        static_cast<std::uint64_t>(before.st_size) > maximum)
      return Status::FailedPrecondition("Metadata must be bounded read-only single-link regular storage");
    std::vector<std::byte> bytes(static_cast<std::size_t>(before.st_size));
    std::size_t offset = 0;
    while (offset < bytes.size()) {
      const auto n = ::pread(fd.value, bytes.data() + offset, bytes.size() - offset, static_cast<off_t>(offset));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) return Status::Unavailable("Cannot read complete metadata snapshot");
      offset += static_cast<std::size_t>(n);
    }
    if (::fstat(fd.value, &after) || !Same(before, after)) return Status::FailedPrecondition("Metadata changed during read");
    auto digest = sha256(bytes); if (!digest.ok()) return digest.status();
    if (*digest != expected) return Status::FailedPrecondition("Metadata differs from admitted digest");
    return bytes;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Metadata snapshot allocation failed"); }
}
const BackboneWeightCatalog& BackboneWeightFiles::catalog() const noexcept { return *impl_->catalog; }
Result<std::unique_ptr<BackboneWeightFiles>> BackboneWeightFiles::OpenManifest(
    const std::filesystem::path& directory, const Sha256Digest& manifest_digest,
    const FlashConfig& config, std::uint32_t world, std::uint32_t rank, std::uint64_t budget) {
  try {
    auto root = Directory(directory); if (!root.ok()) return root.status();
    Fd held(*root);
    Fd fd(::openat(*root, "weights.manifest.json", O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
    struct stat before{}, after{};
    if (fd.value < 0 || ::fstat(fd.value, &before) || !S_ISREG(before.st_mode) ||
        before.st_nlink != 1 || (before.st_mode & 0222) || before.st_size <= 0 ||
        static_cast<std::uint64_t>(before.st_size) > kWeightManifestMaximumBytes)
      return Status::FailedPrecondition("Weight manifest must be a bounded read-only single-link regular member");
    std::string json(static_cast<std::size_t>(before.st_size), '\0');
    std::size_t offset = 0;
    while (offset < json.size()) {
      const auto n = ::pread(fd.value, json.data() + offset, json.size() - offset, static_cast<off_t>(offset));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) return Status::Unavailable("Cannot read complete weight manifest");
      offset += static_cast<std::size_t>(n);
    }
    if (::fstat(fd.value, &after) || !Same(before, after))
      return Status::FailedPrecondition("Weight manifest changed during admission");
    const auto expected = ParseWeightManifest(json, manifest_digest, config, world, rank);
    if (!expected.ok()) return expected.status();
    return Open(directory, config, world, rank, *expected, budget);
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Weight manifest read allocation failed"); }
}
Status BackboneWeightFiles::Revalidate() const {
  for (std::size_t i = 0; i < impl_->descriptors.size(); ++i) {
    const auto status = impl_->Check(i); if (!status.ok()) return status;
  }
  auto root = Directory(impl_->path); if (!root.ok()) return root.status();
  Fd fd(*root); struct stat observed{};
  if (::fstat(fd.value, &observed) || observed.st_dev != impl_->root_identity.st_dev ||
      observed.st_ino != impl_->root_identity.st_ino)
    return Status::FailedPrecondition("V4.1 weight directory replaced");
  return Status::Ok();
}
Result<std::unique_ptr<BackboneWeightFiles>> BackboneWeightFiles::Open(
    const std::filesystem::path& directory, const FlashConfig& config,
    std::uint32_t world, std::uint32_t rank, std::span<const ExpectedWeightShard> expected,
    std::uint64_t budget) {
  if (expected.empty() || expected.size() > BackboneWeightCatalog::kMaximumShards)
    return Status::InvalidArgument("V4.1 expected shard count invalid");
  auto inventory = BackboneWeightInventory::Create(config, world, rank);
  if (!inventory.ok()) return inventory.status();
  if (inventory->bytes() > budget) return Status::ResourceExhausted("V4.1 weights exceed device budget");
  try {
    auto impl = std::make_unique<Impl>(); impl->path = directory;
    impl->expected.assign(expected.begin(), expected.end());
    impl->descriptors.reserve(expected.size()); impl->states.reserve(expected.size());
    std::uint64_t total = 0;
    const auto maximum = inventory->bytes() + BackboneWeightCatalog::kMaximumHeaderBytes;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const auto& e = expected[i];
      if (!IsWeightShardMember(e.name) || e.bytes < 8 || e.bytes > maximum - total ||
          e.bytes > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) || e.sha256 == Sha256Digest{})
        return Status::InvalidArgument("V4.1 expected shard identity or size invalid");
      for (std::size_t j = 0; j < i; ++j) if (expected[j].name == e.name)
        return Status::InvalidArgument("Duplicate expected V4.1 shard");
      total += e.bytes;
    }
    auto root = Directory(directory); if (!root.ok()) return root.status();
    impl->root = *root;
    if (::fstat(*root, &impl->root_identity)) return Status::Unavailable("Cannot inspect weight directory");
    std::vector<std::vector<std::byte>> headers; headers.reserve(expected.size());
    std::vector<WeightShardPrefix> prefixes; prefixes.reserve(expected.size());
    std::vector<std::byte> buffer(1U << 20);
    std::uint64_t header_total = 0;
    for (const auto& e : expected) {
      Fd fd(::openat(*root, e.name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
      struct stat state{};
      if (fd.value < 0 || ::fstat(fd.value, &state) || !S_ISREG(state.st_mode) ||
          state.st_size < 0 || static_cast<std::uint64_t>(state.st_size) != e.bytes ||
          (state.st_mode & 0222) || state.st_nlink != 1)
        return Status::FailedPrecondition("Weight shard must be sealed regular file with one link and exact size");
      for (const auto& old : impl->states) if (old.st_dev == state.st_dev && old.st_ino == state.st_ino)
        return Status::InvalidArgument("V4.1 shards alias the same inode");
      impl->descriptors.push_back(std::exchange(fd.value, -1)); impl->states.push_back(state);
      const auto index = impl->descriptors.size() - 1;
      const auto descriptor = impl->descriptors.back();
      Sha256 hash;
      for (std::uint64_t offset = 0; offset < e.bytes;) {
        auto chunk = std::span(buffer).first(static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), e.bytes - offset)));
        auto status = ReadAt(descriptor, offset, chunk); if (!status.ok()) return status;
        status = hash.update(chunk); if (!status.ok()) return status;
        offset += chunk.size();
      }
      auto digest = hash.finalize(); if (!digest.ok()) return digest.status();
      if (*digest != e.sha256) return Status::FailedPrecondition("V4.1 weight shard SHA256 mismatch");
      std::array<std::byte, 8> length{};
      auto status = ReadAt(descriptor, 0, length); if (!status.ok()) return status;
      std::uint64_t size = 0;
      for (unsigned i = 0; i < 8; ++i) size |= std::uint64_t(std::to_integer<unsigned char>(length[i])) << (8 * i);
      if (!size || size > SafetensorsHeader::kMaxHeaderBytes || size > e.bytes - 8 ||
          size + 8 > BackboneWeightCatalog::kMaximumHeaderBytes - header_total)
        return Status::InvalidArgument("V4.1 weight header exceeds bounds");
      header_total += size + 8;
      headers.emplace_back(static_cast<std::size_t>(size + 8));
      status = ReadAt(descriptor, 0, headers.back()); if (!status.ok()) return status;
      status = impl->Check(index); if (!status.ok()) return status;
      prefixes.push_back({e.name, e.bytes, headers.back()});
    }
    auto catalog = BackboneWeightCatalog::Create(config, world, rank, prefixes, budget);
    if (!catalog.ok()) return catalog.status();
    impl->catalog.emplace(std::move(*catalog));
    auto result = std::unique_ptr<BackboneWeightFiles>(new BackboneWeightFiles(std::move(impl)));
    const auto stable = result->Revalidate(); if (!stable.ok()) return stable;
    return result;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("V4.1 weight file admission allocation failed"); }
}
Status BackboneWeightFiles::Read(std::string_view name, std::uint64_t offset, std::span<std::byte> out) const {
  const auto* weight = catalog().Find(name);
  if (!weight || out.empty() || out.size() > (1U << 20) || offset > weight->weight.tensor.bytes ||
      out.size() > weight->weight.tensor.bytes - offset)
    return Status::InvalidArgument("V4.1 tensor staging read outside bounds");
  auto status = impl_->Check(weight->shard); if (!status.ok()) return status;
  status = ReadAt(impl_->descriptors[weight->shard], weight->file_offset + offset, out);
  if (!status.ok()) return status;
  return impl_->Check(weight->shard);
}
}  // namespace pih::deepseek_v41
