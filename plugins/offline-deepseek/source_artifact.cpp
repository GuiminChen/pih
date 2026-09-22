#include "source_artifact.h"
#include "tensor_inventory.h"
#include "pih/core/canonical_hash.h"
#include "pih/model/deepseek_v4_config.h"
#include "pih/model/safetensors_shard_index.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <set>
#include <utility>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pih::offline_deepseek {
namespace {
struct DirectoryFd final {
  int value;
  explicit DirectoryFd(int fd) : value(fd) {}
  DirectoryFd(const DirectoryFd&) = delete;
  DirectoryFd& operator=(const DirectoryFd&) = delete;
  ~DirectoryFd() { if (value >= 0) ::close(value); }
};
Result<int> OpenDirectory(const std::filesystem::path& path) {
  if (!path.is_absolute()) return Status::InvalidArgument("source artifact path must be absolute");
  DirectoryFd fd(::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (fd.value < 0) return Status::Unavailable("cannot open source root");
  for (const auto& part : path.relative_path()) {
    const auto name = part.string();
    if (name.empty() || name == "." || name == ".." || name.find('\0') != name.npos) {
      return Status::InvalidArgument("source artifact path is not canonical");
    }
    DirectoryFd next(::openat(fd.value, name.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    if (next.value < 0) return Status::FailedPrecondition("cannot open source directory without links");
    std::swap(fd.value, next.value);
  }
  return std::exchange(fd.value, -1);
}
bool Same(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode &&
      a.st_size == b.st_size && a.st_mtim.tv_sec == b.st_mtim.tv_sec &&
      a.st_mtim.tv_nsec == b.st_mtim.tv_nsec && a.st_ctim.tv_sec == b.st_ctim.tv_sec &&
      a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
Status Text(CanonicalHashBuilder& hash, std::uint16_t id, std::string_view value) {
  return hash.add_bytes(id, std::as_bytes(std::span(value)));
}
}  // namespace
struct SourceArtifact::Impl final {
  std::filesystem::path path;
  int root = -1;
  struct stat identity{};
  std::vector<int> descriptors;
  std::vector<struct stat> states;
  std::vector<SourceArtifactObject> objects;
  std::vector<SafetensorsShardBinding> bindings;
  Sha256Digest model_digest;
  ~Impl() { for (const auto fd : descriptors) ::close(fd); if (root >= 0) ::close(root); }
};
SourceArtifact::SourceArtifact(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SourceArtifact::~SourceArtifact() = default;
const std::vector<int>& SourceArtifact::descriptors() const noexcept { return impl_->descriptors; }
const std::vector<SourceArtifactObject>& SourceArtifact::objects() const noexcept { return impl_->objects; }
const Sha256Digest& SourceArtifact::model_digest() const noexcept { return impl_->model_digest; }
Status SourceArtifact::Revalidate() const {
  for (std::size_t i = 0; i < impl_->descriptors.size(); ++i) {
    struct stat observed{}, named{};
    if (::fstat(impl_->descriptors[i], &observed) != 0 ||
        ::fstatat(impl_->root, impl_->objects[i].name.c_str(), &named, AT_SYMLINK_NOFOLLOW) != 0 ||
        !Same(impl_->states[i], observed) || !Same(observed, named))
      return Status::FailedPrecondition("source artifact member changed or was replaced");
  }
  auto root = OpenDirectory(impl_->path);
  if (!root.ok()) return root.status();
  struct stat observed{};
  const auto inspected = ::fstat(*root, &observed);
  ::close(*root);
  if (inspected != 0 || observed.st_dev != impl_->identity.st_dev || observed.st_ino != impl_->identity.st_ino)
    return Status::FailedPrecondition("source artifact directory was replaced");
  return Status::Ok();
}
Result<std::unique_ptr<SourceArtifact>> SourceArtifact::Open(
    const std::filesystem::path& directory, const Sha256Digest& expected_model_digest) {
  if (expected_model_digest == Sha256Digest{})
    return Status::InvalidArgument("expected source model digest missing");
  auto impl = std::make_unique<Impl>();
  impl->path = directory;
  impl->descriptors.reserve(50); impl->objects.reserve(50); impl->states.reserve(50);
  auto root = OpenDirectory(directory);
  if (!root.ok()) return root.status();
  impl->root = *root;
  if (::fstat(*root, &impl->identity) != 0) return Status::Unavailable("cannot inspect source directory");
  std::vector<std::string> names{"config.json", "model.safetensors.index.json"};
  for (unsigned i = 1; i <= 48; ++i) {
    const auto digits = std::to_string(i);
    names.push_back("model-" + std::string(5 - digits.size(), '0') + digits + "-of-00048.safetensors");
  }
  std::set<std::pair<dev_t, ino_t>> identities;
  std::vector<std::byte> buffer(1U << 20);
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < names.size(); ++i) {
    const auto fd = ::openat(impl->root, names[i].c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) return Status::FailedPrecondition("cannot open source artifact member");
    impl->descriptors.push_back(fd);
    struct stat state{};
    const auto maximum = i == 0 ? 1ULL << 20 : i == 1 ? 64ULL << 20 : 16ULL << 30;
    if (::fstat(fd, &state) != 0 || !S_ISREG(state.st_mode) || state.st_size <= 0 ||
        static_cast<std::uint64_t>(state.st_size) > maximum ||
        !identities.emplace(state.st_dev, state.st_ino).second)
      return Status::InvalidArgument("source artifact member bounds or identity invalid");
    const auto size = static_cast<std::uint64_t>(state.st_size);
    if (size > (256ULL << 30) - total) return Status::InvalidArgument("source artifact total exceeds budget");
    total += size;
    Sha256 hash;
    std::string control;
    if (i < 2) control.reserve(static_cast<std::size_t>(size));
    for (std::uint64_t offset = 0; offset < size;) {
      const auto wanted = static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), size - offset));
      const auto count = ::pread(fd, buffer.data(), wanted, static_cast<off_t>(offset));
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) return Status::Unavailable("source artifact read failed or truncated");
      auto chunk = std::span(buffer).first(static_cast<std::size_t>(count));
      auto status = hash.update(chunk);
      if (!status.ok()) return status;
      if (i < 2) control.append(reinterpret_cast<const char*>(chunk.data()), chunk.size());
      offset += chunk.size();
    }
    if (i == 0) {
      auto config = DeepSeekV4Config::ParseFlash0731(control);
      if (!config.ok()) return config.status();
    } else if (i == 1) {
      auto index = SafetensorsShardIndex::Parse(control);
      if (!index.ok()) return index.status();
      auto expected = ExpectedTensorNames(true);
      if (!expected.ok()) return expected.status();
      if (index->total_size() != 166'878'536'440ULL || index->bindings().size() != expected->size() ||
          index->shard_names() != std::vector<std::string>(names.begin() + 2, names.end()))
        return Status::InvalidArgument("source index differs from complete frozen inventory");
      for (std::size_t j = 0; j < expected->size(); ++j)
        if (index->bindings()[j].tensor_name != (*expected)[j])
          return Status::InvalidArgument("source index tensor name differs from frozen inventory");
      impl->bindings = index->bindings();
    }
    auto digest = hash.finalize();
    if (!digest.ok()) return digest.status();
    auto object = CanonicalHashBuilder::Create("pih:deepseek-artifact-object:v1", 4);
    if (!object.ok()) return object.status();
    auto status = Text(*object, 1, i == 0 ? "model_config" : i == 1 ? "weight_index" : "weight_shard");
    if (status.ok()) status = Text(*object, 2, names[i]);
    if (status.ok()) status = object->add_u64(3, size);
    if (status.ok()) status = object->add_hash(4, *digest);
    if (!status.ok()) return status;
    auto object_root = object->finalize();
    if (!object_root.ok()) return object_root.status();
    impl->objects.push_back({names[i], size, *digest, *object_root});
    impl->states.push_back(state);
  }
  auto closure = CanonicalHashBuilder::Create("pih:deepseek-artifact-closure:v1", 56);
  if (!closure.ok()) return closure.status();
  auto status = Text(*closure, 1, "deepseek_artifact_closure_v1");
  if (status.ok()) status = Text(*closure, 2, "deepseek_v4_flash_0731");
  if (status.ok()) status = closure->add_hash(3, impl->objects[0].object_root);
  if (status.ok()) status = closure->add_hash(4, impl->objects[1].object_root);
  if (status.ok()) status = closure->add_u32(5, 48);
  if (status.ok()) status = closure->add_u64(6, 72'317);
  if (status.ok()) status = closure->add_u64(7, 166'878'536'440ULL);
  if (status.ok()) status = closure->add_u64(8, total);
  for (std::size_t i = 0; status.ok() && i < 48; ++i)
    status = closure->add_hash(static_cast<std::uint16_t>(100 + i), impl->objects[i + 2].object_root);
  if (!status.ok()) return status;
  auto digest = closure->finalize();
  if (!digest.ok()) return digest.status();
  if (*digest != expected_model_digest) return Status::FailedPrecondition("source model digest differs from trusted expectation");
  impl->model_digest = *digest;
  auto result = std::unique_ptr<SourceArtifact>(new SourceArtifact(std::move(impl)));
  status = result->Revalidate();
  if (!status.ok()) return status;
  return result;
}
Result<std::vector<SafetensorsHeader>> SourceArtifact::ReadShardHeaders() const {
  auto status = Revalidate();
  if (!status.ok()) return status;
  auto read = [](int fd, std::span<std::byte> output) -> Status {
    std::uint64_t offset = 0;
    while (!output.empty()) {
      const auto count = ::pread(fd, output.data(), output.size(), static_cast<off_t>(offset));
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) return Status::Unavailable("source header read failed or truncated");
      offset += static_cast<std::size_t>(count);
      output = output.subspan(static_cast<std::size_t>(count));
    }
    return Status::Ok();
  };
  std::vector<SafetensorsHeader> headers;
  headers.reserve(48);
  std::vector<bool> seen(impl_->bindings.size(), false);
  std::uint64_t payload_bytes = 0;
  std::size_t tensor_count = 0;
  for (std::size_t i = 2; i < impl_->objects.size(); ++i) {
    std::array<std::byte, 8> length{};
    status = read(impl_->descriptors[i], length);
    if (!status.ok()) return status;
    std::uint64_t bytes = 0;
    for (unsigned b = 0; b < 8; ++b)
      bytes |= static_cast<std::uint64_t>(std::to_integer<unsigned>(length[b])) << (8 * b);
    if (impl_->objects[i].bytes < 8 || bytes == 0 || bytes > SafetensorsHeader::kMaxHeaderBytes ||
        bytes > impl_->objects[i].bytes - 8)
      return Status::InvalidArgument("source header length outside admitted object");
    std::vector<std::byte> prefix(static_cast<std::size_t>(bytes + 8));
    status = read(impl_->descriptors[i], prefix);
    if (!status.ok()) return status;
    auto header = SafetensorsHeader::ParsePrefix(prefix, impl_->objects[i].bytes);
    if (!header.ok()) return header.status();
    for (const auto& tensor : header->tensors()) {
      auto binding = std::lower_bound(impl_->bindings.begin(), impl_->bindings.end(), tensor.name,
          [](const auto& entry, const auto& name) { return entry.tensor_name < name; });
      if (binding == impl_->bindings.end() || binding->tensor_name != tensor.name ||
          binding->shard_name != impl_->objects[i].name)
        return Status::InvalidArgument("actual source tensor name/shard differs from admitted index");
      const auto ordinal = static_cast<std::size_t>(binding - impl_->bindings.begin());
      if (seen[ordinal]) return Status::InvalidArgument("source header tensor occurs more than once");
      seen[ordinal] = true;
      if (tensor.shape.empty() || tensor.file_end <= tensor.file_begin)
        return Status::InvalidArgument("source tensor is empty or scalar outside frozen geometry");
      status = ValidateSourceTensorGeometry(tensor.name, tensor.dtype, tensor.shape);
      if (!status.ok()) return status;
    }
    payload_bytes += header->data_bytes();
    tensor_count += header->tensors().size();
    headers.push_back(std::move(*header));
  }
  if (tensor_count != 72'317 || payload_bytes != 166'878'536'440ULL ||
      std::find(seen.begin(), seen.end(), false) != seen.end())
    return Status::InvalidArgument("source actual headers do not close admitted index ledger");
  status = Revalidate();
  if (!status.ok()) return status;
  return headers;
}
}  // namespace pih::offline_deepseek
