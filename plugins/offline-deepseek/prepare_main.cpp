#include "prepare_generation.h"
#include "generation_verify.h"
#include <iostream>
#include <stdexcept>
#include <utility>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
template<class T> T Require(pih::Result<T> value) {
  if (!value.ok()) throw std::runtime_error(std::string(value.status().message()));
  return std::move(*value);
}
void Require(pih::Status value) {
  if (!value.ok()) throw std::runtime_error(std::string(value.message()));
}
struct Fd final {
  int value;
  explicit Fd(int fd) : value(fd) { if (fd < 0) throw std::runtime_error("cannot open staging without links"); }
  Fd(Fd&& other) noexcept : value(std::exchange(other.value, -1)) {}
  ~Fd() { if (value >= 0) ::close(value); }
};
Fd OpenStaging(const std::filesystem::path& path) {
  if (!path.is_absolute()) throw std::runtime_error("staging path must be absolute");
  Fd fd(::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  for (const auto& part : path.relative_path()) {
    const auto name = part.string();
    if (name.empty() || name == "." || name == ".." || name.find('\0') != name.npos)
      throw std::runtime_error("staging path must be canonical");
    Fd next(::openat(fd.value, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    std::swap(fd.value, next.value);
  }
  return fd;
}
bool Contains(const std::filesystem::path& parent, const std::filesystem::path& child) {
  auto a = parent.begin(), b = child.begin();
  for (; a != parent.end() && b != child.end(); ++a, ++b) if (*a != *b) return false;
  return a == parent.end();
}
}  // namespace
int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: pih-deepseek-artifact-prepare SOURCE_DIRECTORY EMPTY_STAGING_DIRECTORY "
        "MODEL_ROOT SEMANTIC_ROOT INVENTORY_ROOT PAYLOAD_ROOT CONVERTER_IDENTITY_ROOT\n";
    return 2;
  }
  try {
    const std::filesystem::path source_path(argv[1]), staging_path(argv[2]);
    if (!source_path.is_absolute() || !staging_path.is_absolute() ||
        Contains(source_path.lexically_normal(), staging_path.lexically_normal()) ||
        Contains(staging_path.lexically_normal(), source_path.lexically_normal()))
      throw std::runtime_error("source and staging must be disjoint absolute directories");
    pih::offline_deepseek::SourcePreparationAuthority authority{
        Require(pih::Sha256Digest::ParseHex(argv[3])), Require(pih::Sha256Digest::ParseHex(argv[4])),
        Require(pih::Sha256Digest::ParseHex(argv[5])), Require(pih::Sha256Digest::ParseHex(argv[6])),
        Require(pih::Sha256Digest::ParseHex(argv[7]))};
    for (const auto& digest : {authority.model_digest, authority.semantic_root, authority.inventory_root,
                              authority.expected_payload_root, authority.converter_identity_root})
      if (digest == pih::Sha256Digest{}) throw std::runtime_error("authority roots must be nonzero");
    auto staging = OpenStaging(staging_path);
    Require(pih::offline_deepseek::ValidatePreparationStaging(staging.value));
    struct stat identity{};
    if (::fstat(staging.value, &identity) != 0) throw std::runtime_error("cannot inspect staging identity");
    std::cerr << "admitting source files; this hashes the complete checkpoint\n";
    auto source = Require(pih::offline_deepseek::SourceArtifact::Open(source_path, authority.model_digest));
    std::cerr << "compiling source metadata and copying PP1 generation; failures retain staging files\n";
    auto copied = Require(pih::offline_deepseek::PreparePp1Generation(staging.value, *source, authority));
    std::cerr << "independently rebuilding source conversion and verifying target payloads\n";
    auto observed = Require(pih::offline_deepseek::VerifySourceBoundGeneration(*source, authority,
        staging_path, copied.manifest.artifact_root,
        [](std::string_view name, std::uint64_t bytes) {
          std::cerr << "verifying " << name << " (" << bytes << " bytes)\n";
        }));
    if (observed.directory_device != static_cast<std::uint64_t>(identity.st_dev) ||
        observed.directory_inode != static_cast<std::uint64_t>(identity.st_ino))
      throw std::runtime_error("staging identity changed before final verification");
    Require(source->Revalidate());
    std::cout << "{\"schema\":\"pih.deepseek_pp1_preparation_observation.v1\",\"artifact_root\":\""
        << copied.manifest.artifact_root.hex() << "\",\"tensor_count\":" << observed.tensor_count
        << ",\"tensor_bytes\":" << observed.tensor_bytes << ",\"shard_count\":" << observed.shard_count
        << ",\"source_payload_equivalence\":" << (observed.source_payload_equivalence ? "true" : "false")
        << ",\"published\":false,\"immutable_admission\":false,\"model_execution\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native preparation failed: " << error.what()
        << "; any created staging files are retained; no generation was published\n";
    return 2;
  }
}
