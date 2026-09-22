#include "deepseek_semantic_artifact.h"
#include "pih/core/bounded_json.h"
#include "pih/core/sha256.h"
#include <array>
#include <cerrno>
#include <stdexcept>
#include <utility>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pih::plugin_text {
namespace {
struct Object { std::string_view role, path; size_t limit; };
constexpr std::array<Object, 15> objects{{
    {"model_config", "config.json", 64 << 10},
    {"generation_config", "generation_config.json", 64 << 10},
    {"tokenizer_json", "tokenizer.json", 64 << 20},
    {"tokenizer_config", "tokenizer_config.json", 1 << 20},
    {"encoding_readme", "encoding/README.md", 1 << 20},
    {"encoding_source", "encoding/encoding_dsv4.py", 2 << 20},
    {"encoding_test_source", "encoding/test_encoding_dsv4.py", 1 << 20},
    {"encoding_input_1", "encoding/tests/test_input_1.json", 1 << 20},
    {"encoding_input_2", "encoding/tests/test_input_2.json", 1 << 20},
    {"encoding_input_3", "encoding/tests/test_input_3.json", 1 << 20},
    {"encoding_input_4", "encoding/tests/test_input_4.json", 1 << 20},
    {"encoding_output_1", "encoding/tests/test_output_1.txt", 1 << 20},
    {"encoding_output_2", "encoding/tests/test_output_2.txt", 1 << 20},
    {"encoding_output_3", "encoding/tests/test_output_3.txt", 1 << 20},
    {"encoding_output_4", "encoding/tests/test_output_4.txt", 1 << 20}}};
template<class T> T Require(Result<T> value) {
  if (!value.ok()) throw std::invalid_argument(std::string(value.status().message()));
  return std::move(*value);
}
void Hash(Sha256& hash, std::span<const std::byte> bytes) {
  auto status = hash.update(bytes);
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
void Hash(Sha256& hash, std::string_view text) {
  Hash(hash, std::as_bytes(std::span(text.data(), text.size())));
}
void BigEndian(Sha256& hash, uint64_t value, size_t width) {
  std::array<std::byte, 8> bytes{};
  for (size_t i = 0; i < width; ++i) bytes[width - 1 - i] = static_cast<std::byte>((value >> (8 * i)) & 255);
  Hash(hash, std::span<const std::byte>(bytes.data(), width));
}
const JsonValue& Field(const JsonValue& object, std::string_view name) {
  const auto* value = object.at(name);
  if (!value) throw std::invalid_argument("semantic configuration field missing");
  return *value;
}
#if defined(__linux__)
class Descriptor final {
 public:
  explicit Descriptor(int fd) : fd_(fd) { if (fd < 0) throw std::invalid_argument("cannot open semantic artifact without symlinks"); }
  ~Descriptor() { if (fd_ >= 0) ::close(fd_); }
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  Descriptor(Descriptor&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  Descriptor& operator=(Descriptor&& other) noexcept {
    if (this != &other) { if (fd_ >= 0) ::close(fd_); fd_ = std::exchange(other.fd_, -1); }
    return *this;
  }
  int get() const { return fd_; }
 private:
  int fd_;
};
Descriptor OpenRoot(const std::filesystem::path& path) {
  const auto absolute = std::filesystem::absolute(path).lexically_normal();
  if (absolute.native().find('\0') != std::string::npos) throw std::invalid_argument("NUL in semantic artifact path");
  Descriptor directory(::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  for (const auto& part : absolute.relative_path()) {
    if (part.empty() || part == ".") continue;
    if (part == "..") throw std::invalid_argument("parent traversal in semantic artifact path");
    directory = Descriptor(::openat(directory.get(), part.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  }
  return directory;
}
std::string ReadObject(int root, const Object& object) {
  Descriptor directory(::fcntl(root, F_DUPFD_CLOEXEC, 0));
  const std::filesystem::path relative(object.path);
  for (const auto& part : relative.parent_path())
    directory = Descriptor(::openat(directory.get(), part.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  Descriptor file(::openat(directory.get(), relative.filename().c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
  struct stat info{};
  if (::fstat(file.get(), &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0 ||
      static_cast<uint64_t>(info.st_size) > object.limit)
    throw std::invalid_argument("semantic artifact must be a bounded regular file");
  std::string bytes(static_cast<size_t>(info.st_size), '\0');
  size_t consumed = 0;
  while (consumed < bytes.size()) {
    const auto count = ::read(file.get(), bytes.data() + consumed, bytes.size() - consumed);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) throw std::invalid_argument("semantic artifact short read");
    consumed += static_cast<size_t>(count);
  }
  char extra; ssize_t count;
  do { count = ::read(file.get(), &extra, 1); } while (count < 0 && errno == EINTR);
  if (count != 0) throw std::invalid_argument("semantic artifact changed during read");
  // Subsequent use is exclusively this copy, authenticated below. No pathname
  // reopen or post-hash mutable mapping can replace the admitted tokenizer.
  return bytes;
}
#endif
}

DeepSeekSemanticArtifacts DeepSeekSemanticArtifacts::Load(const std::filesystem::path& snapshot_root) {
#if !defined(__linux__)
  (void)snapshot_root;
  throw std::invalid_argument("native semantic artifact admission requires Linux descriptor semantics");
#else
  auto directory = OpenRoot(snapshot_root);
  Sha256 root;
  constexpr char domain[] = "pih.runtime_semantic_sidecar.deepseek_v4_0731.v1";
  Hash(root, std::string_view(domain, sizeof(domain)));  // includes domain NUL
  std::array<std::byte, 20> revision{};
  auto nibble = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
  for (size_t i = 0; i < revision.size(); ++i)
    revision[i] = static_cast<std::byte>((nibble(kRevision[i * 2]) << 4) | nibble(kRevision[i * 2 + 1]));
  Hash(root, revision);
  std::string tokenizer_bytes, config_bytes, generation_bytes, tokenizer_config_bytes;
  size_t total = 0;
  for (const auto& object : objects) {
    auto bytes = ReadObject(directory.get(), object);
    if (bytes.size() > (80 << 20) - total) throw std::invalid_argument("semantic closure exceeds 80 MiB");
    total += bytes.size();
    const auto digest = Require(sha256(std::as_bytes(std::span(bytes.data(), bytes.size()))));
    BigEndian(root, object.role.size(), 2); Hash(root, object.role);
    BigEndian(root, bytes.size(), 8); Hash(root, digest.bytes);
    if (object.role == "tokenizer_json") tokenizer_bytes = std::move(bytes);
    else if (object.role == "model_config") config_bytes = std::move(bytes);
    else if (object.role == "generation_config") generation_bytes = std::move(bytes);
    else if (object.role == "tokenizer_config") tokenizer_config_bytes = std::move(bytes);
  }
  if (Require(root.finalize()).hex() != kClosureRoot)
    throw std::invalid_argument("DeepSeek semantic closure differs from pinned revision");
  const auto config = Require(JsonValue::Parse(config_bytes));
  const auto generation = Require(JsonValue::Parse(generation_bytes));
  const auto tokenizer_config = Require(JsonValue::Parse(tokenizer_config_bytes));
  if (Field(config, "model_type").string() != "deepseek_v4" ||
      Field(config, "vocab_size").integer() != 129280 || Field(generation, "bos_token_id").integer() != 0 ||
      Field(generation, "eos_token_id").integer() != 1)
    throw std::invalid_argument("semantic model or BOS/EOS identity mismatch");
  const auto* chat_template = tokenizer_config.at("chat_template");
  if (chat_template && !chat_template->is_null() &&
      !(chat_template->is_string() && chat_template->string().empty()))
    throw std::invalid_argument("DeepSeek semantic artifacts must not use a Jinja template");
  return DeepSeekSemanticArtifacts(Tokenizer::FromJson(tokenizer_bytes, TokenizerFamily::DeepSeekV4Flash0731));
#endif
}
}
