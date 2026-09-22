#include "token_map_builder.h"
#include "weight_files.h"
#include "../common/bytelevel_tokenizer.h"
#include "pih/core/canonical_json.h"
#include <unicode/uversion.h>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
using namespace pih;
using namespace pih::deepseek_v41;
struct Fd { int value = -1; ~Fd() { if (value >= 0) ::close(value); } };
void Require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class T> T Take(Result<T> result) {
  if (!result.ok()) throw std::runtime_error(std::string(result.status().message()));
  return std::move(*result);
}
int Directory(const std::filesystem::path& path) {
  Require(path.is_absolute() && path.string().size() <= 4096, "directory must be a bounded absolute path");
  Fd current{::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)};
  Require(current.value >= 0, "cannot open filesystem root");
  for (const auto& part : path.relative_path()) {
    const auto name = part.string();
    Require(!name.empty() && name != "." && name != "..", "directory path must be canonical");
    Fd next{::openat(current.value, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)};
    Require(next.value >= 0, "cannot traverse directory without symlinks");
    std::swap(current.value, next.value);
  }
  const int result = current.value; current.value = -1; return result;
}
void Absent(int root, const std::string& name) {
  struct stat st{};
  const int result = ::fstatat(root, name.c_str(), &st, AT_SYMLINK_NOFOLLOW);
  Require(result < 0 && errno == ENOENT, "output or staging exists or cannot be inspected");
}
void Usage() {
  std::fputs("Usage: pih-v41-token-map TOKENIZER_JSON TRUSTED_MAP_SHA256 OUTPUT_DIRECTORY\n"
      "Linux CPU/ICU tool; absolute paths, frozen tokenizer and independent map digest required.\n", stderr);
}
}
int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") { Usage(); return 0; }
  if (argc != 4) { Usage(); return 2; }
  bool published = false;
  try {
    Require(std::signal(SIGPIPE, SIG_IGN) != SIG_ERR, "cannot configure receipt pipe handling");
    const auto expected = Take(Sha256Digest::ParseHex(argv[2]));
    Require(expected != Sha256Digest{} && expected.hex() == argv[2], "map digest must be nonzero lowercase trusted SHA-256");
    const std::filesystem::path output(argv[3]);
    Require(output.is_absolute() && output.string().size() <= 4096, "output directory must be absolute and bounded");
    const auto name = output.filename().string(), staging = name + ".staging";
    Require(!name.empty() && name != "." && name != ".." && name.size() <= 200, "invalid output directory member");
    Fd parent{Directory(output.parent_path())};
    struct stat parent_identity{};
    Require(::fstat(parent.value, &parent_identity) == 0 && parent_identity.st_uid == ::geteuid() &&
        !(parent_identity.st_mode & 0022), "output parent must be owned and not group/other writable");
    Absent(parent.value, name); Absent(parent.value, staging);
    const auto tokenizer_digest = Take(Sha256Digest::ParseHex(plugin_text::kV41TokenizerSha256));
    const auto tokenizer = Take(ReadAuthenticatedMetadata(argv[1], tokenizer_digest, 16 * 1024 * 1024));
    const auto map = Take(BuildEngramTokenMap(std::string_view(
        reinterpret_cast<const char*>(tokenizer.data()), tokenizer.size()), expected));
    Require(::mkdirat(parent.value, staging.c_str(), 0700) == 0, "cannot exclusively create staging directory");
    Fd directory{::openat(parent.value, staging.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)};
    Require(directory.value >= 0, "cannot open staging directory");
    Fd file{::openat(directory.value, "compressed-token-map.bin", O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600)};
    Require(file.value >= 0, "cannot exclusively create token-map file");
    for (std::size_t offset = 0; offset < map.size();) {
      const auto count = ::pwrite(file.value, map.data() + offset, map.size() - offset, static_cast<off_t>(offset));
      if (count < 0 && errno == EINTR) continue;
      Require(count > 0, "token-map write failed"); offset += static_cast<std::size_t>(count);
    }
    Require(::fchmod(file.value, 0400) == 0 && ::fsync(file.value) == 0, "cannot sync read-only token-map file");
    std::vector<std::byte> readback(map.size());
    for (std::size_t offset = 0; offset < readback.size();) {
      const auto count = ::pread(file.value, readback.data() + offset, readback.size() - offset, static_cast<off_t>(offset));
      if (count < 0 && errno == EINTR) continue;
      Require(count > 0, "token-map readback failed"); offset += static_cast<std::size_t>(count);
    }
    Require(Take(sha256(readback)) == expected, "token-map readback digest differs");
    struct stat held{}, named{};
    Require(::fstat(file.value, &held) == 0 &&
        ::fstatat(directory.value, "compressed-token-map.bin", &named, AT_SYMLINK_NOFOLLOW) == 0 &&
        held.st_nlink == 1 && held.st_size == static_cast<off_t>(map.size()) &&
        held.st_dev == named.st_dev && held.st_ino == named.st_ino && !(held.st_mode & 0222), "token-map member changed");
    Require(::fchmod(directory.value, 0500) == 0 && ::fsync(directory.value) == 0, "cannot sync token-map directory");
    Require(::fstat(directory.value, &held) == 0 && ::fstatat(parent.value, staging.c_str(), &named, AT_SYMLINK_NOFOLLOW) == 0 &&
        held.st_dev == named.st_dev && held.st_ino == named.st_ino, "staging directory changed");
    Fd current_parent{Directory(output.parent_path())};
    Require(::fstat(current_parent.value, &held) == 0 && held.st_dev == parent_identity.st_dev &&
        held.st_ino == parent_identity.st_ino, "output parent changed");
    Require(::syscall(SYS_renameat2, parent.value, staging.c_str(), parent.value, name.c_str(), 1U) == 0,
        "no-replace token-map publication failed");
    published = true;
    Require(::fsync(parent.value) == 0, "token-map renamed but parent durability unresolved");
    UVersionInfo unicode{}; u_getUnicodeVersion(unicode);
    char unicode_text[U_MAX_VERSION_STRING_LENGTH]{}; u_versionToString(unicode, unicode_text);
    UVersionInfo icu{}; u_getVersion(icu);
    char icu_text[U_MAX_VERSION_STRING_LENGTH]{}; u_versionToString(icu, icu_text);
    auto receipt = Take(canonical_ascii_json(JsonValue(JsonValue::Object{
        {"schema", JsonValue(std::string("pih.deepseek-v41.token-map-receipt.v1"))},
        {"tokenizer_sha256", JsonValue(tokenizer_digest.hex())}, {"map_sha256", JsonValue(expected.hex())},
        {"bytes", JsonValue(static_cast<std::int64_t>(map.size()))},
        {"icu_version", JsonValue(std::string(icu_text))}, {"unicode_version", JsonValue(std::string(unicode_text))}}), 4096));
    Require(std::fwrite(receipt.data(), 1, receipt.size(), stdout) == receipt.size() &&
        std::fputc('\n', stdout) != EOF && ::fflush(stdout) == 0, "token-map published but receipt delivery failed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "V4.1 token map: %s. %s\n", error.what(), published ?
        "Directory was renamed; inspect output before retrying" : "Inspect retained staging; no automatic cleanup or overwrite");
    return published ? 3 : 2;
  }
}
