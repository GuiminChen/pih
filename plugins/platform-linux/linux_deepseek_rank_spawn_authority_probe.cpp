#include "pih/platform/linux/linux_deepseek_rank_spawn_authority_probe.h"

#include "pih/core/canonical_hash.h"
#include "pih/core/checked_math.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cerrno>
#include <cctype>
#include <climits>
#include <cstdint>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace pih {
namespace {

constexpr std::size_t kMaximumStatusBytes = 64U << 10U;
constexpr std::size_t kMaximumControlBytes = 4U << 20U;
constexpr std::size_t kMaximumProcEntries = 1U << 20U;
constexpr std::size_t kMaximumCgroupAncestors = 16;
constexpr std::uint32_t kCapSysAdmin = 21;
constexpr std::uint32_t kCapSysResource = 24;

struct DirectoryCloser final {
  void operator()(DIR* directory) const noexcept {
    if (directory != nullptr) (void)::closedir(directory);
  }
};

using OwnedDirectory = std::unique_ptr<DIR, DirectoryCloser>;

Status system_failure(std::string operation) {
  return Status::Internal(
      operation + " failed with errno " + std::to_string(errno));
}

Result<std::string> read_bounded(const std::filesystem::path& path,
                                 std::size_t maximum_bytes) {
  errno = 0;
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) return system_failure("open " + path.string());
  std::string result;
  result.reserve(std::min<std::size_t>(maximum_bytes, 4096));
  std::array<char, 4096> buffer{};
  while (input.good()) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count <= 0) break;
    if (static_cast<std::size_t>(count) > maximum_bytes - result.size()) {
      return Status::ResourceExhausted(
          "Linux rank spawn authority exceeds its byte bound");
    }
    result.append(buffer.data(), static_cast<std::size_t>(count));
  }
  if (input.bad()) return system_failure("read " + path.string());
  return result;
}

std::string_view trim_ascii(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

Result<std::uint64_t> decimal(std::string_view value,
                              std::string_view name) {
  value = trim_ascii(value);
  if (value.empty() ||
      !std::all_of(value.begin(), value.end(), [](unsigned char byte) {
        return byte >= '0' && byte <= '9';
      })) {
    return Status::InvalidArgument(
        std::string(name) + " is not canonical decimal");
  }
  std::uint64_t result = 0;
  const auto parsed = std::from_chars(
      value.data(), value.data() + value.size(), result, 10);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
    return Status::ResourceExhausted(
        std::string(name) + " exceeds uint64");
  }
  return result;
}

Result<std::uint64_t> hexadecimal(std::string_view value,
                                  std::string_view name) {
  value = trim_ascii(value);
  if (value.empty()) {
    return Status::InvalidArgument(std::string(name) + " is empty");
  }
  std::uint64_t result = 0;
  const auto parsed = std::from_chars(
      value.data(), value.data() + value.size(), result, 16);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
    return Status::InvalidArgument(
        std::string(name) + " is not bounded hexadecimal");
  }
  return result;
}

Result<std::uint64_t> read_decimal(const std::filesystem::path& path,
                                   bool require_positive = false) {
  auto text = read_bounded(path, kMaximumControlBytes);
  if (!text.ok()) return text.status();
  auto value = decimal(*text, path.string());
  if (!value.ok()) return value.status();
  if (require_positive && *value == 0) {
    return Status::InvalidArgument(
        path.string() + " must be positive");
  }
  return value;
}

struct ProcessStatus final {
  std::uint64_t real_uid = 0;
  std::uint64_t threads = 0;
  std::uint64_t effective_capabilities = 0;
};

Result<ProcessStatus> parse_process_status(
    std::string_view text, bool require_capabilities) {
  std::optional<std::uint64_t> uid;
  std::optional<std::uint64_t> threads;
  std::optional<std::uint64_t> capabilities;
  std::istringstream lines{std::string(text)};
  std::string line;
  while (std::getline(lines, line)) {
    const auto separator = line.find(':');
    if (separator == std::string::npos) continue;
    const auto key = std::string_view(line).substr(0, separator);
    auto value = trim_ascii(
        std::string_view(line).substr(separator + 1));
    if (key == "Uid") {
      if (uid) return Status::InvalidArgument("Linux status repeats Uid");
      std::istringstream uid_fields{std::string(value)};
      std::array<std::string, 4> values;
      std::string extra;
      if (!(uid_fields >> values[0] >> values[1] >> values[2] >> values[3]) ||
          uid_fields >> extra) {
        return Status::InvalidArgument("Linux status Uid is malformed");
      }
      for (std::size_t index = 0; index < values.size(); ++index) {
        auto parsed = decimal(values[index], "Linux status Uid");
        if (!parsed.ok()) return parsed.status();
        if (index == 0) uid = *parsed;
      }
    } else if (key == "Threads") {
      if (threads) {
        return Status::InvalidArgument("Linux status repeats Threads");
      }
      auto parsed = decimal(value, "Linux status Threads");
      if (!parsed.ok()) return parsed.status();
      threads = *parsed;
    } else if (key == "CapEff") {
      if (capabilities) {
        return Status::InvalidArgument("Linux status repeats CapEff");
      }
      auto parsed = hexadecimal(value, "Linux status CapEff");
      if (!parsed.ok()) return parsed.status();
      capabilities = *parsed;
    }
  }
  if (!uid || !threads || *threads == 0 ||
      (require_capabilities && !capabilities)) {
    return Status::InvalidArgument(
        "Linux process status authority is incomplete");
  }
  return ProcessStatus{*uid, *threads, capabilities.value_or(0)};
}

Result<ProcessStatus> self_status() {
  auto text = read_bounded("/proc/self/status", kMaximumStatusBytes);
  if (!text.ok()) return text.status();
  return parse_process_status(*text, true);
}

Result<std::uint64_t> rlimit_value(rlim_t value,
                                   std::string_view name) {
  if constexpr (sizeof(rlim_t) > sizeof(std::uint64_t)) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
      return Status::ResourceExhausted(
          std::string(name) + " exceeds uint64");
    }
  }
  return static_cast<std::uint64_t>(value);
}

Result<std::uint64_t> sample_uid_tasks(std::uint64_t real_uid) {
  OwnedDirectory directory(::opendir("/proc"));
  if (!directory) return system_failure("open /proc");
  std::uint64_t total = 0;
  std::size_t entries = 0;
  while (true) {
    errno = 0;
    const auto* entry = ::readdir(directory.get());
    if (entry == nullptr) {
      if (errno != 0) return system_failure("enumerate /proc");
      break;
    }
    if (++entries > kMaximumProcEntries) {
      return Status::ResourceExhausted(
          "Linux process inventory exceeds its bound");
    }
    const std::string_view name(entry->d_name);
    if (name.empty() ||
        !std::all_of(name.begin(), name.end(), [](unsigned char byte) {
          return byte >= '0' && byte <= '9';
        })) {
      continue;
    }
    auto text = read_bounded(
        std::filesystem::path("/proc") / std::string(name) / "status",
        kMaximumStatusBytes);
    if (!text.ok()) {
      std::error_code error;
      const bool exists = std::filesystem::exists(
          std::filesystem::path("/proc") / std::string(name), error);
      if (error) {
        return Status::Internal(
            "Linux process inventory cannot resolve a status race");
      }
      if (!exists) {
        continue;
      }
      return text.status();
    }
    auto status = parse_process_status(*text, false);
    if (!status.ok()) return status.status();
    if (status->real_uid != real_uid) continue;
    auto sum = checked_add_u64(total, status->threads);
    if (!sum.ok()) return sum.status();
    total = *sum;
  }
  if (total == 0) {
    return Status::FailedPrecondition(
        "Linux UID task inventory omitted the controller");
  }
  return total;
}

Result<std::uint64_t> sample_open_fds() {
  OwnedDirectory directory(::opendir("/proc/self/fd"));
  if (!directory) return system_failure("open /proc/self/fd");
  const int census_fd = ::dirfd(directory.get());
  std::set<int> descriptors;
  std::size_t entries = 0;
  while (true) {
    errno = 0;
    const auto* entry = ::readdir(directory.get());
    if (entry == nullptr) {
      if (errno != 0) {
        return system_failure("enumerate /proc/self/fd");
      }
      break;
    }
    if (++entries > kMaximumProcEntries) {
      return Status::ResourceExhausted(
          "Linux descriptor inventory exceeds its bound");
    }
    const std::string_view name(entry->d_name);
    if (name.empty() ||
        !std::all_of(name.begin(), name.end(), [](unsigned char byte) {
          return byte >= '0' && byte <= '9';
        })) {
      continue;
    }
    auto parsed = decimal(name, "Linux descriptor number");
    if (!parsed.ok() || *parsed > static_cast<std::uint64_t>(INT_MAX)) {
      return Status::InvalidArgument(
          "Linux descriptor inventory is malformed");
    }
    const int descriptor = static_cast<int>(*parsed);
    if (descriptor == census_fd) continue;
    errno = 0;
    if (::fcntl(descriptor, F_GETFD) >= 0) {
      descriptors.insert(descriptor);
    } else if (errno != EBADF) {
      return system_failure("inspect Linux descriptor");
    }
  }
  return static_cast<std::uint64_t>(descriptors.size());
}

Result<std::string> decode_mount_field(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size();) {
    if (value[index] != '\\') {
      const auto byte = static_cast<unsigned char>(value[index++]);
      if (byte < 0x20 || byte > 0x7e) {
        return Status::InvalidArgument(
            "Linux mount field is not canonical ASCII");
      }
      result.push_back(static_cast<char>(byte));
      continue;
    }
    if (index + 3 >= value.size()) {
      return Status::InvalidArgument("Linux mount escape is truncated");
    }
    const auto escaped = value.substr(index + 1, 3);
    char decoded = 0;
    if (escaped == "011") decoded = '\t';
    else if (escaped == "012") decoded = '\n';
    else if (escaped == "040") decoded = ' ';
    else if (escaped == "134") decoded = '\\';
    else return Status::InvalidArgument("Linux mount escape is unsupported");
    result.push_back(decoded);
    index += 4;
  }
  return result;
}

Result<std::filesystem::path> canonical_absolute_path(
    std::string_view value, std::string_view name) {
  if (value.empty() || value.front() != '/' || value.find('\\') != value.npos ||
      std::any_of(value.begin(), value.end(), [](unsigned char byte) {
        return byte < 0x20 || byte > 0x7e;
      })) {
    return Status::InvalidArgument(
        std::string(name) + " is not an absolute path");
  }
  std::filesystem::path path{std::string(value)};
  if (path.lexically_normal().generic_string() != value) {
    return Status::InvalidArgument(
        std::string(name) + " is not canonical");
  }
  return path;
}

bool escapes_parent(std::string_view relative) {
  return relative == ".." || relative.starts_with("../");
}

Result<std::filesystem::path> cgroup_membership(std::string_view text) {
  std::optional<std::filesystem::path> result;
  std::istringstream lines{std::string(text)};
  std::string line;
  while (std::getline(lines, line)) {
    if (!line.starts_with("0::")) continue;
    if (result) {
      return Status::InvalidArgument(
          "Linux unified cgroup membership is not unique");
    }
    auto path = canonical_absolute_path(
        std::string_view(line).substr(3), "Linux cgroup membership");
    if (!path.ok()) return path.status();
    result = std::move(*path);
  }
  if (!result) {
    return Status::InvalidArgument(
        "Linux unified cgroup membership is missing");
  }
  return *result;
}

struct CgroupMount final {
  std::filesystem::path mount_point;
  std::filesystem::path membership_relative;
  friend bool operator==(const CgroupMount&, const CgroupMount&) = default;
};

Result<CgroupMount> cgroup_mount(
    std::string_view mountinfo,
    const std::filesystem::path& membership) {
  std::optional<CgroupMount> result;
  std::istringstream lines{std::string(mountinfo)};
  std::string line;
  while (std::getline(lines, line)) {
    std::istringstream fields(line);
    std::vector<std::string> values;
    std::string value;
    while (fields >> value) values.push_back(std::move(value));
    const auto separator = std::find(values.begin(), values.end(), "-");
    if (separator == values.end() ||
        std::distance(separator, values.end()) < 4 || values.size() < 6 ||
        separator[1] != "cgroup2") {
      continue;
    }
    auto root = decode_mount_field(values[3]);
    auto point = decode_mount_field(values[4]);
    if (!root.ok()) return root.status();
    if (!point.ok()) return point.status();
    auto hierarchy_root = canonical_absolute_path(
        *root, "Linux cgroup2 hierarchy root");
    if (!hierarchy_root.ok()) return hierarchy_root.status();
    auto parsed = canonical_absolute_path(*point, "Linux cgroup2 mount");
    if (!parsed.ok()) return parsed.status();
    std::filesystem::path suffix;
    if (membership == *hierarchy_root) {
      suffix.clear();
    } else {
      const auto membership_text = membership.generic_string();
      auto root_text = hierarchy_root->generic_string();
      if (root_text != "/") root_text.push_back('/');
      if (!membership_text.starts_with(root_text)) continue;
      suffix = membership_text.substr(root_text.size());
    }
    std::error_code error;
    auto canonical = std::filesystem::canonical(*parsed, error);
    if (error) return Status::Internal("Linux cgroup2 mount cannot be resolved");
    CgroupMount candidate{std::move(canonical), std::move(suffix)};
    if (result && *result != candidate) {
      return Status::FailedPrecondition(
          "Linux cgroup2 authority is ambiguous");
    }
    result = std::move(candidate);
  }
  if (!result) {
    return Status::FailedPrecondition(
        "Linux cgroup2 root authority is unavailable");
  }
  return *result;
}

Result<Sha256Digest> cgroup_scope_root(
    const std::filesystem::path& path,
    const std::filesystem::path& mount,
    std::uint32_t depth) {
  struct stat metadata {};
  if (::stat(path.c_str(), &metadata) != 0) {
    return system_failure("stat cgroup scope");
  }
  if (!S_ISDIR(metadata.st_mode)) {
    return Status::FailedPrecondition("Linux cgroup scope is not a directory");
  }
  auto relative = path.lexically_relative(mount).generic_string();
  if (relative.empty() || escapes_parent(relative)) {
    return Status::FailedPrecondition(
        "Linux cgroup scope escapes the mounted authority");
  }
  if (relative == ".") relative = "/";
  else relative.insert(relative.begin(), '/');
  auto builder = CanonicalHashBuilder::Create(
      "pih:linux-deepseek-rank-cgroup-scope:v1", 4);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u64(1, static_cast<std::uint64_t>(metadata.st_dev));
  if (status.ok()) {
    status = builder->add_u64(2, static_cast<std::uint64_t>(metadata.st_ino));
  }
  if (status.ok()) status = builder->add_u32(3, depth);
  if (status.ok()) status = builder->add_ascii_utf8(4, relative);
  if (!status.ok()) return status;
  return builder->finalize();
}

Result<std::vector<DeepSeekRankSpawnCgroupPidsObservation>>
sample_cgroup_ancestors() {
  auto before = read_bounded("/proc/self/cgroup", kMaximumControlBytes);
  if (!before.ok()) return before.status();
  auto membership = cgroup_membership(*before);
  if (!membership.ok()) return membership.status();
  auto mountinfo = read_bounded("/proc/self/mountinfo", kMaximumControlBytes);
  if (!mountinfo.ok()) return mountinfo.status();
  auto mounted = cgroup_mount(*mountinfo, *membership);
  if (!mounted.ok()) return mounted.status();
  const auto& mount = mounted->mount_point;
  std::error_code error;
  auto leaf = std::filesystem::canonical(
      mount / mounted->membership_relative, error);
  if (error) {
    return Status::Internal("Linux cgroup leaf cannot be resolved");
  }
  auto relative = leaf.lexically_relative(mount);
  if (relative.empty() || escapes_parent(relative.generic_string())) {
    return Status::FailedPrecondition(
        "Linux cgroup membership escapes its mounted authority");
  }

  std::vector<DeepSeekRankSpawnCgroupPidsObservation> result;
  for (auto current = leaf;; current = current.parent_path()) {
    if (result.size() >= kMaximumCgroupAncestors) {
      return Status::ResourceExhausted(
          "Linux cgroup ancestry exceeds its bound");
    }
    auto scope = cgroup_scope_root(
        current, mount, static_cast<std::uint32_t>(result.size()));
    if (!scope.ok()) return scope.status();
    auto tasks = read_decimal(current / "pids.current");
    if (!tasks.ok()) return tasks.status();
    auto maximum_text = read_bounded(
        current / "pids.max", kMaximumControlBytes);
    if (!maximum_text.ok()) return maximum_text.status();
    const auto maximum_value = trim_ascii(*maximum_text);
    std::optional<std::uint64_t> maximum;
    if (maximum_value != "max") {
      auto parsed = decimal(maximum_value, "Linux cgroup pids.max");
      if (!parsed.ok()) return parsed.status();
      maximum = *parsed;
    }
    result.push_back({*scope, *tasks, maximum});
    if (current == mount) break;
    const auto parent = current.parent_path();
    if (parent == current) {
      return Status::FailedPrecondition(
          "Linux cgroup ancestry omitted its mount root");
    }
  }
  auto after = read_bounded("/proc/self/cgroup", kMaximumControlBytes);
  if (!after.ok()) return after.status();
  if (*after != *before) {
    return Status::FailedPrecondition(
        "Linux cgroup membership changed during collection");
  }
  return result;
}

struct FileNr final {
  std::uint64_t allocated = 0;
  std::uint64_t maximum = 0;
};

Result<FileNr> sample_file_nr() {
  auto text = read_bounded("/proc/sys/fs/file-nr", kMaximumControlBytes);
  if (!text.ok()) return text.status();
  std::istringstream fields(*text);
  std::string allocated_text;
  std::string unused_text;
  std::string maximum_text;
  std::string extra;
  if (!(fields >> allocated_text >> unused_text >> maximum_text) ||
      fields >> extra) {
    return Status::InvalidArgument("Linux file-nr authority is malformed");
  }
  auto allocated = decimal(allocated_text, "Linux file-nr allocated");
  auto unused = decimal(unused_text, "Linux file-nr unused");
  auto maximum = decimal(maximum_text, "Linux file-nr maximum");
  if (!allocated.ok()) return allocated.status();
  if (!unused.ok()) return unused.status();
  if (!maximum.ok()) return maximum.status();
  auto file_max = read_decimal("/proc/sys/fs/file-max", true);
  if (!file_max.ok()) return file_max.status();
  if (*maximum != *file_max) {
    return Status::FailedPrecondition(
        "Linux file-nr and file-max authorities differ");
  }
  return FileNr{*allocated, *maximum};
}

}  // namespace

Result<DeepSeekRankSpawnAuthoritySnapshot>
LinuxDeepSeekRankSpawnAuthorityProbe::read_authority() {
  auto status = self_status();
  if (!status.ok()) return status.status();
  const auto real_uid = static_cast<std::uint64_t>(::getuid());
  const auto exempt_mask = (std::uint64_t{1} << kCapSysAdmin) |
                           (std::uint64_t{1} << kCapSysResource);
  if (status->real_uid != real_uid) {
    return Status::FailedPrecondition(
        "Linux real UID authorities differ");
  }
  if (real_uid == 0 || (status->effective_capabilities & exempt_mask) != 0) {
    return Status::FailedPrecondition(
        "Linux RLIMIT_NPROC enforcement is not provable for this process");
  }

  struct rlimit nproc {};
  struct rlimit nofile {};
  if (::getrlimit(RLIMIT_NPROC, &nproc) != 0 ||
      ::getrlimit(RLIMIT_NOFILE, &nofile) != 0) {
    return system_failure("getrlimit for rank spawn");
  }
  std::optional<std::uint64_t> nproc_soft;
  if (nproc.rlim_cur != RLIM_INFINITY) {
    auto value = rlimit_value(nproc.rlim_cur, "RLIMIT_NPROC soft");
    if (!value.ok()) return value.status();
    nproc_soft = *value;
  }
  if (nproc.rlim_max != RLIM_INFINITY) {
    auto hard = rlimit_value(nproc.rlim_max, "RLIMIT_NPROC hard");
    if (!hard.ok()) return hard.status();
    if (!nproc_soft || *nproc_soft > *hard) {
      return Status::FailedPrecondition(
          "Linux RLIMIT_NPROC soft/hard authorities are inconsistent");
    }
  }
  if (nofile.rlim_cur == RLIM_INFINITY || nofile.rlim_max == RLIM_INFINITY) {
    return Status::FailedPrecondition("Linux RLIMIT_NOFILE must be finite");
  }
  auto nofile_soft = rlimit_value(nofile.rlim_cur, "RLIMIT_NOFILE soft");
  auto nofile_hard = rlimit_value(nofile.rlim_max, "RLIMIT_NOFILE hard");
  if (!nofile_soft.ok()) return nofile_soft.status();
  if (!nofile_hard.ok()) return nofile_hard.status();
  auto nr_open = read_decimal("/proc/sys/fs/nr_open", true);
  if (!nr_open.ok()) return nr_open.status();
  auto file_max = read_decimal("/proc/sys/fs/file-max", true);
  if (!file_max.ok()) return file_max.status();
  auto map_count = read_decimal("/proc/sys/vm/max_map_count", true);
  if (!map_count.ok()) return map_count.status();
  return DeepSeekRankSpawnAuthoritySnapshot{
      nproc_soft, true, *nofile_soft, *nofile_hard, *nr_open,
      *file_max, *map_count};
}

Result<DeepSeekRankSpawnUsageSnapshot>
LinuxDeepSeekRankSpawnAuthorityProbe::sample_usage() {
  auto status = self_status();
  if (!status.ok()) return status.status();
  const auto real_uid = static_cast<std::uint64_t>(::getuid());
  if (status->real_uid != real_uid) {
    return Status::FailedPrecondition(
        "Linux real UID changed during rank spawn collection");
  }
  auto uid_tasks = sample_uid_tasks(real_uid);
  if (!uid_tasks.ok()) return uid_tasks.status();
  auto cgroups = sample_cgroup_ancestors();
  if (!cgroups.ok()) return cgroups.status();
  auto open_fds = sample_open_fds();
  if (!open_fds.ok()) return open_fds.status();
  auto file_nr = sample_file_nr();
  if (!file_nr.ok()) return file_nr.status();
  return DeepSeekRankSpawnUsageSnapshot{
      *uid_tasks, std::move(*cgroups), *open_fds, file_nr->allocated};
}

}  // namespace pih
