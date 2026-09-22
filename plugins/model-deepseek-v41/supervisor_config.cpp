#include "supervisor_config.h"
#include "supervisor_deployment.h"
#include "pih/core/bounded_json.h"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <new>
#include <stdexcept>

namespace pih::deepseek_v41 {
namespace {
constexpr std::size_t kMaximumBytes = 2U << 20;
[[noreturn]] void Invalid() { throw std::invalid_argument("supervisor configuration schema or value invalid"); }
const JsonValue& Field(const JsonValue& object, std::string_view name) {
  const auto* value = object.at(name); if (!value) Invalid(); return *value;
}
void Object(const JsonValue& value, std::initializer_list<std::string_view> keys) {
  if (!value.is_object() || value.object().size() != keys.size()) Invalid();
  for (auto key : keys) if (!value.at(key)) Invalid();
}
std::uint64_t Integer(const JsonValue& value, std::uint64_t low, std::uint64_t high) {
  if (!value.is_integer() || value.integer() < 0) Invalid();
  const auto result = static_cast<std::uint64_t>(value.integer());
  if (result < low || result > high) Invalid(); return result;
}
std::string Text(const JsonValue& value, std::size_t maximum, bool empty = false) {
  if (!value.is_string() || (!empty && value.string().empty()) || value.string().size() > maximum) Invalid();
  return value.string();
}
std::string Path(const JsonValue& value) {
  auto text = Text(value, 511);
  if (text.size() < 2 || text.front() != '/' || text.back() == '/' || text.find("//") != text.npos) Invalid();
  for (const unsigned char c : text) if (c < 32 || c == 127) Invalid();
  std::string_view rest(text); rest.remove_prefix(1);
  while (!rest.empty()) {
    const auto end = rest.find('/'); const auto part = rest.substr(0, end);
    if (part.empty() || part == "." || part == "..") Invalid();
    if (end == rest.npos) break; rest.remove_prefix(end + 1);
  }
  return text;
}
Sha256Digest Digest(const JsonValue& value) {
  const auto text = Text(value, 64);
  auto hash = Sha256Digest::ParseHex(text); if (!hash.ok() || *hash == Sha256Digest{}) Invalid(); return *hash;
}
SupervisorExecutableConfig Executable(const JsonValue& value) {
  Object(value, {"path", "sha256", "byte_budget"});
  return {Path(Field(value, "path")), Digest(Field(value, "sha256")), Integer(Field(value, "byte_budget"), 64, 512ULL << 20)};
}
WorkerCgroupLimits Limits(const JsonValue& value) {
  Object(value, {"memory_bytes", "pids", "cpu_quota_us", "cpu_period_us"});
  WorkerCgroupLimits result{Integer(Field(value, "memory_bytes"), 4096, INT64_MAX),
      Integer(Field(value, "pids"), 1, 65536), Integer(Field(value, "cpu_quota_us"), 1000, 1000000000),
      Integer(Field(value, "cpu_period_us"), 1000, 1000000)};
  if (result.memory_bytes % 4096) Invalid(); return result;
}
float Number(const JsonValue& value) {
  if (!value.is_number()) Invalid();
  const auto number = value.number(); const auto result = static_cast<float>(number);
  if (!std::isfinite(number) || !std::isfinite(result)) Invalid(); return result;
}
}
Result<SupervisorConfig> SupervisorConfig::Load(const std::filesystem::path& path, const Sha256Digest& expected) {
  auto bytes = ReadAuthenticatedMetadata(path, expected, kMaximumBytes); if (!bytes.ok()) return bytes.status();
  return Parse({reinterpret_cast<const char*>(bytes->data()), bytes->size()}, expected);
}
Result<SupervisorConfig> SupervisorConfig::Parse(std::string_view json, const Sha256Digest& expected) {
  if (json.empty() || json.size() > kMaximumBytes || expected == Sha256Digest{})
    return Status::InvalidArgument("Supervisor configuration extent or trusted digest invalid");
  const auto digest = sha256(std::as_bytes(std::span(json.data(), json.size())));
  if (!digest.ok()) return digest.status();
  if (*digest != expected) return Status::FailedPrecondition("Supervisor configuration differs from trusted digest");
  try {
    auto parsed = JsonValue::Parse(json, {kMaximumBytes, 8, 4096, 1U << 20});
    if (!parsed.ok()) return parsed.status();
    const auto& root = *parsed;
    Object(root, {"schema", "worker", "helper", "library_directories", "cgroup", "request", "model", "ranks", "timeouts"});
    if (Text(Field(root, "schema"), 64) != "pih.deepseek-v41.supervisor.v1") Invalid();
    SupervisorConfig out;
    out.worker = Executable(Field(root, "worker")); out.helper = Executable(Field(root, "helper"));
    const auto& libraries = Field(root, "library_directories");
    if (!libraries.is_array() || libraries.array().size() > 16) Invalid();
    std::size_t library_bytes = 0;
    for (const auto& value : libraries.array()) {
      auto path = Path(value);
      if (path.find_first_of(":;$") != path.npos || std::find(out.library_directories.begin(), out.library_directories.end(), path) != out.library_directories.end()) Invalid();
      library_bytes += path.size() + (out.library_directories.empty() ? 0 : 1);
      if (library_bytes > 4096) Invalid(); out.library_directories.push_back(std::move(path));
    }
    const auto& cgroup = Field(root, "cgroup"); Object(cgroup, {"delegated_parent", "rank_limits", "helper_limits"});
    out.delegated_cgroup = Path(Field(cgroup, "delegated_parent"));
    out.rank_limits = Limits(Field(cgroup, "rank_limits")); out.helper_limits = Limits(Field(cgroup, "helper_limits"));
    const auto& request = Field(root, "request");
    Object(request, {"tokenizer_directory", "rendered_prompt", "identity", "sampling", "stopping", "budgets", "maximum_positions", "first_plan"});
    out.tokenizer_directory = Path(Field(request, "tokenizer_directory"));
    out.rendered_prompt = Text(Field(request, "rendered_prompt"), 1U << 20);
    out.maximum_positions = Integer(Field(request, "maximum_positions"), 2, 1048576);
    out.first_plan = Integer(Field(request, "first_plan"), 1, INT64_MAX);
    const auto& identity = Field(request, "identity"); Object(identity, {"epoch", "sequence_generation", "sampling_config_id"});
    out.identity.epoch = Integer(Field(identity, "epoch"), 1, INT64_MAX);
    out.identity.sequence_generation = Integer(Field(identity, "sequence_generation"), 1, INT64_MAX);
    out.identity.sampling_config_id = Integer(Field(identity, "sampling_config_id"), 1, INT64_MAX);
    const auto& sampling = Field(request, "sampling"); Object(sampling, {"temperature", "top_p", "top_k", "seed", "logprobs", "top_count"});
    out.sampling.temperature = Number(Field(sampling, "temperature")); out.sampling.top_p = Number(Field(sampling, "top_p"));
    out.sampling.top_k = Integer(Field(sampling, "top_k"), 0, 129280); out.sampling.seed = Integer(Field(sampling, "seed"), 0, INT64_MAX);
    const auto& logprobs = Field(sampling, "logprobs"); if (!logprobs.is_boolean()) Invalid(); out.sampling.logprobs = logprobs.boolean();
    out.sampling.top_count = Integer(Field(sampling, "top_count"), 0, 20);
    auto valid = ValidateSamplingParameters(out.sampling); if (!valid.ok()) return valid;
    const auto& stopping = Field(request, "stopping"); Object(stopping, {"minimum", "maximum", "tokens", "strings"});
    out.stopping.minimum = Integer(Field(stopping, "minimum"), 0, 1048576);
    out.stopping.maximum = Integer(Field(stopping, "maximum"), 1, out.maximum_positions - 1);
    const auto& tokens = Field(stopping, "tokens"); const auto& strings = Field(stopping, "strings");
    if (!tokens.is_array() || tokens.array().size() > 17 || !strings.is_array() || strings.array().size() > 16) Invalid();
    for (const auto& token : tokens.array()) out.stopping.tokens[out.stopping.token_count++] = Integer(token, 0, 129279);
    for (const auto& text : strings.array()) out.stopping.patterns[out.stopping.pattern_count++] = Text(text, 256);
    auto stop = TokenStopState::Create(out.stopping); if (!stop.ok()) return stop.status();
    const auto& budgets = Field(request, "budgets"); Object(budgets, {"vocabulary_bytes", "publication_bytes", "record_bytes", "output_slots"});
    out.budgets = {Integer(Field(budgets, "vocabulary_bytes"), 1, 64U << 20), Integer(Field(budgets, "publication_bytes"), 1, INT64_MAX),
        Integer(Field(budgets, "record_bytes"), 1, INT64_MAX), static_cast<std::uint32_t>(Integer(Field(budgets, "output_slots"), 1, 4096))};
    const auto& times = Field(root, "timeouts"); Object(times, {"startup_ms", "sequence_ms", "retirement_ms", "terminate_grace_ms"});
    out.startup_ms = Integer(Field(times, "startup_ms"), 1, 86400000); out.sequence_ms = Integer(Field(times, "sequence_ms"), 1, 86400000);
    out.retirement_ms = Integer(Field(times, "retirement_ms"), 1, 300000); out.grace_ms = Integer(Field(times, "terminate_grace_ms"), 0, 299999);
    if (out.sequence_ms <= out.startup_ms || out.grace_ms >= out.retirement_ms) Invalid();
    const auto& model = Field(root, "model"); Object(model, {"config_sha256", "map_sha256", "plugin_lock", "plugin_lock_sha256", "sm_major", "sm_minor", "staging_bytes"});
    WorkerBootstrap common;
    common.config_sha256 = Digest(Field(model, "config_sha256")); common.map_sha256 = Digest(Field(model, "map_sha256"));
    common.plugin_lock = Path(Field(model, "plugin_lock")); common.plugin_lock_sha256 = Digest(Field(model, "plugin_lock_sha256"));
    common.sm_major = Integer(Field(model, "sm_major"), 1, 99); common.sm_minor = Integer(Field(model, "sm_minor"), 0, 9);
    common.staging_bytes = Integer(Field(model, "staging_bytes"), 256, 1U << 20);
    if (common.staging_bytes % 256) Invalid(); common.retirement_ms = out.retirement_ms;
    const auto& ranks = Field(root, "ranks");
    if (!ranks.is_array() || (ranks.array().size() != 2 && ranks.array().size() != 4 && ranks.array().size() != 8)) Invalid();
    for (const auto& rank : ranks.array()) {
      Object(rank, {"device", "artifact_directory", "weight_manifest_sha256", "device_budget", "host_budget"});
      auto placement = common; placement.world = ranks.array().size(); placement.rank = out.placements.size();
      placement.device = Integer(Field(rank, "device"), 0, INT32_MAX); placement.artifact_directory = Path(Field(rank, "artifact_directory"));
      placement.weight_manifest_sha256 = Digest(Field(rank, "weight_manifest_sha256"));
      placement.device_budget = Integer(Field(rank, "device_budget"), 1, INT64_MAX); placement.host_budget = Integer(Field(rank, "host_budget"), common.staging_bytes, INT64_MAX);
      for (const auto& prior : out.placements) if (prior.device == placement.device) Invalid();
      out.placements.push_back(std::move(placement));
    }
    const auto policy = ValidateSupervisorRequestPolicy(out, out.rendered_prompt, out.sampling, out.stopping);
    if (!policy.ok()) return policy;
    return out;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Supervisor configuration allocation failed"); }
  catch (const std::invalid_argument&) { return Status::InvalidArgument("Supervisor configuration schema or value invalid"); }
}
}  // namespace pih::deepseek_v41
