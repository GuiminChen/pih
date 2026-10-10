#include "../supervisor_config.h"
#include <cstdlib>
using namespace pih;
using namespace pih::deepseek_v41;
static void Require(bool value) { if (!value) std::abort(); }
#ifdef _WIN32
// Parse-only portability test: authenticated file loading is Linux-specific.
namespace pih::deepseek_v41 {
Result<std::vector<std::byte>> ReadAuthenticatedMetadata(const std::filesystem::path&,
    const Sha256Digest&, std::uint64_t) { return Status::Unavailable("file loading not exercised"); }
}
#endif
static Result<SupervisorConfig> Parse(std::string json) {
  auto hash = sha256(std::as_bytes(std::span(json.data(), json.size())));
  Require(hash.ok()); return SupervisorConfig::Parse(json, *hash);
}
int main() {
  const std::string hash(64, '1');
  std::string json = R"({"schema":"pih.deepseek-v41.supervisor.v2","batching":{"maximum_batch_sequences":4,"maximum_batch_tokens":256,"maximum_pending_requests":8},
"worker":{"path":"/bundle/worker","sha256":"HASH","byte_budget":1024},"helper":{"path":"/bundle/helper","sha256":"HASH","byte_budget":1024},"library_directories":[],
"cgroup":{"delegated_parent":"/delegated/pih","rank_limits":{"memory_bytes":4096,"pids":8,"cpu_quota_us":1000,"cpu_period_us":1000},"helper_limits":{"memory_bytes":4096,"pids":8,"cpu_quota_us":1000,"cpu_period_us":1000}},
"request":{"tokenizer_directory":"/tokenizer","rendered_prompt":"Hello","identity":{"epoch":1,"sequence_generation":1,"sampling_config_id":1},"sampling":{"temperature":0,"top_p":1,"top_k":0,"seed":7,"logprobs":false,"top_count":0},"stopping":{"minimum":0,"maximum":8,"tokens":[],"strings":[]},"budgets":{"vocabulary_bytes":1024,"publication_bytes":1048576,"record_bytes":1048576,"output_slots":8},"maximum_positions":128,"first_plan":1},
"model":{"config_sha256":"HASH","map_sha256":"HASH","plugin_lock":"/bundle/rank.lock","plugin_lock_sha256":"HASH","sm_major":10,"sm_minor":3,"staging_bytes":256},
"ranks":[{"device":0,"artifact_directory":"/weights/rank0","weight_manifest_sha256":"HASH","device_budget":1048576,"host_budget":1048576},{"device":1,"artifact_directory":"/weights/rank1","weight_manifest_sha256":"HASH","device_budget":1048576,"host_budget":1048576}],"timeouts":{"startup_ms":100,"sequence_ms":1000,"retirement_ms":1000,"terminate_grace_ms":100}})";
  for (auto at = json.find("HASH"); at != std::string::npos; at = json.find("HASH")) json.replace(at, 4, hash);
  auto result = Parse(json); Require(result.ok());
  Require(result->maximum_batch_sequences == 4 && result->maximum_batch_tokens == 256 && result->maximum_pending_requests == 8);
  Require(result->placements.size() == 2);
  for (const auto& placement : result->placements)
    Require(placement.packed_requests && placement.maximum_batch_sequences == 4 && placement.maximum_batch_tokens == 256 && placement.maximum_pending_requests == 8);
  auto invalid = json; auto at = invalid.find("\"maximum_batch_tokens\":256");
  Require(at != std::string::npos); invalid.replace(at, 26, "\"maximum_batch_tokens\":3"); Require(!Parse(invalid).ok());
  invalid = json; at = invalid.find("\"maximum_batch_sequences\":4");
  Require(at != std::string::npos); invalid.replace(at, 27, "\"maximum_batch_sequences\":33"); Require(!Parse(invalid).ok());
  invalid = json; at = invalid.find("\"maximum_pending_requests\":8");
  Require(at != std::string::npos); invalid.replace(at, 28, "\"maximum_pending_requests\":0"); Require(!Parse(invalid).ok());
  for (unsigned count : {1U,2U}) {
    auto single = json; at = single.find("\"maximum_batch_sequences\":4");
    single.replace(at, 27, "\"maximum_batch_sequences\":" + std::to_string(count));
    auto config = Parse(single); Require(config.ok() && config->maximum_batch_sequences == count && config->placements[0].packed_requests);
  }
}
