// CLI presentation only; verification is shared with offline conversion.
#include "generation_verify.h"
#include "prepare_generation.h"
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
template<class T> T Require(pih::Result<T> result) {
  if (!result.ok()) throw std::runtime_error(std::string(result.status().message()));
  return std::move(*result);
}
pih::Sha256Digest Root(const char* text) {
  auto root = Require(pih::Sha256Digest::ParseHex(text));
  if (root == pih::Sha256Digest{}) throw std::runtime_error("authority roots must be nonzero");
  return root;
}
}  // namespace

int main(int argc, char** argv) {
  const bool projection = argc == 10 && std::string_view(argv[1]) == "--source-bound-projection";
  const bool source_bound = projection || (argc == 10 && std::string_view(argv[1]) == "--source-bound");
  if (argc != 3 && !source_bound) {
    std::cerr << "usage: pih-deepseek-artifact-verify ABSOLUTE_GENERATION_ROOT EXPECTED_ARTIFACT_ROOT_SHA256\n"
        "   or: pih-deepseek-artifact-verify --source-bound SOURCE_DIRECTORY GENERATION_DIRECTORY "
        "MODEL_ROOT SEMANTIC_ROOT INVENTORY_ROOT PAYLOAD_ROOT CONVERTER_IDENTITY_ROOT ARTIFACT_ROOT\n"
        "Use --source-bound-projection instead to emit the canonical verification projection.\n";
    return 2;
  }
  try {
    const auto progress = [](std::string_view name, std::uint64_t bytes) {
        std::cerr << "hashing " << name << " (" << bytes << " bytes)\n";
    };
    pih::offline_deepseek::GenerationObservation observation;
    if (source_bound) {
      const pih::offline_deepseek::SourcePreparationAuthority authority{
          Root(argv[4]), Root(argv[5]), Root(argv[6]), Root(argv[7]), Root(argv[8])};
      const auto artifact_root = Root(argv[9]);
      std::cerr << "admitting source checkpoint, then rebuilding source-to-target equivalence; read-only\n";
      auto source = Require(pih::offline_deepseek::SourceArtifact::Open(argv[2], authority.model_digest));
      observation = Require(pih::offline_deepseek::VerifySourceBoundGeneration(
          *source, authority, argv[3], artifact_root, progress));
    } else {
      observation = Require(pih::offline_deepseek::VerifyGeneration(argv[1], Root(argv[2]), progress));
    }
    if (projection) {
      if (!observation.source_payload_equivalence || observation.verification_projection_json.empty())
        throw std::runtime_error("source-bound verification projection unavailable");
      // No trailing newline: hashing stdout reproduces the projection digest.
      std::cout << observation.verification_projection_json;
      return 0;
    }
    std::cout << "{\"schema\":\"pih.deepseek_pp1_artifact_observation.v1\",\"artifact_root\":\""
        << observation.artifact_root.hex() << "\",\"tensor_count\":" << observation.tensor_count
        << ",\"tensor_bytes\":" << observation.tensor_bytes << ",\"shard_count\":" << observation.shard_count
        << ",\"source_payload_equivalence\":" << (observation.source_payload_equivalence ? "true" : "false")
        << ",\"immutable_admission\":false,\"model_execution\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native DeepSeek artifact verification failed: " << error.what() << '\n';
    return 2;
  }
}
