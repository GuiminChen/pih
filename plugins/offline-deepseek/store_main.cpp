#include "generation_store.h"
#include "generation_commit.h"
#include "generation_activate.h"
#include "active_generation.h"
#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {
template<class T> T Require(pih::Result<T> result) {
  if (!result.ok()) throw std::runtime_error(std::string(result.status().message()));
  return std::move(*result);
}
pih::Sha256Digest Root(const char* text) {
  const auto value = Require(pih::Sha256Digest::ParseHex(text));
  if (value == pih::Sha256Digest{}) throw std::runtime_error("commit authority root must be nonzero");
  return value;
}
int Commit(int argc, char** argv) {
  if (argc != 11) {
    std::cerr << "usage: pih-deepseek-generation-store commit SOURCE_DIRECTORY STORE_DIRECTORY STAGING_NAME "
        "MODEL_ROOT SEMANTIC_ROOT INVENTORY_ROOT PAYLOAD_ROOT CONVERTER_IDENTITY_ROOT ARTIFACT_ROOT\n";
    return 2;
  }
  try {
    const pih::offline_deepseek::SourcePreparationAuthority authority{
        Root(argv[5]), Root(argv[6]), Root(argv[7]), Root(argv[8]), Root(argv[9])};
    const auto artifact = Root(argv[10]);
    std::cerr << "admitting source checkpoint before generation commit; no activation\n";
    auto source = Require(pih::offline_deepseek::SourceArtifact::Open(argv[2], authority.model_digest));
    const auto outcome = pih::offline_deepseek::CommitGeneration(*source, authority, argv[3], argv[4], artifact,
        [](std::string_view name, std::uint64_t size) { std::cerr << "verifying " << name << " (" << size << " bytes)\n"; });
    const auto boolean = [](bool value) { return value ? "true" : "false"; };
    std::cout << "{\"schema\":\"pih.deepseek_generation_commit_outcome.v1\",\"success\":" << boolean(outcome.status.ok())
        << ",\"artifact_root\":\"" << artifact.hex() << "\",\"renamed\":" << boolean(outcome.promotion.renamed)
        << ",\"directories_synced\":" << boolean(outcome.promotion.directories_synced)
        << ",\"read_only_sealed\":" << boolean(outcome.promotion.read_only_sealed)
        << ",\"receipt_may_exist\":" << boolean(outcome.receipt_may_exist)
        << ",\"receipt_committed\":" << boolean(outcome.receipt_committed)
        << ",\"receipt_root\":\"" << outcome.receipt_root.hex()
        << "\",\"verification_projection_sha256\":\"" << outcome.verification_projection_sha256.hex()
        << "\",\"activated\":false,\"immutable_admission\":false,\"model_execution\":false}\n";
    if (!outcome.status.ok()) {
      std::cerr << "generation commit failed: " << outcome.status.message()
          << "; inspect phase flags and retained files; no automatic rollback\n";
      return 2;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "commit source admission failed: " << error.what() << '\n';
    return 2;
  }
}
int Activate(int argc, char** argv) {
  if (argc != 14) {
    std::cerr << "usage: pih-deepseek-generation-store activate SOURCE_DIRECTORY STORE_DIRECTORY "
        "MODEL_ROOT SEMANTIC_ROOT INVENTORY_ROOT PAYLOAD_ROOT CONVERTER_IDENTITY_ROOT ARTIFACT_ROOT "
        "RECEIPT_ROOT CATALOG_ROOT ORDINAL PREVIOUS_POINTER_ROOT_OR_none\n";
    return 2;
  }
  try {
    const pih::offline_deepseek::SourcePreparationAuthority authority{
        Root(argv[4]), Root(argv[5]), Root(argv[6]), Root(argv[7]), Root(argv[8])};
    const auto artifact = Root(argv[9]), receipt = Root(argv[10]), catalog = Root(argv[11]);
    const std::string_view ordinal_text(argv[12]);
    std::uint64_t ordinal = 0;
    const auto parsed = std::from_chars(ordinal_text.data(), ordinal_text.data() + ordinal_text.size(), ordinal);
    if (parsed.ec != std::errc{} || parsed.ptr != ordinal_text.data() + ordinal_text.size() ||
        std::to_string(ordinal) != ordinal_text)
      throw std::runtime_error("activation ordinal must be a canonical positive decimal integer");
    std::optional<pih::Sha256Digest> previous;
    if (std::string_view(argv[13]) != "none") previous = Root(argv[13]);
    (void)Require(pih::offline_deepseek::EncodeGenerationPointer({artifact, receipt, catalog, ordinal, previous}));
    std::cerr << "admitting source checkpoint before atomic activation; no Worker reload\n";
    auto source = Require(pih::offline_deepseek::SourceArtifact::Open(argv[2], authority.model_digest));
    const auto outcome = pih::offline_deepseek::ActivateGeneration(*source, authority, argv[3], artifact,
        receipt, catalog, ordinal, previous,
        [](std::string_view name, std::uint64_t size) { std::cerr << "verifying " << name << " (" << size << " bytes)\n"; });
    const auto boolean = [](bool value) { return value ? "true" : "false"; };
    std::cout << "{\"schema\":\"pih.deepseek_generation_activation_outcome.v1\",\"success\":" << boolean(outcome.status.ok())
        << ",\"temporary_may_exist\":" << boolean(outcome.temporary_may_exist)
        << ",\"pointer_replaced\":" << boolean(outcome.pointer_replaced)
        << ",\"directories_synced\":" << boolean(outcome.directories_synced)
        << ",\"activated\":" << boolean(outcome.activated)
        << ",\"pointer_root\":\"" << outcome.pointer_root.hex()
        << "\",\"worker_reloaded\":false,\"immutable_admission\":false,\"model_execution\":false}\n";
    if (!outcome.status.ok()) {
      std::cerr << "activation failed: " << outcome.status.message()
          << "; inspect phase flags and retained pointer files; no automatic rollback\n";
      return 2;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "activation admission failed: " << error.what() << '\n';
    return 2;
  }
}
int Resolve(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: pih-deepseek-generation-store resolve STORE_DIRECTORY EXPECTED_POINTER_ROOT EXPECTED_CATALOG_ROOT\n";
    return 2;
  }
  try {
    const auto observed = Require(pih::offline_deepseek::ObserveActiveGeneration(argv[2], Root(argv[3]), Root(argv[4]),
        [](std::string_view name, std::uint64_t size) { std::cerr << "verifying " << name << " (" << size << " bytes)\n"; }));
    const auto& pointer = observed.pointer;
    std::cout << "{\"schema\":\"pih.deepseek_active_generation_observation.v1\",\"pointer_root\":\""
        << pointer.pointer_root.hex() << "\",\"catalog_root\":\"" << pointer.input.catalog_root.hex()
        << "\",\"artifact_root\":\"" << pointer.input.artifact_root.hex()
        << "\",\"receipt_root\":\"" << pointer.input.receipt_root.hex()
        << "\",\"generation_name\":\"sha256-" << pointer.input.artifact_root.hex()
        << "\",\"activation_ordinal\":" << pointer.input.activation_ordinal
        << ",\"receipt_binding_verified\":true,\"source_payload_equivalence\":false,\"immutable_admission\":false,\"model_execution\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "active generation resolution failed: " << error.what() << '\n';
    return 2;
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string_view(argv[1]) == "commit") return Commit(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "activate") return Activate(argc, argv);
  if (argc > 1 && std::string_view(argv[1]) == "resolve") return Resolve(argc, argv);
  if (argc != 3 || (std::string_view(argv[1]) != "init" && std::string_view(argv[1]) != "verify")) {
    std::cerr << "usage: pih-deepseek-generation-store init|verify ABSOLUTE_STORE_DIRECTORY\n";
    return 2;
  }
  const bool initialize = std::string_view(argv[1]) == "init";
  if (initialize) {
    auto status = pih::offline_deepseek::InitializeGenerationStore(argv[2]);
    if (!status.ok()) {
      std::cerr << "store initialization failed: " << status.message() << '\n';
      return 2;
    }
  }
  auto store = pih::offline_deepseek::GenerationStore::Open(argv[2]);
  if (!store.ok()) {
    std::cerr << "store structure verification failed: " << store.status().message() << '\n';
    return 2;
  }
  std::cout << "{\"schema\":\"pih.deepseek_generation_store_observation.v1\",\"initialized\":"
      << (initialize ? "true" : "false")
      << ",\"structure_verified\":true,\"member_contents_verified\":false,\"activated\":false,\"model_execution\":false}\n";
  return 0;
}
