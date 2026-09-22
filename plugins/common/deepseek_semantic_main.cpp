#include "deepseek_semantic_artifact.h"
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: pih-deepseek-semantic-verify SNAPSHOT_DIR\n";
    return 2;
  }
  try {
    using pih::plugin_text::DeepSeekSemanticArtifacts;
    const auto admitted = DeepSeekSemanticArtifacts::Load(argv[1]);
    std::cout << "{\"schema\":\"pih.deepseek_v4_0731.native_semantic_admission.v1\","
      "\"source_revision\":\"" << DeepSeekSemanticArtifacts::kRevision << "\","
      "\"closure_root\":\"" << DeepSeekSemanticArtifacts::kClosureRoot << "\","
      "\"objects\":15,\"bos_token_id\":0,\"eos_token_id\":1,\"vocab_size\":129280}\n";
    if (!std::cout) throw std::runtime_error("receipt output failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native semantic admission failed: " << error.what() << '\n';
    return 2;
  }
}
