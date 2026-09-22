#pragma once
#include "source_artifact.h"
namespace pih::offline_deepseek {
struct SourceSemanticTensor final {
  SafetensorRecord tensor;
  Sha256Digest semantic_root;
};
struct SourceSemanticShard final {
  std::string name;
  std::uint64_t header_bytes = 0;
  std::uint64_t data_bytes = 0;
  Sha256Digest header_prefix_sha256;
  Sha256Digest tensor_set_root;
  Sha256Digest shard_root;
  std::vector<SourceSemanticTensor> tensors;
};
struct SourceSemantics final {
  std::vector<SourceSemanticShard> shards;
  Sha256Digest dtype_summary_root;
  Sha256Digest semantic_root;
};
// Actual retained source headers -> existing typed semantic closure. Numeric
// shard order, lexical tensor/dtype order, fixed dtype-byte ledger. Does not
// create source inventory records, copy payloads or authorize publication.
Result<SourceSemantics> CompileSourceSemantics(const SourceArtifact& source);
}  // namespace pih::offline_deepseek
