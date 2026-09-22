#pragma once
#include "source_semantics.h"
#include "source_payload.h"
namespace pih::offline_deepseek {
struct SourceInventoryShards final {
  Sha256Digest semantic_root;
  // Tensor/source-record roots and shard inventory roots are derived, not
  // supplied. The global namespace/grammar/inventory root is a separate step.
  std::vector<SourcePayloadShardInput> shards;
};
Result<SourceInventoryShards> CompileSourceInventoryShards(const SourceArtifact& source);
struct SourceInventory final {
  SourceInventoryShards plan;
  Sha256Digest grammar_root;
  Sha256Digest namespace_summary_root;
  Sha256Digest inventory_root;
};
// Rebuilds source semantics/records and complete inventory from the retained
// byte-admitted source, checking every frozen namespace count/byte ledger.
Result<SourceInventory> CompileSourceInventory(const SourceArtifact& source);
}  // namespace pih::offline_deepseek
