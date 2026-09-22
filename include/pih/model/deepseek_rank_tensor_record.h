#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pih/core/dtype.h"
#include "pih/core/sha256.h"
#include "pih/model/deepseek_runtime_records_manifest.h"
#include "pih/model/deepseek_tensor_ownership_plan.h"

namespace pih {

struct DeepSeekRankTensorRecord final {
  std::string tensor_name;
  std::string shard_name;
  DeepSeekTensorRole role{};
  DType dtype{};
  std::vector<std::uint64_t> shape;
  std::uint64_t file_begin = 0;
  std::uint64_t file_end = 0;
  std::uint32_t logical_layer = UINT32_MAX;
  std::uint64_t tensor_bytes = 0;
  DeepSeekStorageSemantics storage_semantics{};
  Sha256Digest artifact_root;
  Sha256Digest layout_root;
  Sha256Digest disposition_root;
  Sha256Digest target_logical_root;
  Sha256Digest disposition_record_root;
  Sha256Digest layout_record_root;
  Sha256Digest runtime_record_root;
};

}  // namespace pih
