#pragma once

#include <cstdint>

#include "pih/core/sha256.h"
#include "pih/model/deepseek_resident_weight_arena.h"

namespace pih {

enum class DeepSeekWeightFinalizeState : std::uint8_t {
  kFinalValidated,
  kDryRunPassed,
  kSealed,
  kFailed,
};

class DeepSeekWeightConsumerDryRun {
 public:
  virtual ~DeepSeekWeightConsumerDryRun() = default;
  struct Receipt final {
    std::uint32_t consumer_path_count = 0;
    std::uint64_t tensor_binding_count = 0;
    std::uint64_t arena_generation = 0;
    Sha256Digest layout_digest;
  };
  virtual Result<Receipt> run(
      const DeepSeekWeightMaterializationPlan& plan,
      const DeepSeekResidentWeightArena& arena,
      const Sha256Digest& expected_layout_digest) = 0;
};

class DeepSeekWeightFinalizer final {
 public:
  static Result<DeepSeekWeightFinalizer> Create(
      const DeepSeekWeightMaterializationPlan& plan,
      const DeepSeekResidentWeightArena& arena);

  Status run_dry_run(DeepSeekWeightConsumerDryRun& dry_run);
  Status seal();
  Status verify_unchanged() const;

  [[nodiscard]] DeepSeekWeightFinalizeState state() const noexcept {
    return state_;
  }
  [[nodiscard]] const Sha256Digest& seal_digest() const noexcept {
    return seal_digest_;
  }
  [[nodiscard]] const Sha256Digest& layout_digest() const noexcept {
    return layout_digest_;
  }

 private:
  DeepSeekWeightFinalizer(const DeepSeekWeightMaterializationPlan& plan,
                          const DeepSeekResidentWeightArena& arena,
                          Sha256Digest layout_digest)
      : plan_(&plan), arena_(&arena), layout_digest_(layout_digest) {}
  Result<Sha256Digest> current_layout_digest() const;
  Result<Sha256Digest> current_pointer_digest(
      const Sha256Digest& layout_digest) const;

  const DeepSeekWeightMaterializationPlan* plan_ = nullptr;
  const DeepSeekResidentWeightArena* arena_ = nullptr;
  DeepSeekWeightFinalizeState state_ =
      DeepSeekWeightFinalizeState::kFinalValidated;
  Sha256Digest layout_digest_{};
  Sha256Digest seal_digest_{};
};

}  // namespace pih
