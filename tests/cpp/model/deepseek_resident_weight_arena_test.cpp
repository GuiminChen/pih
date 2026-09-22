#include "pih/model/deepseek_resident_weight_arena.h"
#include "pih/model/deepseek_weight_finalizer.h"
#include "pih/model/deepseek_engine_bootstrap_barrier.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <new>

namespace pih { namespace {

std::vector<DeepSeekRankTensorRecord> arena_records() {
  std::vector<DeepSeekRankTensorRecord> records;
  std::uint64_t offset = 0;
  for (std::uint32_t expert = 0; expert < 256; ++expert) {
    for (const auto* matrix : {"w1", "w2", "w3"}) {
      const bool w2 = std::string_view(matrix) == "w2";
      const auto prefix = "layers.4.ffn.experts." + std::to_string(expert) +
                          "." + matrix;
      records.push_back({prefix + ".weight", "a.safetensors",
                         DeepSeekTensorRole::kMainLayer, DType::kInt8,
                         w2 ? std::vector<std::uint64_t>{4096, 1024}
                            : std::vector<std::uint64_t>{2048, 2048},
                         offset, offset + 4194304});
      offset += 4194304;
      records.push_back({prefix + ".scale", "a.safetensors",
                         DeepSeekTensorRole::kMainLayer, DType::kFloat8E8M0,
                         w2 ? std::vector<std::uint64_t>{4096, 64}
                            : std::vector<std::uint64_t>{2048, 128},
                         offset, offset + 262144});
      offset += 262144;
    }
  }
  records.push_back({"layers.4.attn_norm.weight", "b.safetensors",
                     DeepSeekTensorRole::kMainLayer, DType::kBFloat16,
                     {4096}, 4096, 12288});
  return records;
}

DeepSeekWeightMaterializationPlan host_spill_plan() {
  auto tensors = arena_records();
  auto disposition = DeepSeekRankWeightDispositionManifest::Create(
      0, {4, 4}, DeepSeekRoutedExpertResidency::kHostSpill, tensors).value();
  auto experts = DeepSeekExpertBundleManifest::Create({4, 4}, tensors).value();
  return DeepSeekWeightMaterializationPlan::Create(
      disposition, experts, tensors).value();
}

class FixedByteSource final : public DeepSeekWeightByteSource {
 public:
  FixedByteSource() : bytes(8192, std::byte{0x5a}) {}
  Result<std::span<const std::byte>> resolve_weight_bytes(
      std::string_view name) const override {
    if (name != "layers.4.attn_norm.weight") {
      return Status::InvalidArgument("unexpected source name");
    }
    return std::span<const std::byte>(bytes.data(),
        malformed ? bytes.size() - 1U : bytes.size());
  }
  std::vector<std::byte> bytes;
  bool malformed = false;
};

class CountingAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++allocations;
    void* data = ::operator new(static_cast<std::size_t>(bytes),
                                std::align_val_t(alignment), std::nothrow);
    if (data == nullptr) return Status::ResourceExhausted("test allocation failed");
    return Allocation{data, bytes, alignment, 77, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    ++releases;
    ::operator delete(allocation.data, std::align_val_t(allocation.alignment));
  }
  int allocations = 0;
  int releases = 0;
};

class ScriptedCopier final : public MemoryCopier {
 public:
  Status copy(void* destination, Device, const void* source, Device,
              std::uint64_t bytes) override {
    ++calls;
    if (fail) return Status::Unavailable("injected DeepSeek copy failure");
    std::memcpy(destination, source, static_cast<std::size_t>(bytes));
    return Status::Ok();
  }
  bool fail = false;
  int calls = 0;
};

class ScriptedWeightDryRun final : public DeepSeekWeightConsumerDryRun {
 public:
  Result<Receipt> run(const DeepSeekWeightMaterializationPlan& plan,
                      const DeepSeekResidentWeightArena& arena,
                      const Sha256Digest& expected_layout_digest) override {
    ++calls;
    if (fail) return Status::Internal("injected consumer dry-run failure");
    auto tensor = arena.tensor(plan.copies().front().tensor_name);
    if (!tensor.ok()) return tensor.status();
    return Receipt{consumer_path_count,
                   malformed_receipt ? plan.copies().size() - 1U
                                     : plan.copies().size(),
                   arena.generation(), expected_layout_digest};
  }
  bool fail = false;
  bool malformed_receipt = false;
  std::uint32_t consumer_path_count = 1;
  int calls = 0;
};

TEST(DeepSeekResidentWeightArenaTest, PublishesOneGenerationAfterCompleteCopy) {
  auto plan = host_spill_plan();
  FixedByteSource source;
  CountingAllocator allocator;
  ScriptedCopier copier;
  auto arena = DeepSeekResidentWeightArena::Load(plan, source, allocator, copier);
  ASSERT_TRUE(arena.ok()) << arena.status().message();
  EXPECT_EQ(allocator.allocations, 1);
  EXPECT_EQ(arena->backing_bytes(), 8192U);
  EXPECT_EQ(arena->payload_bytes(), 8192U);
  EXPECT_EQ(arena->generation(), 77U);
  auto tensor = arena->tensor("layers.4.attn_norm.weight");
  ASSERT_TRUE(tensor.ok());
  EXPECT_EQ(tensor->generation(), 77U);
  EXPECT_EQ(*static_cast<const std::byte*>(tensor->data()), std::byte{0x5a});
  EXPECT_FALSE(arena->tensor("missing").ok());
}

TEST(DeepSeekResidentWeightArenaTest, ValidatesAllSourcesBeforeAllocation) {
  auto plan = host_spill_plan();
  FixedByteSource source;
  source.malformed = true;
  CountingAllocator allocator;
  ScriptedCopier copier;
  EXPECT_FALSE(DeepSeekResidentWeightArena::Load(
      plan, source, allocator, copier).ok());
  EXPECT_EQ(allocator.allocations, 0);
}

TEST(DeepSeekResidentWeightArenaTest, CopyFailureRollsBackWholeArena) {
  auto plan = host_spill_plan();
  FixedByteSource source;
  CountingAllocator allocator;
  ScriptedCopier copier;
  copier.fail = true;
  EXPECT_FALSE(DeepSeekResidentWeightArena::Load(
      plan, source, allocator, copier).ok());
  EXPECT_EQ(allocator.allocations, 1);
  EXPECT_EQ(allocator.releases, 1);
}

TEST(DeepSeekResidentWeightArenaTest, SealsOnlyAfterConsumerDryRun) {
  auto plan = host_spill_plan();
  FixedByteSource source;
  CountingAllocator allocator;
  ScriptedCopier copier;
  auto arena = DeepSeekResidentWeightArena::Load(plan, source, allocator, copier).value();
  auto finalizer = DeepSeekWeightFinalizer::Create(plan, arena);
  ASSERT_TRUE(finalizer.ok()) << finalizer.status().message();
  EXPECT_EQ(finalizer->state(), DeepSeekWeightFinalizeState::kFinalValidated);
  EXPECT_FALSE(finalizer->seal().ok());
  ScriptedWeightDryRun dry_run;
  EXPECT_TRUE(finalizer->run_dry_run(dry_run).ok());
  EXPECT_EQ(finalizer->state(), DeepSeekWeightFinalizeState::kDryRunPassed);
  EXPECT_TRUE(finalizer->seal().ok());
  EXPECT_EQ(finalizer->state(), DeepSeekWeightFinalizeState::kSealed);
  EXPECT_NE(finalizer->layout_digest(), Sha256Digest{});
  EXPECT_NE(finalizer->seal_digest(), Sha256Digest{});
  EXPECT_TRUE(finalizer->verify_unchanged().ok());
  EXPECT_FALSE(finalizer->run_dry_run(dry_run).ok());
  EXPECT_EQ(dry_run.calls, 1);
}

TEST(DeepSeekResidentWeightArenaTest, ConsumerFailurePermanentlyFailsFinalize) {
  auto plan = host_spill_plan();
  FixedByteSource source;
  CountingAllocator allocator;
  ScriptedCopier copier;
  auto arena = DeepSeekResidentWeightArena::Load(plan, source, allocator, copier).value();
  auto finalizer = DeepSeekWeightFinalizer::Create(plan, arena).value();
  ScriptedWeightDryRun dry_run;
  dry_run.fail = true;
  EXPECT_FALSE(finalizer.run_dry_run(dry_run).ok());
  EXPECT_EQ(finalizer.state(), DeepSeekWeightFinalizeState::kFailed);
  EXPECT_FALSE(finalizer.seal().ok());
  EXPECT_FALSE(finalizer.verify_unchanged().ok());
}

TEST(DeepSeekResidentWeightArenaTest, IncompleteDryRunReceiptFailsFinalize) {
  auto plan = host_spill_plan();
  FixedByteSource source;
  CountingAllocator allocator;
  ScriptedCopier copier;
  auto arena = DeepSeekResidentWeightArena::Load(plan, source, allocator, copier).value();
  auto finalizer = DeepSeekWeightFinalizer::Create(plan, arena).value();
  ScriptedWeightDryRun dry_run;
  dry_run.malformed_receipt = true;
  EXPECT_FALSE(finalizer.run_dry_run(dry_run).ok());
  EXPECT_EQ(finalizer.state(), DeepSeekWeightFinalizeState::kFailed);
}

TEST(DeepSeekResidentWeightArenaTest, RankReceiptCompletesWorldOneBootstrap) {
  auto plan = host_spill_plan();
  FixedByteSource source;
  CountingAllocator allocator;
  ScriptedCopier copier;
  auto arena = DeepSeekResidentWeightArena::Load(plan, source, allocator, copier).value();
  auto finalizer = DeepSeekWeightFinalizer::Create(plan, arena).value();
  ScriptedWeightDryRun dry_run;
  ASSERT_TRUE(finalizer.run_dry_run(dry_run).ok());
  ASSERT_TRUE(finalizer.seal().ok());
  DeepSeekRankMappingPlan mapping_plan;
  mapping_plan.rank = 0;
  mapping_plan.mapped_interval_bytes = 4096;
  mapping_plan.shards.push_back({"a.safetensors", 4096});
  mapping_plan.intervals.push_back({"a.safetensors", 0, 4096});
  auto lease_root = std::filesystem::temp_directory_path() /
                    "pih-bootstrap-receipt-test";
  std::filesystem::create_directory(lease_root);
  std::ofstream(lease_root / "a.safetensors", std::ios::binary | std::ios::trunc)
      .seekp(4095).put('\0');
  auto lease = ControllerFileLease::OpenBeneath(
      lease_root, "a.safetensors", 4096,
      ArtifactImmutabilityMode::kUncalibrated).value();
  std::vector<DeepSeekWorkerShardDescriptor> descriptors;
  descriptors.emplace_back("a.safetensors", lease.duplicate_for_worker().value());
  auto inventory = DeepSeekStageMappedInventory::Create(
      mapping_plan, std::move(descriptors)).value();
  auto pager = DeepSeekExpertPager::Create({4, 4}, 2, 2).value();
  auto receipt = DeepSeekRankBootstrapReceipt::Create(
      9, 1, inventory, plan, finalizer,
      DeepSeekRoutedExpertResidency::kHostSpill, &pager);
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  auto guard = DeepSeekArtifactEpochGuard::Create(9, 1000, 1).value();
  auto barrier = DeepSeekEngineBootstrapBarrier::Create(guard).value();
  EXPECT_TRUE(barrier.accept(*receipt).ok());
  EXPECT_TRUE(barrier.ready());
  EXPECT_TRUE(guard.admission_allowed());
  EXPECT_FALSE(barrier.accept(*receipt).ok());
  std::filesystem::remove_all(lease_root);
}

TEST(DeepSeekResidentWeightArenaTest, RankReceiptRejectsUnsealedWeights) {
  auto plan = host_spill_plan();
  FixedByteSource source;
  CountingAllocator allocator;
  ScriptedCopier copier;
  auto arena = DeepSeekResidentWeightArena::Load(plan, source, allocator, copier).value();
  auto finalizer = DeepSeekWeightFinalizer::Create(plan, arena).value();
  DeepSeekRankMappingPlan mapping_plan;
  mapping_plan.rank = 0;
  mapping_plan.mapped_interval_bytes = 1;
  mapping_plan.shards.push_back({"a.safetensors", 1});
  mapping_plan.intervals.push_back({"a.safetensors", 0, 1});
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-bootstrap-unsealed-test";
  std::filesystem::create_directory(root);
  std::ofstream(root / "a.safetensors", std::ios::binary | std::ios::trunc).put('x');
  auto lease = ControllerFileLease::OpenBeneath(
      root, "a.safetensors", 1,
      ArtifactImmutabilityMode::kUncalibrated).value();
  std::vector<DeepSeekWorkerShardDescriptor> descriptors;
  descriptors.emplace_back("a.safetensors", lease.duplicate_for_worker().value());
  auto inventory = DeepSeekStageMappedInventory::Create(
      mapping_plan, std::move(descriptors)).value();
  auto pager = DeepSeekExpertPager::Create({4, 4}, 2, 2).value();
  EXPECT_FALSE(DeepSeekRankBootstrapReceipt::Create(
      9, 1, inventory, plan, finalizer,
      DeepSeekRoutedExpertResidency::kHostSpill, &pager).ok());
  std::filesystem::remove_all(root);
}

} }  // namespace pih
