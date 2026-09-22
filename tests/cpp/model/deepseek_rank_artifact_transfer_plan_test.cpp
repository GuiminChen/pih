#include "pih/model/deepseek_rank_artifact_transfer_plan.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>

#include <gtest/gtest.h>

#include "deepseek_rank_startup_test_fixture.h"
#include "deepseek_target_artifact_test_fixture.h"
#include "pih/model/deepseek_controller_artifact_catalog.h"
#include "pih/model/runtime_evidence_projection.h"

namespace pih {
namespace {

DeepSeekRuntimeArtifactEvidenceManifest transfer_evidence(
    const Sha256Digest& artifact_root) {
  return DeepSeekRuntimeArtifactEvidenceManifest::Create(
             RuntimeProfileGpuFamily::kH100Pcie80GiB, 1,
             RuntimeProfileResidency::kHostSpill,
             {test_fixture::rank_capacity_digest(1),
              test_fixture::rank_capacity_digest(2),
              test_fixture::rank_capacity_digest(3),
              test_fixture::rank_capacity_digest(5),
              test_fixture::rank_capacity_digest(6),
              test_fixture::rank_capacity_digest(7)},
             artifact_root,
             {test_fixture::rank_capacity_digest(61),
              test_fixture::rank_capacity_digest(62),
              test_fixture::rank_capacity_digest(63)})
      .value();
}

RuntimeEvidenceProjection transfer_projection(
    const DeepSeekRuntimeArtifactEvidenceManifest& evidence,
    std::uint8_t bundle_seed = 30) {
  return RuntimeEvidenceProjection::Create(
             RuntimeEvidenceBundleState::kComplete,
             RuntimeEvidenceReplayState::kPassed,
             RuntimeHardwareQualificationState::kSupported,
             {test_fixture::rank_capacity_digest(bundle_seed),
              test_fixture::rank_capacity_digest(31),
              test_fixture::rank_capacity_digest(32),
              evidence.domain_decision_roots_digest(),
              test_fixture::rank_capacity_digest(34)})
      .value();
}

TEST(DeepSeekRankArtifactTransferPlanTest,
     OwnsExactAdmissionStartupPipelineAndDescriptorAntecedents) {
  const auto artifact_root = test_fixture::reduced_target_artifact_root();
  auto evidence = transfer_evidence(artifact_root);
  auto projection = transfer_projection(evidence);
  auto startup = test_fixture::DeepSeekRankStartupFixture::Create(
      1, &projection, false);
  ASSERT_NE(startup, nullptr);
  auto model_startup = startup->compile_model_startup();
  ASSERT_TRUE(model_startup.ok()) << model_startup.status().message();
  auto binding = DeepSeekRuntimeArtifactAdmissionBinding::Issue(
      startup->admission(), evidence);
  ASSERT_TRUE(binding.ok()) << binding.status().message();

  const auto root = std::filesystem::temp_directory_path() /
                    "pih-rank-artifact-transfer-plan-test";
  std::filesystem::remove_all(root);
  test_fixture::write_reduced_target_generation_fixture(root);
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  auto catalog = DeepSeekControllerArtifactCatalog::OpenTargetGeneration(
      root, artifact_root, pipeline, 1024,
      ArtifactImmutabilityMode::kUncalibrated);
  ASSERT_TRUE(catalog.ok()) << catalog.status().message();
  auto handoff = DeepSeekRankArtifactHandoffPlan::Compile(*catalog);
  ASSERT_TRUE(handoff.ok()) << handoff.status().message();
  const auto model_startup_root = model_startup->plan_root();
  const auto binding_root = binding->binding_root();
  const auto handoff_root = handoff->plan_root();

  auto transfer = DeepSeekRankArtifactTransferPlan::Compile(
      std::move(*model_startup), std::move(*binding), std::move(*handoff));
  ASSERT_TRUE(transfer.ok()) << transfer.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactTransferPlanAbi,
            "pih_deepseek_rank_artifact_transfer_plan_v1");
  EXPECT_EQ(transfer->world_size(), 1U);
  EXPECT_EQ(transfer->artifact_root(), artifact_root);
  EXPECT_TRUE(transfer->owns_all_antecedents());
  EXPECT_EQ(transfer->artifact_binding_root(), binding_root);
  EXPECT_NE(transfer->plan_root(), Sha256Digest{});

  const auto& manifest = transfer->rank_manifest(0);
  EXPECT_EQ(manifest.fields().engine_epoch, 7U);
  EXPECT_EQ(manifest.fields().worker_generation, 8U);
  EXPECT_EQ(manifest.fields().rank, 0U);
  EXPECT_EQ(manifest.fields().descriptor_count, 1U);
  EXPECT_EQ(manifest.fields().tensor_record_count, 1U);
  EXPECT_EQ(manifest.fields().artifact_root, artifact_root);
  EXPECT_EQ(manifest.fields().artifact_admission_binding_root, binding_root);
  EXPECT_EQ(manifest.fields().model_startup_plan_root, model_startup_root);
  EXPECT_EQ(manifest.fields().artifact_handoff_plan_root, handoff_root);
  auto decoded = decode_deepseek_rank_artifact_transfer_manifest(
      transfer->encode_rank_manifest(0));
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->manifest_root(), manifest.manifest_root());

  auto replay_startup = test_fixture::DeepSeekRankStartupFixture::Create(
      1, &projection, false);
  ASSERT_NE(replay_startup, nullptr);
  auto replay_model = replay_startup->compile_model_startup().value();
  auto replay_binding = DeepSeekRuntimeArtifactAdmissionBinding::Issue(
                            replay_startup->admission(), evidence)
                            .value();
  auto replay_handoff =
      DeepSeekRankArtifactHandoffPlan::Compile(*catalog).value();
  auto replay = DeepSeekRankArtifactTransferPlan::Compile(
      std::move(replay_model), std::move(replay_binding),
      std::move(replay_handoff));
  ASSERT_TRUE(replay.ok()) << replay.status().message();
  EXPECT_EQ(replay->plan_root(), transfer->plan_root());
  EXPECT_EQ(replay->rank_manifest(0).manifest_root(), manifest.manifest_root());
  EXPECT_THROW((void)transfer->rank_manifest(1), std::out_of_range);
  EXPECT_THROW((void)transfer->encode_rank_manifest(1), std::out_of_range);
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactTransferPlanTest,
     RejectsArtifactBindingFromAnotherAdmissionAndArtifact) {
  const auto artifact_root = test_fixture::reduced_target_artifact_root();
  auto evidence = transfer_evidence(artifact_root);
  auto projection = transfer_projection(evidence);
  auto startup = test_fixture::DeepSeekRankStartupFixture::Create(
      1, &projection, false);
  ASSERT_NE(startup, nullptr);
  auto model_startup = startup->compile_model_startup().value();

  const auto foreign_artifact = test_fixture::rank_capacity_digest(70);
  auto foreign_evidence = transfer_evidence(foreign_artifact);
  auto foreign_projection = transfer_projection(foreign_evidence, 29);
  auto foreign_startup = test_fixture::DeepSeekRankStartupFixture::Create(
      1, &foreign_projection, false);
  ASSERT_NE(foreign_startup, nullptr);
  auto foreign_binding = DeepSeekRuntimeArtifactAdmissionBinding::Issue(
      foreign_startup->admission(), foreign_evidence)
                             .value();

  const auto root = std::filesystem::temp_directory_path() /
                    "pih-rank-artifact-transfer-splice-test";
  std::filesystem::remove_all(root);
  test_fixture::write_reduced_target_generation_fixture(root);
  auto pipeline = DeepSeekPipelinePlan::Create(1, false).value();
  auto catalog = DeepSeekControllerArtifactCatalog::OpenTargetGeneration(
                     root, artifact_root, pipeline, 1024,
                     ArtifactImmutabilityMode::kUncalibrated)
                     .value();
  auto handoff = DeepSeekRankArtifactHandoffPlan::Compile(catalog).value();
  EXPECT_FALSE(DeepSeekRankArtifactTransferPlan::Compile(
                   std::move(model_startup), std::move(foreign_binding),
                   std::move(handoff))
                   .ok());
  std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace pih
