#include "../../../plugins/model-deepseek-v41/supervisor_deployment.h"
#include <gtest/gtest.h>

namespace pih::deepseek_v41 {
namespace {
SupervisorConfig Config() {
  SupervisorConfig config;
  config.identity = {1, 0, 1, 1};
  config.first_plan = 1;
  config.maximum_positions = 128;
  config.stopping.maximum = 32;
  config.stopping.token_count = 1; config.stopping.tokens[0] = 7;
  config.budgets.record_bytes = 32 * sizeof(AcceptedToken);
  config.budgets.vocabulary_bytes = 64U << 20;
  config.budgets.output_slots = 1;
  config.budgets.publication_bytes = sizeof(TokenPublication);
  return config;
}
TEST(NativeV41SupervisorPolicy, AllowsBoundedRequestWithoutChangingDefaults) {
  const auto config = Config();
  auto stopping = config.stopping; stopping.maximum = 8;
  stopping.pattern_count = 1; stopping.patterns[0] = "END";
  auto sampling = config.sampling; sampling.seed = 99;
  EXPECT_TRUE(ValidateSupervisorRequestPolicy(config, "rendered prompt", sampling, stopping).ok());
  EXPECT_EQ(config.stopping.maximum, 32U); EXPECT_EQ(config.stopping.pattern_count, 0U);
}
TEST(NativeV41SupervisorPolicy, RejectsCompletionAboveDeploymentOrRecordBudget) {
  auto config = Config(); auto stopping = config.stopping;
  stopping.maximum = 33;
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, "x", {}, stopping).ok());
  stopping.maximum = 32; config.budgets.record_bytes = 31 * sizeof(AcceptedToken);
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, "x", {}, stopping).ok());
}
TEST(NativeV41SupervisorPolicy, RejectsRemovingOrReplacingModelStopTokens) {
  const auto config = Config(); auto stopping = config.stopping;
  stopping.token_count = 0; stopping.tokens[0] = 0;
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, "x", {}, stopping).ok());
  stopping = config.stopping; stopping.tokens[0] = 8;
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, "x", {}, stopping).ok());
}
TEST(NativeV41SupervisorPolicy, RejectsResumedSamplingAndInvalidStopPolicy) {
  const auto config = Config(); auto sampling = config.sampling;
  sampling.ordinal = 1;
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, "x", sampling, config.stopping).ok());
  auto stopping = config.stopping; stopping.minimum = stopping.maximum + 1;
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, "x", {}, stopping).ok());
}
TEST(NativeV41SupervisorPolicy, RejectsInsufficientPublicationAndVocabularyBudgets) {
  auto config = Config(); --config.budgets.publication_bytes;
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, "x", {}, config.stopping).ok());
  config = Config(); config.budgets.output_slots = 4097;
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, "x", {}, config.stopping).ok());
  config = Config(); config.budgets.vocabulary_bytes = 0;
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, "x", {}, config.stopping).ok());
}
TEST(NativeV41SupervisorPolicy, RejectsDirtySuppressionAndOutOfRangeIdentity) {
  auto config = Config(); SamplingParameters sampling; sampling.suppressed[16] = 1;
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, "x", sampling, config.stopping).ok());
  config.identity.sequence_generation = static_cast<std::uint64_t>(INT64_MAX) + 1;
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, "x", {}, config.stopping).ok());
}
TEST(NativeV41SupervisorPolicy, RejectsEmptyOrOversizedPromptAndIdentity) {
  auto config = Config();
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, "", {}, config.stopping).ok());
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, std::string((1U << 20) + 1, 'x'), {}, config.stopping).ok());
  config.identity.sequence_generation = 0;
  EXPECT_FALSE(ValidateSupervisorRequestPolicy(config, "x", {}, config.stopping).ok());
}
}  // namespace
}  // namespace pih::deepseek_v41
