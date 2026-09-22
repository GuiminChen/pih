#include "pih/model/engine_owned_resource_baseline_compiler.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest compiler_digest(std::uint8_t value) {
  Sha256Digest result{};
  result.bytes.fill(static_cast<std::byte>(value));
  return result;
}

std::array<EngineOwnedResourceBaselineEntry, 6> compiler_baselines() {
  std::array<EngineOwnedResourceBaselineEntry, 6> result{};
  for (std::uint8_t kind = 0; kind < result.size(); ++kind)
    result[kind] = {static_cast<EngineOwnedResourceKind>(kind),
                    compiler_digest(kind + 1)};
  return result;
}

class Provider final : public EngineOwnedResourceProvider {
 public:
  Provider() {
    for (std::uint8_t kind = 0; kind < values.size(); ++kind)
      values[kind] = compiler_digest(kind + 1);
  }
  Result<Sha256Digest> observe(EngineOwnedResourceKind kind) override {
    const auto index = static_cast<std::size_t>(kind);
    ++calls[index];
    if (!statuses[index].ok()) return statuses[index];
    return values[index];
  }
  std::array<Sha256Digest, 6> values{};
  std::array<Status, 6> statuses{
      Status::Ok(), Status::Ok(), Status::Ok(),
      Status::Ok(), Status::Ok(), Status::Ok()};
  std::array<int, 6> calls{};
};

TEST(EngineOwnedResourceBaselineCompilerTest, CompilesAllSixKindsInOrder) {
  Provider provider;
  auto compiler = EngineOwnedResourceBaselineCompiler::Create(
      7, compiler_digest(9), compiler_baselines(), provider).value();
  auto first = compiler.capture(100, 101);
  ASSERT_TRUE(first.ok());
  ASSERT_EQ(first->resources.size(), 6U);
  EXPECT_EQ(first->sample_identity, 1U);
  for (std::size_t index = 0; index < 6; ++index) {
    EXPECT_EQ(first->resources[index].kind,
              static_cast<EngineOwnedResourceKind>(index));
    EXPECT_TRUE(first->resources[index].visibility_complete);
    EXPECT_EQ(provider.calls[index], 1);
  }
  EXPECT_EQ(compiler.capture(101, 102)->sample_identity, 2U);
}

TEST(EngineOwnedResourceBaselineCompilerTest, PreservesPerKindUnavailable) {
  Provider provider;
  provider.statuses[1] = Status::Unavailable("network census unavailable");
  auto compiler = EngineOwnedResourceBaselineCompiler::Create(
      7, compiler_digest(9), compiler_baselines(), provider).value();
  auto receipt = compiler.capture(100, 101);
  ASSERT_TRUE(receipt.ok());
  EXPECT_FALSE(receipt->resources[1].visibility_complete);
  EXPECT_EQ(receipt->resources[1].observed_digest, Sha256Digest{});
  EXPECT_TRUE(receipt->resources[0].visibility_complete);
  EXPECT_TRUE(receipt->resources[2].visibility_complete);

  provider.statuses[1] = Status::Ok();
  auto recovered = compiler.capture(101, 102);
  ASSERT_TRUE(recovered.ok());
  EXPECT_EQ(recovered->sample_identity, 2U);
  EXPECT_TRUE(recovered->resources[1].visibility_complete);
  EXPECT_EQ(recovered->resources[1].observed_digest, compiler_digest(2));
}

TEST(EngineOwnedResourceBaselineCompilerTest, HardFailureAndZeroDigestPoison) {
  for (int mutation = 0; mutation < 2; ++mutation) {
    Provider provider;
    if (mutation == 0)
      provider.statuses[3] = Status::Internal("listener provider corrupted");
    else
      provider.values[3] = {};
    auto compiler = EngineOwnedResourceBaselineCompiler::Create(
        7, compiler_digest(9), compiler_baselines(), provider).value();
    EXPECT_FALSE(compiler.capture(100, 101).ok()) << mutation;
    EXPECT_FALSE(compiler.capture(101, 102).ok()) << mutation;
  }
}

TEST(EngineOwnedResourceBaselineCompilerTest, RejectsTimeDriftBeforeProvider) {
  Provider provider;
  auto compiler = EngineOwnedResourceBaselineCompiler::Create(
      7, compiler_digest(9), compiler_baselines(), provider).value();
  EXPECT_FALSE(compiler.capture(101, 100).ok());
  for (const auto calls : provider.calls) EXPECT_EQ(calls, 0);
}

}  // namespace
}  // namespace pih
