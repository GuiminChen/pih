#include "pih/model/engine_owned_resource_provider_set.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest provider_set_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

class KindProvider final : public EngineOwnedResourceKindProvider {
 public:
  explicit KindProvider(Result<Sha256Digest> result)
      : result_(std::move(result)) {}

  Result<Sha256Digest> observe() override {
    ++calls;
    return result_;
  }

  int calls = 0;

 private:
  Result<Sha256Digest> result_;
};

TEST(EngineOwnedResourceProviderSetTest, RoutesEachKindExactly) {
  std::array<KindProvider, 6> providers{
      KindProvider(provider_set_digest(1)), KindProvider(provider_set_digest(2)),
      KindProvider(provider_set_digest(3)), KindProvider(provider_set_digest(4)),
      KindProvider(provider_set_digest(5)), KindProvider(provider_set_digest(6))};
  std::array<EngineOwnedResourceProviderBinding, 6> bindings{};
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    const auto reversed = bindings.size() - index - 1;
    bindings[index] = {static_cast<EngineOwnedResourceKind>(reversed),
                       &providers[reversed]};
  }
  auto provider_set = EngineOwnedResourceProviderSet::Create(bindings).value();

  for (std::size_t index = 0; index < bindings.size(); ++index) {
    auto observed = provider_set.observe(
        static_cast<EngineOwnedResourceKind>(index));
    ASSERT_TRUE(observed.ok());
    EXPECT_EQ(*observed, provider_set_digest(static_cast<std::uint8_t>(index + 1)));
    EXPECT_EQ(providers[index].calls, 1);
  }
}

TEST(EngineOwnedResourceProviderSetTest, PreservesSelectedProviderFailure) {
  KindProvider ok(provider_set_digest(1));
  KindProvider unavailable(Status::Unavailable("network namespace unavailable"));
  std::array<EngineOwnedResourceProviderBinding, 6> bindings{
      EngineOwnedResourceProviderBinding{EngineOwnedResourceKind::kShm, &ok},
      {EngineOwnedResourceKind::kNetwork, &unavailable},
      {EngineOwnedResourceKind::kPinnedMemory, &ok},
      {EngineOwnedResourceKind::kListener, &ok},
      {EngineOwnedResourceKind::kArtifact, &ok},
      {EngineOwnedResourceKind::kGpuAllocation, &ok}};
  auto provider_set = EngineOwnedResourceProviderSet::Create(bindings).value();

  auto observed = provider_set.observe(EngineOwnedResourceKind::kNetwork);
  ASSERT_FALSE(observed.ok());
  EXPECT_EQ(observed.status().code(), StatusCode::kUnavailable);
  EXPECT_EQ(unavailable.calls, 1);
  EXPECT_EQ(ok.calls, 0);
}

TEST(EngineOwnedResourceProviderSetTest, RejectsIncompleteDuplicateAndNullBindings) {
  KindProvider provider(provider_set_digest(1));
  std::array<EngineOwnedResourceProviderBinding, 6> valid{
      EngineOwnedResourceProviderBinding{EngineOwnedResourceKind::kShm, &provider},
      {EngineOwnedResourceKind::kNetwork, &provider},
      {EngineOwnedResourceKind::kPinnedMemory, &provider},
      {EngineOwnedResourceKind::kListener, &provider},
      {EngineOwnedResourceKind::kArtifact, &provider},
      {EngineOwnedResourceKind::kGpuAllocation, &provider}};

  EXPECT_FALSE(EngineOwnedResourceProviderSet::Create(
                   std::span(valid).first(5)).ok());
  valid[5].kind = EngineOwnedResourceKind::kArtifact;
  EXPECT_FALSE(EngineOwnedResourceProviderSet::Create(valid).ok());
  valid[5] = {EngineOwnedResourceKind::kGpuAllocation, nullptr};
  EXPECT_FALSE(EngineOwnedResourceProviderSet::Create(valid).ok());
}

TEST(EngineOwnedResourceProviderSetTest, RejectsInvalidObserveKind) {
  KindProvider provider(provider_set_digest(1));
  std::array<EngineOwnedResourceProviderBinding, 6> bindings{
      EngineOwnedResourceProviderBinding{EngineOwnedResourceKind::kShm, &provider},
      {EngineOwnedResourceKind::kNetwork, &provider},
      {EngineOwnedResourceKind::kPinnedMemory, &provider},
      {EngineOwnedResourceKind::kListener, &provider},
      {EngineOwnedResourceKind::kArtifact, &provider},
      {EngineOwnedResourceKind::kGpuAllocation, &provider}};
  auto provider_set = EngineOwnedResourceProviderSet::Create(bindings).value();

  auto observed = provider_set.observe(
      static_cast<EngineOwnedResourceKind>(6));
  ASSERT_FALSE(observed.ok());
  EXPECT_EQ(observed.status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(provider.calls, 0);
}

}  // namespace
}  // namespace pih
