#include "pih/model/runtime_profile_authority_lease.h"

#include <gtest/gtest.h>

#include <string_view>
#include <vector>

namespace pih {
namespace {

std::vector<std::byte> test_bytes(std::string_view value) {
  const auto raw = std::as_bytes(std::span(value.data(), value.size()));
  return {raw.begin(), raw.end()};
}

Sha256Digest test_digest(std::string_view value) {
  return sha256(test_bytes(value)).value();
}

ProfileSignature test_signature() {
  return {"release-key", std::vector<std::byte>(64, std::byte{7})};
}

RuntimeProfileReferenceClosure test_closure() {
  std::vector<RuntimeProfileReferenceDescriptor> records;
  for (std::uint8_t role = 1; role <= 7; ++role) {
    Sha256Digest object_root{};
    object_root.bytes.fill(static_cast<std::byte>(role));
    records.push_back({static_cast<RuntimeProfileReferenceRole>(role),
                       "schema_v1", role, object_root});
  }
  return RuntimeProfileReferenceClosure::Create(std::move(records)).value();
}

struct AuthorityFixture final {
  SealedReachableObjectDag graph = verify_sealed_reachable_object_dag(
      {{"schema_v1", test_digest("object"), 6, 1, true, {}}},
      {test_digest("object")}).value();
  RuntimeProfileReferenceClosure closure = test_closure();
  SignedProfileEnvelope envelope = SignedProfileEnvelope::Create(
      "profile", "r1", 1, test_bytes("payload"), closure.closure_root(),
      {test_signature()}).value();
  SignedProfileCatalog catalog = SignedProfileCatalog::CreateGraphBound(
      1, {graph.snapshot_root(), graph.node_count(), graph.edge_count()},
      {{"profile", "r1", envelope.envelope_root(), envelope.envelope_bytes()}},
      {test_signature()}).value();
  ProfileTrustPolicy policy = [](const SignedProfileCatalog& value) {
    ProfileTrustKey key{};
    key.key_id = "release-key";
    key.role = ProfileSignerRole::kRelease;
    key.public_key.fill(std::byte{3});
    return ProfileTrustPolicy::Create(1, value.catalog_root(), {key}).value();
  }(catalog);
};

class TestOwner final : public RuntimeProfileAuthorityLeaseOwner {};

class TestProbe final : public RuntimeProfileAuthorityLeaseProbe {
 public:
  Result<RuntimeProfileAuthorityLeaseObservation> observe(
      const RuntimeProfileAuthorityDescriptor& descriptor) override {
    auto observation = RuntimeProfileAuthorityLeaseObservation{
        descriptor.role, descriptor.exact_bytes, descriptor.content_root,
        regular, immutable, owner ? std::make_shared<TestOwner>() : nullptr};
    if (wrong_root && descriptor.role == RuntimeProfileAuthorityRole::kEnvelope) {
      observation.content_root = test_digest("wrong");
    }
    return observation;
  }

  bool regular = true;
  bool immutable = true;
  bool owner = true;
  bool wrong_root = false;
};

TEST(RuntimeProfileAuthorityLeaseTest, RequiresFivePersistentExactLeases) {
  AuthorityFixture fixture;
  TestProbe probe;
  auto verified = verify_runtime_profile_authority_leases(
      fixture.policy, fixture.catalog, fixture.envelope, fixture.closure,
      fixture.graph, probe);
  ASSERT_TRUE(verified.ok()) << verified.status().message();
  EXPECT_EQ(verified->size(), 5U);
  EXPECT_EQ(verified->policy_digest(), fixture.policy.policy_digest());
  EXPECT_EQ(verified->catalog_root(), fixture.catalog.catalog_root());
  EXPECT_EQ(verified->envelope_root(), fixture.envelope.envelope_root());
  EXPECT_EQ(verified->reference_closure_root(), fixture.closure.closure_root());
  EXPECT_EQ(verified->graph_snapshot_root(), fixture.graph.snapshot_root());

  probe.immutable = false;
  EXPECT_FALSE(verify_runtime_profile_authority_leases(
      fixture.policy, fixture.catalog, fixture.envelope, fixture.closure,
      fixture.graph, probe).ok());
  probe.immutable = true;
  probe.owner = false;
  EXPECT_FALSE(verify_runtime_profile_authority_leases(
      fixture.policy, fixture.catalog, fixture.envelope, fixture.closure,
      fixture.graph, probe).ok());
  probe.owner = true;
  probe.wrong_root = true;
  EXPECT_FALSE(verify_runtime_profile_authority_leases(
      fixture.policy, fixture.catalog, fixture.envelope, fixture.closure,
      fixture.graph, probe).ok());

  auto other_graph = verify_sealed_reachable_object_dag(
      {{"schema_v1", test_digest("other"), 5, 1, true, {}}},
      {test_digest("other")}).value();
  probe.wrong_root = false;
  EXPECT_FALSE(verify_runtime_profile_authority_leases(
      fixture.policy, fixture.catalog, fixture.envelope, fixture.closure,
      other_graph, probe).ok());
}

}  // namespace
}  // namespace pih
