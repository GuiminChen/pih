#include "pih/model/runtime_profile_reference_lease.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class Owner final : public RuntimeProfileReferenceLeaseOwner {};

Sha256Digest lease_root(std::uint8_t value) {
  Sha256Digest result{};
  result.bytes.fill(static_cast<std::byte>(value));
  return result;
}

RuntimeProfileReferenceClosure lease_closure() {
  std::vector<RuntimeProfileReferenceDescriptor> records;
  for (std::uint8_t role = 1; role <= 7; ++role) {
    records.push_back({static_cast<RuntimeProfileReferenceRole>(role),
                       "schema_v1", role, lease_root(role)});
  }
  return RuntimeProfileReferenceClosure::Create(std::move(records)).value();
}

class Probe final : public RuntimeProfileReferenceLeaseProbe {
 public:
  Result<RuntimeProfileReferenceLeaseObservation> observe(
      const RuntimeProfileReferenceDescriptor& descriptor) override {
    auto result = RuntimeProfileReferenceLeaseObservation{
        descriptor.role, descriptor.exact_bytes, descriptor.object_root,
        regular, immutable, owner ? std::make_shared<Owner>() : nullptr};
    if (wrong_root && descriptor.role ==
                          RuntimeProfileReferenceRole::kReleaseEvidence) {
      result.content_root = lease_root(9);
    }
    return result;
  }
  bool regular = true;
  bool immutable = true;
  bool owner = true;
  bool wrong_root = false;
};

TEST(RuntimeProfileReferenceLeaseTest, RequiresSevenPersistentExactLeases) {
  auto closure = lease_closure();
  Probe probe;
  auto verified = verify_runtime_profile_reference_leases(closure, probe);
  ASSERT_TRUE(verified.ok());
  EXPECT_EQ(verified->size(), 7U);
  probe.immutable = false;
  EXPECT_FALSE(verify_runtime_profile_reference_leases(closure, probe).ok());
  probe.immutable = true;
  probe.owner = false;
  EXPECT_FALSE(verify_runtime_profile_reference_leases(closure, probe).ok());
  probe.owner = true;
  probe.wrong_root = true;
  EXPECT_FALSE(verify_runtime_profile_reference_leases(closure, probe).ok());
}

}  // namespace
}  // namespace pih
