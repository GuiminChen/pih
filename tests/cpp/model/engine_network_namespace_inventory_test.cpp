#include "pih/model/engine_network_namespace_inventory.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest network_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

class NetworkOperations final : public EngineNetworkNamespaceOperations {
 public:
  Result<EngineNetworkNamespaceSnapshot> capture(
      std::uint64_t namespace_identity) override {
    ++calls;
    last_identity = namespace_identity;
    return result;
  }

  int calls = 0;
  std::uint64_t last_identity = 0;
  Result<EngineNetworkNamespaceSnapshot> result =
      EngineNetworkNamespaceSnapshot{
          50, true, false,
          {{network_digest(1), network_digest(2), 65536, 2,
            EngineOwnedResourceLifecycleState::kActive},
           {network_digest(1), network_digest(3), 32768, 1,
            EngineOwnedResourceLifecycleState::kRetired}}};
};

TEST(EngineNetworkNamespaceInventoryTest, CapturesActiveAndRetiredOwners) {
  NetworkOperations operations;
  auto inventory = EngineNetworkNamespaceInventory::Create(50, operations)
                       .value();
  auto records = inventory.capture(EngineOwnedResourceKind::kNetwork);
  ASSERT_TRUE(records.ok());
  ASSERT_EQ(records->size(), 2U);
  EXPECT_EQ(records->at(0).state,
            EngineOwnedResourceLifecycleState::kActive);
  EXPECT_EQ(records->at(0).object_count, 2U);
  EXPECT_EQ(records->at(1).state,
            EngineOwnedResourceLifecycleState::kRetired);
  EXPECT_EQ(operations.last_identity, 50U);
}

TEST(EngineNetworkNamespaceInventoryTest, AcceptsDestroyedEmptyNamespace) {
  NetworkOperations operations;
  operations.result = EngineNetworkNamespaceSnapshot{50, false, true, {}};
  auto inventory = EngineNetworkNamespaceInventory::Create(50, operations)
                       .value();
  auto records = inventory.capture(EngineOwnedResourceKind::kNetwork);
  ASSERT_TRUE(records.ok());
  EXPECT_TRUE(records->empty());
}

TEST(EngineNetworkNamespaceInventoryTest, RejectsAmbiguousNamespaceState) {
  for (int mutation = 0; mutation < 4; ++mutation) {
    NetworkOperations operations;
    auto snapshot = *operations.result;
    if (mutation == 0) ++snapshot.namespace_identity;
    if (mutation == 1) snapshot.namespace_destroyed = true;
    if (mutation == 2) snapshot.namespace_handle_present = false;
    if (mutation == 3) {
      snapshot.namespace_handle_present = false;
      snapshot.namespace_destroyed = true;
    }
    operations.result = snapshot;
    auto inventory = EngineNetworkNamespaceInventory::Create(50, operations)
                         .value();
    auto records = inventory.capture(EngineOwnedResourceKind::kNetwork);
    ASSERT_FALSE(records.ok()) << mutation;
    EXPECT_EQ(records.status().code(), StatusCode::kFailedPrecondition)
        << mutation;
  }
}

TEST(EngineNetworkNamespaceInventoryTest, RejectsInvalidOwnerRecords) {
  for (int mutation = 0; mutation < 5; ++mutation) {
    NetworkOperations operations;
    auto snapshot = *operations.result;
    if (mutation == 0) snapshot.owners[0].owner_identity = {};
    if (mutation == 1) snapshot.owners[0].resource_identity = {};
    if (mutation == 2) snapshot.owners[0].object_count = 0;
    if (mutation == 3)
      snapshot.owners[0].state = EngineOwnedResourceLifecycleState::kPassive;
    if (mutation == 4) snapshot.owners.push_back(snapshot.owners[0]);
    operations.result = snapshot;
    auto inventory = EngineNetworkNamespaceInventory::Create(50, operations)
                         .value();
    EXPECT_FALSE(inventory.capture(EngineOwnedResourceKind::kNetwork).ok())
        << mutation;
  }
}

TEST(EngineNetworkNamespaceInventoryTest, PreservesUnavailableAndKindBoundary) {
  NetworkOperations operations;
  auto inventory = EngineNetworkNamespaceInventory::Create(50, operations)
                       .value();
  operations.result = Status::Unavailable("inet_diag unavailable");
  auto unavailable = inventory.capture(EngineOwnedResourceKind::kNetwork);
  ASSERT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), StatusCode::kUnavailable);
  EXPECT_FALSE(inventory.capture(EngineOwnedResourceKind::kListener).ok());
  EXPECT_FALSE(EngineNetworkNamespaceInventory::Create(0, operations).ok());
}

}  // namespace
}  // namespace pih
