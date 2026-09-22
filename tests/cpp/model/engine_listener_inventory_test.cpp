#include "pih/model/engine_listener_inventory.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest listener_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

std::array<EngineListenerBinding, 1> listener_bindings() {
  return {EngineListenerBinding{30, listener_digest(1), listener_digest(2),
                                9, 90, true}};
}

class ListenerOperations final : public EngineListenerOperations {
 public:
  Result<EngineListenerObservation> observe(
      std::uint64_t listener_identity) override {
    ++calls;
    if (!status.ok()) return status;
    return observation;
  }

  int calls = 0;
  Status status = Status::Ok();
  EngineListenerObservation observation{30, true, true, true, false, 9, 90};
};

TEST(EngineListenerInventoryTest, CapturesPassiveInheritedListener) {
  ListenerOperations operations;
  auto inventory = EngineListenerInventory::Create(
      listener_bindings(), operations).value();

  auto records = inventory.capture(EngineOwnedResourceKind::kListener);
  ASSERT_TRUE(records.ok());
  ASSERT_EQ(records->size(), 1U);
  EXPECT_EQ(records->at(0).state,
            EngineOwnedResourceLifecycleState::kPassive);
  EXPECT_EQ(records->at(0).owner_identity, listener_digest(1));
  EXPECT_EQ(records->at(0).backing_bytes, 0U);
}

TEST(EngineListenerInventoryTest, CapturesAcceptAuthorityAsActive) {
  ListenerOperations operations;
  operations.observation.accept_authority = true;
  auto inventory = EngineListenerInventory::Create(
      listener_bindings(), operations).value();
  auto records = inventory.capture(EngineOwnedResourceKind::kListener);
  ASSERT_TRUE(records.ok());
  EXPECT_EQ(records->at(0).state,
            EngineOwnedResourceLifecycleState::kActive);
}

TEST(EngineListenerInventoryTest, OmitsClosedListener) {
  ListenerOperations operations;
  operations.observation.open = false;
  auto inventory = EngineListenerInventory::Create(
      listener_bindings(), operations).value();
  EXPECT_TRUE(inventory.capture(EngineOwnedResourceKind::kListener)->empty());
}

TEST(EngineListenerInventoryTest, RejectsSocketAndAuthorityDrift) {
  for (int mutation = 0; mutation < 6; ++mutation) {
    ListenerOperations operations;
    if (mutation == 0) ++operations.observation.listener_identity;
    if (mutation == 1) operations.observation.socket = false;
    if (mutation == 2) operations.observation.kernel_listening = false;
    if (mutation == 3) ++operations.observation.device_identity;
    if (mutation == 4) ++operations.observation.inode_identity;
    if (mutation == 5) {
      operations.observation.kernel_listening = false;
      operations.observation.accept_authority = true;
    }
    auto inventory = EngineListenerInventory::Create(
        listener_bindings(), operations).value();
    auto records = inventory.capture(EngineOwnedResourceKind::kListener);
    ASSERT_FALSE(records.ok()) << mutation;
    EXPECT_EQ(records.status().code(), StatusCode::kFailedPrecondition)
        << mutation;
  }
}

TEST(EngineListenerInventoryTest, PreservesUnavailableAndKindBoundary) {
  ListenerOperations operations;
  auto inventory = EngineListenerInventory::Create(
      listener_bindings(), operations).value();
  operations.status = Status::Unavailable("listener authority unavailable");
  auto unavailable = inventory.capture(EngineOwnedResourceKind::kListener);
  ASSERT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), StatusCode::kUnavailable);
  operations.status = Status::Ok();
  EXPECT_TRUE(inventory.capture(EngineOwnedResourceKind::kListener).ok());
  EXPECT_FALSE(inventory.capture(EngineOwnedResourceKind::kNetwork).ok());
}

TEST(EngineListenerInventoryTest, RejectsInvalidManifest) {
  ListenerOperations operations;
  auto bindings = listener_bindings();
  bindings[0].listener_identity = 0;
  EXPECT_FALSE(EngineListenerInventory::Create(bindings, operations).ok());
  bindings = listener_bindings();
  bindings[0].expected_inode_identity = 0;
  EXPECT_FALSE(EngineListenerInventory::Create(bindings, operations).ok());
}

}  // namespace
}  // namespace pih
