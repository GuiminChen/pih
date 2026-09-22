#include "pih/model/engine_network_port_owner_classifier.h"

#include <array>
#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest port_owner_digest(std::uint8_t value) {
  Sha256Digest digest{};
  digest.bytes.fill(static_cast<std::byte>(value));
  return digest;
}

class PortBackend final : public EngineNetworkPortCensusBackend {
 public:
  Result<EngineNetworkPortRawSnapshot> capture(
      std::uint64_t namespace_identity) override {
    ++calls;
    last_identity = namespace_identity;
    return result;
  }

  int calls = 0;
  std::uint64_t last_identity = 0;
  Result<EngineNetworkPortRawSnapshot> result =
      EngineNetworkPortRawSnapshot{
          50, true, false,
          {{40000, EngineNetworkSocketLifecycle::kActive, true, 100},
           {40001, EngineNetworkSocketLifecycle::kTimeWait, true, 80},
           {41000, EngineNetworkSocketLifecycle::kActive, true, 120}}};
};

std::array<EngineNetworkPortOwnerBinding, 2> port_bindings() {
  return {EngineNetworkPortOwnerBinding{40000, 40099,
                                        port_owner_digest(1),
                                        port_owner_digest(2)},
          EngineNetworkPortOwnerBinding{41000, 41099,
                                        port_owner_digest(1),
                                        port_owner_digest(3)}};
}

TEST(EngineNetworkPortOwnerClassifierTest, ClassifiesActiveAndTimeWaitRows) {
  PortBackend backend;
  auto classifier = EngineNetworkPortOwnerClassifier::Create(
      50, port_bindings(), backend).value();
  auto snapshot = classifier.capture(50);
  ASSERT_TRUE(snapshot.ok());
  ASSERT_EQ(snapshot->rows.size(), 3U);
  EXPECT_TRUE(snapshot->rows[0].ownership_complete);
  EXPECT_EQ(snapshot->rows[0].owner_identity, port_owner_digest(1));
  EXPECT_EQ(snapshot->rows[0].resource_identity, port_owner_digest(2));
  EXPECT_EQ(snapshot->rows[1].lifecycle,
            EngineNetworkSocketLifecycle::kTimeWait);
  EXPECT_EQ(snapshot->rows[2].resource_identity, port_owner_digest(3));
}

TEST(EngineNetworkPortOwnerClassifierTest, MarksUnleasedPortUnknown) {
  PortBackend backend;
  backend.result->rows.push_back(
      {42000, EngineNetworkSocketLifecycle::kActive, true, 1});
  auto classifier = EngineNetworkPortOwnerClassifier::Create(
      50, port_bindings(), backend).value();
  auto snapshot = classifier.capture(50);
  ASSERT_TRUE(snapshot.ok());
  EXPECT_FALSE(snapshot->rows.back().ownership_complete);
  EXPECT_EQ(snapshot->rows.back().owner_identity, Sha256Digest{});
}

TEST(EngineNetworkPortOwnerClassifierTest, PreservesDestroyedNamespace) {
  PortBackend backend;
  backend.result = EngineNetworkPortRawSnapshot{50, false, true, {}};
  auto classifier = EngineNetworkPortOwnerClassifier::Create(
      50, port_bindings(), backend).value();
  auto snapshot = classifier.capture(50);
  ASSERT_TRUE(snapshot.ok());
  EXPECT_TRUE(snapshot->namespace_destroyed);
  EXPECT_TRUE(snapshot->rows.empty());
}

TEST(EngineNetworkPortOwnerClassifierTest, RejectsInvalidRawRows) {
  for (int mutation = 0; mutation < 4; ++mutation) {
    PortBackend backend;
    auto snapshot = *backend.result;
    if (mutation == 0) ++snapshot.namespace_identity;
    if (mutation == 1) snapshot.namespace_destroyed = true;
    if (mutation == 2) snapshot.rows[0].local_port = 0;
    if (mutation == 3)
      snapshot.rows[0].lifecycle =
          static_cast<EngineNetworkSocketLifecycle>(2);
    backend.result = snapshot;
    auto classifier = EngineNetworkPortOwnerClassifier::Create(
        50, port_bindings(), backend).value();
    EXPECT_FALSE(classifier.capture(50).ok()) << mutation;
  }
}

TEST(EngineNetworkPortOwnerClassifierTest, RejectsOverlappingManifest) {
  PortBackend backend;
  auto bindings = port_bindings();
  bindings[1].first_port = 40050;
  EXPECT_FALSE(EngineNetworkPortOwnerClassifier::Create(
                   50, bindings, backend).ok());
  bindings = port_bindings();
  bindings[0].first_port = 0;
  EXPECT_FALSE(EngineNetworkPortOwnerClassifier::Create(
                   50, bindings, backend).ok());
  bindings = port_bindings();
  bindings[0].owner_identity = {};
  EXPECT_FALSE(EngineNetworkPortOwnerClassifier::Create(
                   50, bindings, backend).ok());
}

}  // namespace
}  // namespace pih
