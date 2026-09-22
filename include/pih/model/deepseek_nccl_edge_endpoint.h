#pragma once

#include <memory>

#include "pih/model/deepseek_nccl_api_driver.h"

namespace pih {

class DeepSeekNcclEdgeEndpoint final {
 public:
  static Result<DeepSeekNcclEdgeEndpoint> Create(
      DeepSeekNcclCApi& api,
      std::unique_ptr<DeepSeekNcclBootstrapLease> bootstrap,
      DeepSeekNcclReleaseConfig config,
      DeepSeekNcclCommunicatorManifest manifest);

  DeepSeekNcclEdgeEndpoint(const DeepSeekNcclEdgeEndpoint&) = delete;
  DeepSeekNcclEdgeEndpoint& operator=(const DeepSeekNcclEdgeEndpoint&) = delete;
  DeepSeekNcclEdgeEndpoint(DeepSeekNcclEdgeEndpoint&&) noexcept = default;
  DeepSeekNcclEdgeEndpoint& operator=(
      DeepSeekNcclEdgeEndpoint&& other) noexcept;
  ~DeepSeekNcclEdgeEndpoint();

  Status accept_bootstrap(bool both_endpoints_validated);
  Status begin_init();
  Status poll_init();
  Status reconcile();
  Status mark_warmed(bool min_boundary_passed, bool max_boundary_passed);
  Status seal();
  Status begin_finalize();
  Status poll_finalize();
  Status destroy();
  Status abort();

  [[nodiscard]] DeepSeekBoundaryTransportDriver* transport() noexcept;
  [[nodiscard]] DeepSeekBoundaryTransportDriver* warmup_transport() noexcept;
  [[nodiscard]] bool bootstrap_zeroized() const noexcept;
  [[nodiscard]] DeepSeekNcclCommunicatorState state() const noexcept;
  [[nodiscard]] const DeepSeekNcclCommunicatorManifest& manifest()
      const noexcept;

 private:
  DeepSeekNcclEdgeEndpoint(
      std::unique_ptr<DeepSeekNcclBootstrapLease> bootstrap,
      std::unique_ptr<DeepSeekNcclApiDriver> driver,
      std::unique_ptr<DeepSeekNcclCommunicator> communicator) noexcept
      : bootstrap_(std::move(bootstrap)), driver_(std::move(driver)),
        communicator_(std::move(communicator)) {}

  // Driver borrows bootstrap. Communicator state is torn down before both.
  std::unique_ptr<DeepSeekNcclBootstrapLease> bootstrap_;
  std::unique_ptr<DeepSeekNcclApiDriver> driver_;
  std::unique_ptr<DeepSeekNcclCommunicator> communicator_;
};

}  // namespace pih
