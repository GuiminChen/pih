#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include "pih/scheduler/controller_mailbox.h"
#include "pih/scheduler/controller_request_arena.h"

namespace pih {

struct ControllerIngressLimits final {
  ControllerMailboxLimits mailbox;
  ControllerRequestArenaLimits requests;
};

struct ControllerAdmissionView final {
  ControllerCommand command;
  ControllerRequestView request;
};

struct ControllerIngressCommandView final {
  ControllerCommand command;
  std::optional<ControllerRequestView> request;
};

Result<Sha256Digest> controller_cancel_payload_digest(
    std::uint64_t epoch, std::uint64_t request_generation);

class ControllerIngress final {
 public:
  static constexpr std::string_view kAbi = "controller_ingress_transaction_v1";

  static Result<ControllerIngress> Create(ControllerIngressLimits limits);
  ControllerIngress(const ControllerIngress&) = delete;
  ControllerIngress& operator=(const ControllerIngress&) = delete;
  ControllerIngress(ControllerIngress&&) noexcept = default;
  ControllerIngress& operator=(ControllerIngress&&) noexcept = delete;

  Result<ControllerCommand> submit_admit(
      std::uint64_t epoch, std::uint64_t request_generation,
      std::span<const std::uint32_t> prompt_token_ids,
      std::uint32_t maximum_new_tokens,
      const ControllerRequestSampling& sampling = {});
  Result<ControllerCommand> submit_cancel(
      std::uint64_t epoch, std::uint64_t request_generation);
  Result<std::optional<ControllerIngressCommandView>> try_take_command();
  Result<std::optional<ControllerAdmissionView>> try_take_admission();
  Status release_request(const ControllerRequestView& request);

  Result<ControllerOutputEvent> publish_event(
      const ControllerOutputEventDraft& draft);
  Result<std::optional<ControllerOutputEvent>> try_take_event();

  [[nodiscard]] std::uint32_t queued_command_count() const;
  [[nodiscard]] std::uint32_t published_request_count() const;
  [[nodiscard]] std::uint32_t claimed_request_count() const;
  [[nodiscard]] std::uint32_t available_event_credit() const;
  [[nodiscard]] std::uint32_t queued_event_count() const {
    return mailbox_.event_size();
  }

 private:
  ControllerIngress(ControllerMailbox mailbox,
                    ControllerRequestArena requests)
      : mailbox_(std::move(mailbox)), requests_(std::move(requests)) {}

  ControllerMailbox mailbox_;
  ControllerRequestArena requests_;
};

}  // namespace pih
