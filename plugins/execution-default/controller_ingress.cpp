#include "pih/scheduler/controller_ingress.h"

#include <array>

namespace {
void write_u64(std::array<std::byte, 8>& wire, std::uint64_t value) {
  for (std::size_t i = 0; i < wire.size(); ++i)
    wire[i] = static_cast<std::byte>(value >> (i * 8U));
}
}

namespace pih {

Result<Sha256Digest> controller_cancel_payload_digest(
    std::uint64_t epoch, std::uint64_t request_generation) {
  if (epoch == 0 || request_generation == 0)
    return Status::InvalidArgument("controller cancel identity is invalid");
  constexpr std::string_view domain = "pih-controller-cancel-v1";
  Sha256 digest;
  auto status = digest.update(std::as_bytes(std::span(domain)));
  std::array<std::byte, 8> wire{};
  write_u64(wire, epoch);
  if (status.ok()) status = digest.update(wire);
  write_u64(wire, request_generation);
  if (status.ok()) status = digest.update(wire);
  if (!status.ok()) return status;
  return digest.finalize();
}

Result<ControllerIngress> ControllerIngress::Create(
    ControllerIngressLimits limits) {
  auto mailbox = ControllerMailbox::Create(limits.mailbox);
  if (!mailbox.ok()) return mailbox.status();
  auto requests = ControllerRequestArena::Create(limits.requests);
  if (!requests.ok()) return requests.status();
  return ControllerIngress(std::move(*mailbox), std::move(*requests));
}

Result<ControllerCommand> ControllerIngress::submit_admit(
    std::uint64_t epoch, std::uint64_t request_generation,
    std::span<const std::uint32_t> prompt_token_ids,
    std::uint32_t maximum_new_tokens,
    const ControllerRequestSampling& sampling) {
  if (epoch == 0) {
    return Status::InvalidArgument("controller admission epoch is invalid");
  }
  auto receipt = requests_.publish(request_generation, prompt_token_ids,
                                   maximum_new_tokens, sampling);
  if (!receipt.ok()) return receipt.status();
  auto command = mailbox_.submit_command(
      ControllerCommandKind::kAdmit, epoch, request_generation,
      receipt->payload_digest);
  if (!command.ok()) {
    const auto rollback = requests_.rollback_published(*receipt);
    if (!rollback.ok()) {
      return Status::Internal("controller admission rollback failed");
    }
    return command.status();
  }
  return *command;
}

Result<ControllerCommand> ControllerIngress::submit_cancel(
    std::uint64_t epoch, std::uint64_t request_generation) {
  auto digest = controller_cancel_payload_digest(epoch, request_generation);
  if (!digest.ok()) return digest.status();
  return mailbox_.submit_command(ControllerCommandKind::kCancel, epoch,
                                 request_generation, *digest);
}

Result<std::optional<ControllerIngressCommandView>>
ControllerIngress::try_take_command() {
  auto command = mailbox_.try_take_command();
  if (!command.ok()) return command.status();
  if (!command->has_value())
    return std::optional<ControllerIngressCommandView>{};
  if ((**command).kind == ControllerCommandKind::kAdmit) {
    auto request = requests_.claim((**command).request_generation,
                                   (**command).payload_digest);
    if (!request.ok())
      return Status::Internal(
          "controller command has no matching request payload");
    return std::optional<ControllerIngressCommandView>{
        ControllerIngressCommandView{**command, *request}};
  }
  return std::optional<ControllerIngressCommandView>{
      ControllerIngressCommandView{**command, std::nullopt}};
}

Result<std::optional<ControllerAdmissionView>>
ControllerIngress::try_take_admission() {
  auto item = try_take_command();
  if (!item.ok()) return item.status();
  if (!item->has_value()) return std::optional<ControllerAdmissionView>{};
  if ((**item).command.kind != ControllerCommandKind::kAdmit ||
      !(**item).request.has_value()) {
    return Status::Internal("non-admit command reached admission ingress");
  }
  return std::optional<ControllerAdmissionView>{
      ControllerAdmissionView{(**item).command, *(**item).request}};
}

Status ControllerIngress::release_request(
    const ControllerRequestView& request) {
  return requests_.release_claimed(request.slot_index,
                                   request.slot_generation);
}

Result<ControllerOutputEvent> ControllerIngress::publish_event(
    const ControllerOutputEventDraft& draft) {
  return mailbox_.publish_event(draft);
}

Result<std::optional<ControllerOutputEvent>>
ControllerIngress::try_take_event() {
  return mailbox_.try_take_event();
}

std::uint32_t ControllerIngress::queued_command_count() const {
  return mailbox_.command_size();
}

std::uint32_t ControllerIngress::published_request_count() const {
  return requests_.published_count();
}

std::uint32_t ControllerIngress::claimed_request_count() const {
  return requests_.claimed_count();
}

std::uint32_t ControllerIngress::available_event_credit() const {
  return mailbox_.event_available_capacity();
}

}  // namespace pih
