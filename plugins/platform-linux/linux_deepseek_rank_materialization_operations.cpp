#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "pih/platform/linux/linux_deepseek_rank_materialization_operations.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

namespace pih {
namespace {

Status system_failure(const char* operation) {
  return Status::Internal(std::string(operation) + " failed with errno " +
                          std::to_string(errno));
}

bool same_handle(const DeepSeekRankProcessHandle& left,
                 const DeepSeekRankProcessHandle& right) noexcept {
  return left.process_identity == right.process_identity &&
         left.pidfd_identity == right.pidfd_identity &&
         left.control_identity == right.control_identity;
}

Status validate_controller_socket(int descriptor) {
  int socket_type = 0;
  socklen_t type_size = sizeof(socket_type);
  int pass_credentials = 0;
  socklen_t pass_size = sizeof(pass_credentials);
  if (descriptor < 0 ||
      ::getsockopt(descriptor, SOL_SOCKET, SO_TYPE, &socket_type,
                   &type_size) != 0 ||
      type_size != sizeof(socket_type) || socket_type != SOCK_SEQPACKET ||
      ::getsockopt(descriptor, SOL_SOCKET, SO_PASSCRED, &pass_credentials,
                   &pass_size) != 0 ||
      pass_size != sizeof(pass_credentials) || pass_credentials != 1) {
    return Status::FailedPrecondition(
        "DeepSeek materialization controller socket is not credentialed seqpacket");
  }
  return Status::Ok();
}

Status validate_worker_socket(int descriptor,
                              std::uint64_t expected_controller) {
  if (expected_controller == 0 ||
      expected_controller >
          static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    return Status::InvalidArgument(
        "DeepSeek materialization controller process identity is invalid");
  }
  auto status = validate_controller_socket(descriptor);
  if (!status.ok()) return status;
  struct ucred peer {};
  socklen_t peer_size = sizeof(peer);
  if (::getsockopt(descriptor, SOL_SOCKET, SO_PEERCRED, &peer,
                   &peer_size) != 0 ||
      peer_size != sizeof(peer) ||
      peer.pid != static_cast<pid_t>(expected_controller) ||
      peer.uid != ::geteuid() || peer.gid != ::getegid()) {
    return Status::FailedPrecondition(
        "DeepSeek materialization worker socket peer differs");
  }
  return Status::Ok();
}

struct CredentialObservation final {
  std::uint32_t count = 0;
  struct ucred credential {};
};

Status collect_credential_without_rights(
    msghdr& message, CredentialObservation* result) {
  if (result == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek materialization credential output is absent");
  }
  bool invalid = false;
  for (auto* header = CMSG_FIRSTHDR(&message); header != nullptr;
       header = CMSG_NXTHDR(&message, header)) {
    if (header->cmsg_level == SOL_SOCKET &&
        header->cmsg_type == SCM_RIGHTS &&
        header->cmsg_len >= CMSG_LEN(sizeof(int))) {
      const auto payload_bytes = header->cmsg_len - CMSG_LEN(0);
      const auto count = payload_bytes / sizeof(int);
      const auto* descriptors =
          reinterpret_cast<const int*>(CMSG_DATA(header));
      for (std::size_t index = 0; index < count; ++index) {
        if (descriptors[index] >= 0) (void)::close(descriptors[index]);
      }
      invalid = true;
      continue;
    }
    if (header->cmsg_level != SOL_SOCKET ||
        header->cmsg_type != SCM_CREDENTIALS ||
        header->cmsg_len != CMSG_LEN(sizeof(struct ucred))) {
      invalid = true;
      continue;
    }
    ++result->count;
    std::memcpy(&result->credential, CMSG_DATA(header),
                sizeof(struct ucred));
  }
  if (invalid) {
    return Status::FailedPrecondition(
        "DeepSeek materialization ancillary record is invalid");
  }
  return Status::Ok();
}

Result<std::vector<std::byte>> receive_credentialed_packet(
    int socket, std::size_t expected_bytes,
    std::uint64_t expected_process_identity) {
  if (expected_process_identity == 0 ||
      expected_process_identity >
          static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    return Status::InvalidArgument(
        "DeepSeek materialization packet process identity is invalid");
  }
  std::vector<std::byte> frame(expected_bytes);
  iovec vector{frame.data(), frame.size()};
  alignas(cmsghdr) std::array<
      std::byte,
      CMSG_SPACE(sizeof(struct ucred)) + CMSG_SPACE(sizeof(int) * 16U)>
      control{};
  msghdr message{};
  message.msg_iov = &vector;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  const auto received = ::recvmsg(
      socket, &message,
      MSG_DONTWAIT | MSG_TRUNC | MSG_CMSG_CLOEXEC);
  if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek materialization packet is pending");
  }
  if (received == 0) {
    return Status::FailedPrecondition(
        "DeepSeek materialization control socket closed");
  }
  if (received < 0) {
    return system_failure("recvmsg materialization packet");
  }
  CredentialObservation credential;
  auto status = collect_credential_without_rights(message, &credential);
  if (!status.ok()) return status;
  if ((message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0 ||
      received != static_cast<ssize_t>(expected_bytes) ||
      credential.count != 1 ||
      credential.credential.pid !=
          static_cast<pid_t>(expected_process_identity) ||
      credential.credential.uid != ::geteuid() ||
      credential.credential.gid != ::getegid()) {
    return Status::FailedPrecondition(
        "DeepSeek materialization packet identity is invalid");
  }
  return frame;
}

Result<std::uint64_t> monotonic_now() {
  timespec value{};
  if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0 ||
      value.tv_sec < 0 || value.tv_nsec < 0 ||
      value.tv_nsec >= 1'000'000'000L) {
    return system_failure("clock_gettime materialization");
  }
  return static_cast<std::uint64_t>(value.tv_sec) *
             UINT64_C(1'000'000'000) +
         static_cast<std::uint64_t>(value.tv_nsec);
}

}  // namespace

Result<LinuxDeepSeekRankMaterializationControllerOperations>
LinuxDeepSeekRankMaterializationControllerOperations::Create(
    LinuxDeepSeekRankProcessDriver& driver,
    DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessHandle> ordered_handles) {
  if (!supervisor.ready() || supervisor.failed() ||
      ordered_handles.empty() || ordered_handles.size() > 4) {
    return Status::FailedPrecondition(
        "DeepSeek materialization controller lacks a ready process set");
  }
  if (ordered_handles.size() < 4 &&
      (supervisor.process_handle(static_cast<std::uint32_t>(
           ordered_handles.size())) != nullptr ||
       supervisor.exec_ready(static_cast<std::uint32_t>(
           ordered_handles.size())) != nullptr)) {
    return Status::FailedPrecondition(
        "DeepSeek materialization controller process set is truncated");
  }
  for (std::uint32_t rank = 0; rank < ordered_handles.size(); ++rank) {
    const auto* supervised = supervisor.process_handle(rank);
    const auto* ready = supervisor.exec_ready(rank);
    if (supervised == nullptr || ready == nullptr ||
        !same_handle(*supervised, ordered_handles[rank]) ||
        ready->receipt.rank != rank ||
        ready->receipt.process_identity !=
            ordered_handles[rank].process_identity) {
      return Status::FailedPrecondition(
          "DeepSeek materialization controller process identity differs");
    }
    auto* process = driver.find(ordered_handles[rank]);
    if (process == nullptr) {
      return Status::FailedPrecondition(
          "DeepSeek materialization driver handle differs");
    }
    auto status = validate_controller_socket(process->control);
    if (!status.ok()) return status;
  }
  const auto* first_ready = supervisor.exec_ready(0);
  if (first_ready == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek materialization controller generation is absent");
  }
  return LinuxDeepSeekRankMaterializationControllerOperations(
      driver, supervisor,
      std::vector<DeepSeekRankProcessHandle>(ordered_handles.begin(),
                                             ordered_handles.end()),
      first_ready->receipt.engine_epoch,
      first_ready->receipt.worker_generation);
}

Result<std::uint32_t>
LinuxDeepSeekRankMaterializationControllerOperations::rank_for(
    const DeepSeekRankProcessHandle& handle) const {
  for (std::uint32_t rank = 0; rank < handles_.size(); ++rank) {
    if (same_handle(handles_[rank], handle)) return rank;
  }
  return Status::FailedPrecondition(
      "DeepSeek materialization process handle is unknown");
}

Result<std::int32_t>
LinuxDeepSeekRankMaterializationControllerOperations::control_fd(
    std::uint32_t rank) const {
  if (rank >= handles_.size()) {
    return Status::InvalidArgument(
        "DeepSeek materialization controller rank is invalid");
  }
  auto* process = driver_->find(handles_[rank]);
  if (process == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek materialization controller handle disappeared");
  }
  auto status = validate_controller_socket(process->control);
  if (!status.ok()) return status;
  return process->control;
}

Status LinuxDeepSeekRankMaterializationControllerOperations::send_grant(
    const DeepSeekRankProcessHandle& handle,
    std::span<const std::byte> frame) {
  auto rank = rank_for(handle);
  if (!rank.ok()) return rank.status();
  auto grant = decode_deepseek_rank_materialization_grant(frame);
  if (!grant.ok()) return grant.status();
  if (grant->rank != *rank ||
      grant->process_identity != handle.process_identity ||
      grant->pidfd_identity != handle.pidfd_identity ||
      grant->control_identity != handle.control_identity) {
    return Status::FailedPrecondition(
        "DeepSeek materialization grant handle differs");
  }
  auto socket = control_fd(*rank);
  if (!socket.ok()) return socket.status();
  const auto written = ::send(
      *socket, frame.data(), frame.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek materialization grant socket is backpressured");
  }
  if (written < 0) return system_failure("send materialization grant");
  if (written != static_cast<ssize_t>(frame.size())) {
    return Status::Internal(
        "DeepSeek materialization grant was partially sent");
  }
  return Status::Ok();
}

Result<std::optional<std::vector<std::byte>>>
LinuxDeepSeekRankMaterializationControllerOperations::poll_ack(
    const DeepSeekRankProcessHandle& handle) {
  auto rank = rank_for(handle);
  if (!rank.ok()) return rank.status();
  auto socket = control_fd(*rank);
  if (!socket.ok()) return socket.status();
  auto frame = receive_credentialed_packet(
      *socket, kDeepSeekRankMaterializationGrantAckFrameBytes,
      handle.process_identity);
  if (!frame.ok()) {
    if (frame.status().code() == StatusCode::kUnavailable) {
      return std::optional<std::vector<std::byte>>{};
    }
    return frame.status();
  }
  auto ack = decode_deepseek_rank_materialization_grant_ack(*frame);
  if (!ack.ok()) return ack.status();
  if (ack->rank != *rank ||
      ack->process_identity != handle.process_identity ||
      ack->pidfd_identity != handle.pidfd_identity ||
      ack->control_identity != handle.control_identity) {
    return Status::FailedPrecondition(
        "DeepSeek materialization ACK handle differs");
  }
  return std::optional<std::vector<std::byte>>{std::move(*frame)};
}

Result<std::optional<std::vector<std::byte>>>
LinuxDeepSeekRankMaterializationControllerOperations::poll_completion(
    const DeepSeekRankProcessHandle& handle) {
  auto rank = rank_for(handle);
  if (!rank.ok()) return rank.status();
  auto socket = control_fd(*rank);
  if (!socket.ok()) return socket.status();
  auto frame = receive_credentialed_packet(
      *socket, kDeepSeekRankMaterializationCompletionFrameBytes,
      handle.process_identity);
  if (!frame.ok()) {
    if (frame.status().code() == StatusCode::kUnavailable) {
      return std::optional<std::vector<std::byte>>{};
    }
    return frame.status();
  }
  auto completion =
      decode_deepseek_rank_materialization_completion(*frame);
  if (!completion.ok()) return completion.status();
  if (completion->rank != *rank ||
      completion->process_identity != handle.process_identity ||
      completion->pidfd_identity != handle.pidfd_identity ||
      completion->control_identity != handle.control_identity) {
    return Status::FailedPrecondition(
        "DeepSeek materialization completion handle differs");
  }
  return std::optional<std::vector<std::byte>>{std::move(*frame)};
}

Result<std::uint64_t>
LinuxDeepSeekRankMaterializationControllerOperations::monotonic_now_ns() {
  return monotonic_now();
}

Status LinuxDeepSeekRankMaterializationControllerOperations::
    abort_generation(std::uint64_t engine_epoch,
                     std::uint64_t worker_generation,
                     const Status& cause) {
  if (engine_epoch != engine_epoch_ ||
      worker_generation != worker_generation_) {
    return Status::FailedPrecondition(
        "DeepSeek materialization warmup abort identity differs");
  }
  if (supervisor_->failed()) return Status::Ok();
  return supervisor_->abort_post_exec(cause);
}

Result<LinuxDeepSeekRankMaterializationReceiverOperations>
LinuxDeepSeekRankMaterializationReceiverOperations::Create(
    std::int32_t control_fd,
    std::uint64_t expected_controller_process_identity) {
  auto status = validate_worker_socket(
      control_fd, expected_controller_process_identity);
  if (!status.ok()) return status;
  return LinuxDeepSeekRankMaterializationReceiverOperations(
      control_fd, expected_controller_process_identity);
}

Status LinuxDeepSeekRankMaterializationReceiverOperations::
    validate_bound_control(std::int32_t control_fd) const {
  if (control_fd != control_fd_) {
    return Status::FailedPrecondition(
        "DeepSeek materialization worker control descriptor changed");
  }
  return validate_worker_socket(
      control_fd_, controller_process_identity_);
}

Result<std::optional<std::vector<std::byte>>>
LinuxDeepSeekRankMaterializationReceiverOperations::receive_grant(
    std::int32_t control_fd) {
  auto status = validate_bound_control(control_fd);
  if (!status.ok()) return status;
  auto frame = receive_credentialed_packet(
      control_fd_, kDeepSeekRankMaterializationGrantFrameBytes,
      controller_process_identity_);
  if (!frame.ok()) {
    if (frame.status().code() == StatusCode::kUnavailable) {
      return std::optional<std::vector<std::byte>>{};
    }
    return frame.status();
  }
  auto grant = decode_deepseek_rank_materialization_grant(*frame);
  if (!grant.ok()) return grant.status();
  return std::optional<std::vector<std::byte>>{std::move(*frame)};
}

Result<std::uint64_t>
LinuxDeepSeekRankMaterializationReceiverOperations::monotonic_now_ns() {
  return monotonic_now();
}

Status LinuxDeepSeekRankMaterializationReceiverOperations::send_ack(
    std::int32_t control_fd, std::span<const std::byte> frame) {
  auto status = validate_bound_control(control_fd);
  if (!status.ok()) return status;
  auto ack = decode_deepseek_rank_materialization_grant_ack(frame);
  if (!ack.ok()) return ack.status();
  const auto written = ::send(
      control_fd_, frame.data(), frame.size(),
      MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek materialization ACK socket is backpressured");
  }
  if (written < 0) return system_failure("send materialization ACK");
  if (written != static_cast<ssize_t>(frame.size())) {
    return Status::Internal(
        "DeepSeek materialization ACK was partially sent");
  }
  return Status::Ok();
}

Status LinuxDeepSeekRankMaterializationReceiverOperations::send_completion(
    std::int32_t control_fd, std::span<const std::byte> frame) {
  auto status = validate_bound_control(control_fd);
  if (!status.ok()) return status;
  auto completion =
      decode_deepseek_rank_materialization_completion(frame);
  if (!completion.ok()) return completion.status();
  const auto written = ::send(
      control_fd_, frame.data(), frame.size(),
      MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek materialization completion socket is backpressured");
  }
  if (written < 0) {
    return system_failure("send materialization completion");
  }
  if (written != static_cast<ssize_t>(frame.size())) {
    return Status::Internal(
        "DeepSeek materialization completion was partially sent");
  }
  return Status::Ok();
}

Result<LinuxDeepSeekRankServingWorkerOperations>
LinuxDeepSeekRankServingWorkerOperations::Create(
    std::int32_t control_fd,
    std::uint64_t expected_controller_process_identity) {
  const auto status = validate_worker_socket(
      control_fd, expected_controller_process_identity);
  if (!status.ok()) return status;
  return LinuxDeepSeekRankServingWorkerOperations(
      control_fd, expected_controller_process_identity);
}

Status LinuxDeepSeekRankServingWorkerOperations::validate_bound_control()
    const {
  return validate_worker_socket(control_fd_, controller_process_identity_);
}

Result<std::optional<std::vector<std::byte>>>
LinuxDeepSeekRankServingWorkerOperations::receive_command() {
  auto status = validate_bound_control();
  if (!status.ok()) return status;
  auto frame = receive_credentialed_packet(
      control_fd_, kDeepSeekRankServingCommandBytes,
      controller_process_identity_);
  if (!frame.ok()) {
    if (frame.status().code() == StatusCode::kUnavailable) {
      return std::optional<std::vector<std::byte>>{};
    }
    return frame.status();
  }
  auto command = decode_deepseek_rank_serving_command(*frame);
  if (!command.ok()) return command.status();
  return std::optional<std::vector<std::byte>>{std::move(*frame)};
}

Status LinuxDeepSeekRankServingWorkerOperations::send_completion(
    std::span<const std::byte> frame) {
  auto status = validate_bound_control();
  if (!status.ok()) return status;
  auto completion = decode_deepseek_rank_serving_completion(frame);
  if (!completion.ok()) return completion.status();
  const auto written = ::send(control_fd_, frame.data(), frame.size(),
                              MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable(
        "DeepSeek rank serving completion socket is backpressured");
  }
  if (written < 0) return system_failure("send DeepSeek rank serving completion");
  if (written != static_cast<ssize_t>(frame.size())) {
    return Status::Internal(
        "DeepSeek rank serving completion was partially sent");
  }
  return Status::Ok();
}

Result<LinuxDeepSeekRankServingWorkerTransport>
LinuxDeepSeekRankServingWorkerTransport::Create(
    std::int32_t control_fd,
    std::uint64_t expected_controller_process_identity) {
  auto operations = LinuxDeepSeekRankServingWorkerOperations::Create(
      control_fd, expected_controller_process_identity);
  if (!operations.ok()) return operations.status();
  return LinuxDeepSeekRankServingWorkerTransport(std::move(*operations));
}

Result<std::optional<DeepSeekRankServingCommand>>
LinuxDeepSeekRankServingWorkerTransport::receive() {
  auto frame = operations_.receive_command();
  if (!frame.ok()) return frame.status();
  if (!frame->has_value()) return std::optional<DeepSeekRankServingCommand>{};
  auto command = decode_deepseek_rank_serving_command(frame->value());
  if (!command.ok()) return command.status();
  return std::optional<DeepSeekRankServingCommand>(std::move(*command));
}

Status LinuxDeepSeekRankServingWorkerTransport::send(
    const DeepSeekRankServingCompletion& completion) {
  const auto frame = encode_deepseek_rank_serving_completion(completion);
  return operations_.send_completion(frame);
}

Result<LinuxDeepSeekRankServingWorker> LinuxDeepSeekRankServingWorker::Create(
    DeepSeekRankServingSessionBinding session, std::int32_t control_fd,
    std::uint64_t expected_controller_process_identity,
    DeepSeekRankServingPlanExecutionFactory& factory,
    DeepSeekRankServingLeaseResolver& resolver) {
  auto transport = LinuxDeepSeekRankServingWorkerTransport::Create(
      control_fd, expected_controller_process_identity);
  if (!transport.ok()) return transport.status();
  auto executor = DeepSeekRankServingPlanWorkerExecutor::Create(factory, resolver);
  if (!executor.ok()) return executor.status();
  auto gate = DeepSeekRankServingWorkerGate::Create(std::move(session));
  if (!gate.ok()) return gate.status();
  auto loop = DeepSeekRankServingWorkerLoop::Create(
      std::move(*gate), *transport, *executor);
  if (!loop.ok()) return loop.status();
  return LinuxDeepSeekRankServingWorker(
      std::move(*transport), std::move(*executor), std::move(*loop));
}

Status LinuxDeepSeekRankServingWorker::advance() { return loop_.advance(); }

Result<LinuxDeepSeekRankServingControllerOperations>
LinuxDeepSeekRankServingControllerOperations::Create(
    LinuxDeepSeekRankProcessDriver& driver,
    DeepSeekRankProcessSupervisor& supervisor,
    std::span<const DeepSeekRankProcessHandle> ordered_handles) {
  if (!supervisor.ready() || supervisor.failed() || ordered_handles.empty() ||
      ordered_handles.size() > 4) {
    return Status::FailedPrecondition(
        "DeepSeek rank serving controller lacks a ready process set");
  }
  const auto* first = supervisor.exec_ready(0);
  if (first == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek rank serving controller generation is absent");
  }
  if (ordered_handles.size() < 4 &&
      (supervisor.process_handle(static_cast<std::uint32_t>(
           ordered_handles.size())) != nullptr ||
       supervisor.exec_ready(static_cast<std::uint32_t>(
           ordered_handles.size())) != nullptr)) {
    return Status::FailedPrecondition(
        "DeepSeek rank serving controller process set is truncated");
  }
  for (std::uint32_t rank = 0; rank < ordered_handles.size(); ++rank) {
    const auto* handle = supervisor.process_handle(rank);
    const auto* ready = supervisor.exec_ready(rank);
    auto* process = driver.find(ordered_handles[rank]);
    if (handle == nullptr || ready == nullptr || process == nullptr ||
        !same_handle(*handle, ordered_handles[rank]) ||
        ready->receipt.rank != rank ||
        ready->receipt.engine_epoch != first->receipt.engine_epoch ||
        ready->receipt.worker_generation != first->receipt.worker_generation ||
        ready->receipt.process_identity != ordered_handles[rank].process_identity ||
        !validate_controller_socket(process->control).ok()) {
      return Status::FailedPrecondition(
          "DeepSeek rank serving controller process identity differs");
    }
  }
  return LinuxDeepSeekRankServingControllerOperations(
      driver, supervisor,
      std::vector<DeepSeekRankProcessHandle>(ordered_handles.begin(),
                                             ordered_handles.end()),
      first->receipt.engine_epoch, first->receipt.worker_generation);
}

Result<std::int32_t> LinuxDeepSeekRankServingControllerOperations::control_fd(
    std::uint32_t rank) const {
  if (rank >= handles_.size()) {
    return Status::InvalidArgument("DeepSeek rank serving controller rank is invalid");
  }
  auto* process = driver_->find(handles_[rank]);
  if (process == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek rank serving controller handle disappeared");
  }
  auto status = validate_controller_socket(process->control);
  if (!status.ok()) return status;
  return process->control;
}

Status LinuxDeepSeekRankServingControllerOperations::send_command(
    std::uint32_t rank, std::span<const std::byte> frame) {
  auto command = decode_deepseek_rank_serving_command(frame);
  if (!command.ok()) return command.status();
  if (rank >= handles_.size() || command->session.rank != rank ||
      command->session.engine_epoch != engine_epoch_ ||
      command->session.worker_generation != worker_generation_ ||
      command->session.world_size != handles_.size() ||
      command->session.process_identity != handles_[rank].process_identity ||
      command->session.pidfd_identity != handles_[rank].pidfd_identity ||
      command->session.control_identity != handles_[rank].control_identity) {
    return Status::FailedPrecondition(
        "DeepSeek rank serving command handle differs");
  }
  auto socket = control_fd(rank);
  if (!socket.ok()) return socket.status();
  const auto written = ::send(*socket, frame.data(), frame.size(),
                              MSG_NOSIGNAL | MSG_DONTWAIT);
  if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return Status::Unavailable("DeepSeek rank serving command is backpressured");
  }
  if (written < 0) return system_failure("send DeepSeek rank serving command");
  return written == static_cast<ssize_t>(frame.size())
             ? Status::Ok()
             : Status::Internal("DeepSeek rank serving command was partially sent");
}

Result<std::optional<std::vector<std::byte>>>
LinuxDeepSeekRankServingControllerOperations::poll_completion(
    std::uint32_t rank) {
  auto socket = control_fd(rank);
  if (!socket.ok()) return socket.status();
  auto frame = receive_credentialed_packet(*socket,
      kDeepSeekRankServingCompletionBytes, handles_[rank].process_identity);
  if (!frame.ok()) {
    return frame.status().code() == StatusCode::kUnavailable
               ? Result<std::optional<std::vector<std::byte>>>(
                     std::optional<std::vector<std::byte>>{})
               : Result<std::optional<std::vector<std::byte>>>(frame.status());
  }
  auto completion = decode_deepseek_rank_serving_completion(*frame);
  if (!completion.ok()) return completion.status();
  if (completion->session.rank != rank ||
      completion->session.engine_epoch != engine_epoch_ ||
      completion->session.worker_generation != worker_generation_ ||
      completion->session.world_size != handles_.size() ||
      completion->session.process_identity != handles_[rank].process_identity ||
      completion->session.pidfd_identity != handles_[rank].pidfd_identity ||
      completion->session.control_identity != handles_[rank].control_identity) {
    return Status::FailedPrecondition(
        "DeepSeek rank serving completion handle differs");
  }
  return std::optional<std::vector<std::byte>>{std::move(*frame)};
}

Status LinuxDeepSeekRankServingControllerOperations::abort_generation(
    std::uint64_t engine_epoch, std::uint64_t worker_generation,
    const Status& cause) {
  if (engine_epoch != engine_epoch_ || worker_generation != worker_generation_) {
    return Status::FailedPrecondition(
        "DeepSeek rank serving abort identity differs");
  }
  return supervisor_->failed() ? Status::Ok() : supervisor_->abort_post_exec(cause);
}

}  // namespace pih
