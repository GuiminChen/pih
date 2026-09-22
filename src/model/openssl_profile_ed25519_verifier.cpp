#include "pih/model/openssl_profile_ed25519_verifier.h"

#include <algorithm>
#include <memory>

#include <openssl/err.h>
#include <openssl/evp.h>

namespace pih {
namespace {

bool valid_key_id(std::string_view value) {
  return !value.empty() && value.size() <= 128 &&
         std::all_of(value.begin(), value.end(), [](unsigned char byte) {
           return byte >= 0x21 && byte <= 0x7e;
         });
}

struct PkeyDeleter final {
  void operator()(EVP_PKEY* value) const noexcept { EVP_PKEY_free(value); }
};

struct MdContextDeleter final {
  void operator()(EVP_MD_CTX* value) const noexcept { EVP_MD_CTX_free(value); }
};

}  // namespace

Status OpenSslProfileEd25519Verifier::verify(
    std::string_view key_id, std::span<const std::byte> public_key,
    std::span<const std::byte> message,
    std::span<const std::byte> signature) {
  if (!valid_key_id(key_id) || public_key.size() != 32 ||
      signature.size() != kEd25519SignatureBytes ||
      message.size() > kProfileEnvelopeMaximumBytes) {
    return Status::InvalidArgument(
        "profile Ed25519 verification input is invalid");
  }

  ERR_clear_error();
  std::unique_ptr<EVP_PKEY, PkeyDeleter> key(
      EVP_PKEY_new_raw_public_key_ex(
          nullptr, "ED25519", nullptr,
          reinterpret_cast<const unsigned char*>(public_key.data()),
          public_key.size()));
  if (!key) {
    ERR_clear_error();
    return Status::Internal("OpenSSL could not create Ed25519 public key");
  }
  std::unique_ptr<EVP_MD_CTX, MdContextDeleter> context(EVP_MD_CTX_new());
  if (!context) {
    ERR_clear_error();
    return Status::Internal("OpenSSL could not allocate Ed25519 context");
  }
  if (EVP_DigestVerifyInit(context.get(), nullptr, nullptr, nullptr,
                           key.get()) != 1) {
    ERR_clear_error();
    return Status::Internal("OpenSSL could not initialize Ed25519 verifier");
  }
  const int result = EVP_DigestVerify(
      context.get(),
      reinterpret_cast<const unsigned char*>(signature.data()),
      signature.size(),
      reinterpret_cast<const unsigned char*>(message.data()), message.size());
  ERR_clear_error();
  if (result == 1) return Status::Ok();
  if (result == 0) {
    return Status::FailedPrecondition("profile Ed25519 signature is invalid");
  }
  return Status::Internal("OpenSSL Ed25519 verification failed");
}

}  // namespace pih
