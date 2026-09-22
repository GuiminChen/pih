# NCCL collective transport

`pih.transport.nccl` is the sole rank-process owner of NCCL communicator
lifecycle and collective submission. It provides the process-scoped
`transport.collective.v1` C ABI from
`include/pih/contracts/transport_collective_v1.h`. DeepSeek V4.1 rank code
holds only an opaque provider handle and must not cast it to `ncclComm_t`.

The provider pins NCCL 2.31.2, checks the prepared CUDA device and rank
identity, validates device pointer extents, and offers bounded BF16/FP32
all-reduce plus FP32 all-gather. A failed initialization that returned a
handle still requires one abort attempt. Failed release/abort retains the
provider-owned record and prevents plugin retirement; the rank process must
fail-stop under its supervisor. No destructor silently retries an NCCL call.

`nccl_bootstrap_main.cpp` is the separate sealed-ID helper executable source.
It is not loaded as a plugin and is not a Python fallback. The V4.1 rank Lock
includes this plugin; single-GPU bundles omit it.

The source has Linux-target C++ syntax coverage. Linux ELF linking, `.cu`
compilation, process integration and B300/NCCL execution remain unverified.
