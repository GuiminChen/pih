# PIH
English | [中文](README.zh.md)

High-performance, plugin-first LLM inference framework built for low latency,
high throughput, and efficient memory usage.

The target architecture is `pih-worker` plus a sealed Lock-selected graph of
native plugins and architecture-specific Kernel Packs. The former monolithic
native Python runtime is no longer shipped by `pip install .`; that installs
only `pih_client`, an HTTP client and host diagnostics. Historical runtime source
still exists pending migration and is not a supported compatibility path.
An unspecialized CMake configure therefore builds the dependency-light Phase 1
worker and `pih.testing.minimal`. Checked-in deployment presets bind
`PIH_DEPLOYMENT_PROFILE` to an exact `PIH_ENABLED_PLUGINS` set; use the
`custom` profile only when independently building or packaging a partial
plugin set. The
repository is under active implementation. The current `v0.1.0-alpha` line is a
developer preview of the contracts, plugin system, worker and build composition;
it is not a production-support release. No GPU or model profile is currently
claimed as supported: RTX 4090 D and H100 PCIe evidence remains open.

## Why PIH

The [native inference guide](docs/native-inference.md) covers Qwen's plugin-only
build and launch commands, exact GPU targets, HTTP requests and observation
scripts. This path has completed Linux/CUDA cross compilation and linking; hardware execution is unverified and does not
use a Python inference compatibility bridge.

- Load only the platform, storage, backend, execution, memory, model, surface,
  and Kernel Pack plugins selected by a deployment Lock.
- Keep deployment composition explicit and reject missing, duplicate or
  incompatible capabilities at startup.
- Separate reusable runtime contracts from model- and architecture-specific
  kernels.
- Keep heavyweight model artifacts outside source and build trees.

PIH currently targets Linux. APIs, ABIs, plugin contracts and package layout may
change before the first stable release.

For the fastest dependency-light microkernel iteration, configure and run the
Phase 1 native smoke. It builds no CUDA, Python extension, tests, or monolithic
runtime:

```bash
cmake --preset phase1-native
cmake --build --preset phase1-native
```

On the Linux RTX 4090 D development host, build the exact DeepSeek PP1 bundle
with an external verified model generation:

```bash
cmake --preset deepseek-4090d-pp1 \
  -DPIH_DEEPSEEK_4090D_ARTIFACT_DIRECTORY=/absolute/model/generation \
  -DPIH_DEEPSEEK_4090D_ARTIFACT_ROOT_SHA256=<64-lowercase-hex>
cmake --build --preset deepseek-4090d-pp1
cmake --build out/build/deepseek-4090d-pp1 \
  --target pih_deepseek_4090d_pp1_smoke
```

The PP1 smoke keeps model bytes outside the build tree and reports
`hardware_evidence_open`; successful assembly is not a release-support claim.

Start with the [documentation index](docs/index.md), the
[architecture guide](docs/architecture.md), the
[development guide](docs/development.md), and the
[plugin guide](docs/plugins.md). Each document links to its Chinese version.

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a change. Security issues
should be reported according to [SECURITY.md](SECURITY.md).
Third-party dependencies and reference projects are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The `cpu-contract` workflow builds Linux CPU-only C++/Python artifacts and
runs the contract suites. It deliberately disables CUDA and NCCL; a green CPU
workflow is not GPU, model, performance, or supported-profile evidence. Its
CI-only build/test toolchain is pinned in `requirements/ci-cpu.txt`.

Before copying artifacts to a target server, a Linux operator can run the
read-only target preflight. It checks the selected 1–4 device tuple against the
exact target SKU, confirms the visible-device namespace is unmasked, and reports
whether physical host memory reaches the 256 GiB planning baseline:

```bash
pih-preflight-target-host --hardware-profile rtx4090d --devices 0
# or: pih-preflight-target-host --hardware-profile h100-pcie --devices 0,1
# Replay a saved JSON receipt without querying a live GPU:
pih-preflight-target-host --verify-receipt target-preflight.json
# From an uninstalled source checkout, invoke the same CLI directly:
python3 python/pih_client/preflight.py --hardware-profile rtx4090d --devices 0
```

The command emits `hardware_evidence_open` even when it succeeds. It is only a
server-readiness diagnostic. The verification mode checks the complete receipt
schema, immutable root, target SKU/topology and host-memory derivation, but
model loading, CUDA/NCCL execution, capacity, numerical correctness, and
release support still require the exact target qualification suites.
