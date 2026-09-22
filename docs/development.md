# Development
English | [中文](development.zh.md)

## Standalone native contract compilation

The old `pih_core`/`PIH::core` and `pih_cuda`/`PIH::cuda` aggregate targets,
their full internal-header install path, and the monolithic test entry have
been removed. `PIH_BUILD_MONOLITH=ON`, `PIH_BUILD_TESTS=ON` and the old
`PIH_ENABLE_NCCL=ON` aggregate now fail configuration explicitly. Select native
plugins and `PIH_BUILD_NATIVE_CONTRACT_TESTS`; V4.1's independent NCCL development
targets use their separate documented option. The CPU contract CI builds the
native contract target, not the old C++ aggregate. Historical C++ test sources
still need migration; the smaller native suite does not establish their coverage
or complete plugin capability binding. No compatibility archive is produced.

To include the migrated execution, storage, host-spill and Linux platform
implementations in syntax compilation, run
`python3 tools/check_native_syntax.py --compiler clang++ --plugin-support` after
preparing the pinned Linux headers described below. This adds each owner's
current top-level `.cpp` files (including entrypoints) to the standard selection.
The observation filename ends in `-plugin-support.json`; it records exact source
hashes, but is not a complete header closure or proof of linking, capability
bindings, model execution or runtime correctness. The CI syntax job uses the
same option; editing that job does not constitute an executed CI result.

HTTP framing, Qwen stop-string and V4.1 output-lease contract tests can be compiled independently of the historical
monolithic test suite. On Linux, install `libgtest-dev` alongside CMake, Ninja,
a C++20 compiler and OpenSSL 3 development files, then run:

```bash
cmake -S . -B out/build/native-contracts -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPIH_DEPLOYMENT_PROFILE=custom \
  -DPIH_BUILD_WORKER=OFF -DPIH_BUILD_PLUGINS=OFF \
  -DPIH_BUILD_MONOLITH=OFF -DPIH_BUILD_PYTHON=OFF \
  -DPIH_ENABLE_CUDA=OFF -DPIH_ENABLE_NCCL=OFF \
  -DPIH_BUILD_TESTS=OFF -DPIH_BUILD_NATIVE_CONTRACT_TESTS=ON
cmake --build out/build/native-contracts --target pih-native-contracts --parallel 2
```

Test discovery is deferred until CTest execution, so the build does not execute
tests or load models. The targets link only GTest plus the framing implementation
or Qwen stop filtering with bounded JSON/status primitives, or the V4.1
publication queue with output-credit/status primitives. The output cases cover
reservation, publication, consumer release, retired in-flight discard, stale and
foreign leases, invalid envelopes and empty visible output. They do not prove
rank retirement: the caller must establish that before discarding in-flight work;
it is not whole-service, socket, GPU or model qualification. The native contract
targets have been compiled and finally linked with the production implementation
dependencies in the release integration environment, but the resulting Linux ELF
tests were not executed. The remote compile-only CI workflow is configured but has
not been run. When test execution is intended on Linux, use
`ctest --test-dir out/build/native-contracts --output-on-failure`.

## Development worker

Use Linux with CMake 3.26+, Ninja, a C++20 compiler and OpenSSL 3 development headers. On Ubuntu 24.04:

```bash
sudo apt-get update
sudo apt-get install --yes cmake ninja-build g++ libssl-dev
cmake --preset phase1-native
cmake --build --preset phase1-native --parallel 2
```

The build target runs the minimal plugin lifecycle smoke. No model, CUDA toolkit or Python extension is needed. This is not a text-generation test.

Alternatively, with Docker available, build the CPU preview container:

```bash
docker build -f deploy/Dockerfile.preview -t pih-preview .
docker run --rm --network none --read-only --cap-drop ALL pih-preview
```

The image builds the worker and external SDK example, runs the contract and
loading checks during its build, then runs the minimal plugin as a non-root
user. Its process exits after the smoke; it does not start an HTTP server.

The old `_pih` extension bindings, runtime package, build and installation targets have been removed from the public tree. `PIH_BUILD_PYTHON=ON` now fails explicitly instead of suggesting the monolith. The extension-injection test runner and its dedicated tests have also been deleted. [CPU CI](../.github/workflows/cpu-contract.yml) retains C++ contracts and the client-only wheel check, but no longer produces the extension or runs its old Python suite. C++ contracts use `PIH_DEPLOYMENT_PROFILE=custom`; this is not a production compatibility entry. Removing the old tests is not evidence of GPU or model qualification.

Offline DeepSeek M5 evidence processing now lives in `tools/evidence/deepseek_optimization_evidence.py`. Its tools/tests import this tooling package directly, without initializing `pih` or loading `_pih`; no forwarding alias remains at the old location. The module handles raw observations/evidence, not inference, and is not included in the `pih_client` wheel. Migration did not run observation collection or model tests. Other legacy-package dependencies still need cleanup.

The Qwen teacher-forced quality compiler likewise lives in `tools/evidence/qwen_teacher_forced_quality.py`, with no legacy package export or forwarding module. Run `python tools/compile_qwen_teacher_forced_quality.py --input INPUT.json --output OUTPUT.json` from the checkout to process existing measurements; it does not run a model or collect measurements. Schema identifiers remain unchanged. Legacy receipt/deployed-role consumers now reference the tooling module explicitly, but their other dependencies still need migration.

`python -m pip wheel . --wheel-dir dist` builds a pure Python 3.11+ client wheel.
It contains only `pih_client`; CMake, pybind11, CUDA and `_pih` are not part of
this package. Old `from pih import Engine` applications must migrate to the HTTP
client; there is no compatibility alias. Native plugins are built separately.
Prebuilt published wheels are not promised by this preview.

If CMake is missing, install the prerequisites above. If OpenSSL is missing, install `libssl-dev`. If a stale cache selects the wrong profile, configure a new build directory. For SDK consumption see [plugins](plugins.md).

## Native Qwen tokenizer reference comparison

For later qualification on Linux, the comparator executes the native tokenizer
shared with the model plugin, not the retired Python candidate:

```bash
cmake -S . -B out/build/tokenizer -G Ninja -DPIH_DEPLOYMENT_PROFILE=custom -DPIH_BUILD_WORKER=OFF -DPIH_BUILD_PLUGINS=OFF -DPIH_BUILD_MONOLITH=OFF -DPIH_BUILD_PYTHON=OFF -DPIH_BUILD_TESTS=OFF -DPIH_ENABLE_CUDA=OFF -DPIH_BUILD_TOKENIZER_TOOLS=ON
cmake --build out/build/tokenizer --target pih-tokenize --parallel 2
python tools/verify_qwen_tokenizer_reference.py --native-tokenizer out/build/tokenizer/plugins/common/pih-tokenize < /models/Qwen3/tokenizer.json
```

The optional comparator requires the Python `tokenizers` reference library;
the native build also requires ICU development libraries. It accepts only the
script-pinned tokenizer SHA-256, copies that payload into a private temporary
directory, and compares eight encode cases with a per-case timeout (default
60 seconds, configurable with `--timeout-seconds`). Supply only a trusted local
executable. The report records its before/after SHA-256, not an authenticated
executable/dependency closure; subprocess logs use temporary files, not a disk
quota sandbox. The report no longer contains the old Python `execution_root`.
Passing this corpus does not qualify special tokens, decoding, chat rendering,
model generation or GPU support. This command has not been run as part of the
migration; other legacy tokenizer consumers still require removal.

## Native Qwen plain-chat reference comparison

The service and `pih-qwen-format` now share `plugins/model-qwen3/chat_format.h`.
The plan preserves separate control-token segments, but coalesces adjacent
ordinary text before BPE instead of splitting role/newline/content arbitrarily.
Caller text uses literal encoding, so added-token spellings cannot become
template controls; only the constant assistant prefix recognizes such spellings.
It imposes a 2 MiB rendered-byte ceiling.
The CPU tool accepts a JSON file containing only `messages`, returning a JSON
`text` field; it does not tokenize, load weights or execute inference.

Using the CPU build directory configured above, later qualification can run:

```bash
cmake --build out/build/tokenizer --target pih-qwen-format --parallel 2
python tools/verify_qwen_chat_renderer_reference.py --native-renderer out/build/tokenizer/plugins/common/pih-qwen-format < /models/Qwen3/tokenizer_config.json
python tools/verify_qwen_chat_renderer_reference.py --native-renderer out/build/tokenizer/plugins/common/pih-qwen-format --tokenizer /models/Qwen3/tokenizer.json < /models/Qwen3/tokenizer_config.json
```

This requires the optional Python `jinja2` reference dependency and the pinned
chat-template hash. The comparator no longer imports the old Python renderer.
It checks five plain-text cases with `enable_thinking=False`, including leading
newlines and whitespace-only content. Optional `--tokenizer` requires the
`tokenizers` reference library and pinned tokenizer bytes, and also compares
token IDs through the service's shared assembly function. The native tool's
corresponding mode is `pih-qwen-format REQUEST.json --tokens TOKENIZER.json`;
that direct mode now pins tokenizer bytes to the same SHA-256 as the service.
It does not authenticate the full model or filesystem. Tools, reasoning histories,
thinking mode, control-token literals in content and generation remain outside
this gate and are not declared implemented by it. Do not substitute this corpus
for the old Python renderer's broader tests. Use a trusted executable; the same
temporary-log and executable-hash limitations as the tokenizer comparator apply.
The default per-case timeout is 30 seconds. Only syntax/import checks have been
performed during migration, not this comparison or GPU tests.

## Syntax-only checks from a Windows host

When Linux execution is unavailable but Clang is installed, these optional
commands check Linux-target C++20 syntax without linking or running anything:

```text
python tools/prepare_linux_syntax_headers.py
python tools/check_native_syntax.py --compiler PATH_TO_CLANG_PLUS_PLUS
```

Preparation downloads five fixed Ubuntu Noble development packages, including
Linux OpenSSL headers, from the
official archive, verifies sizes/SHA-256 against `tools/linux-syntax-sysroot.lock.json`,
and extracts headers only into a new `out/linux-syntax-sysroot`. It requires
bsdtar with ar/zstd support (Windows system `tar` works; elsewhere select
`--tar bsdtar`). No system package installation, maintainer scripts, runtime
binaries, CUDA or model files are involved. Existing output is not overwritten;
download staging directories are retained under `out` for inspection. The
package lock is for development syntax checks, not a supported production OS
or security-update policy.

After a package-lock change, create a new sysroot rather than editing an old
receipt. For example, prepare with `--output out/linux-syntax-sysroot-openssl`
and pass `--sysroot out/linux-syntax-sysroot-openssl` to the checker. Old receipts
intentionally fail the updated lock check. Headers do not supply linker libraries.

For the Qwen cuBLAS Lt adapter, also prepare its small pinned NVIDIA development
package with `python tools/prepare_cublas_syntax_headers.py`. This reads only six
allowlisted regular header files and the NVIDIA license into
`out/cublas-syntax-headers-13.4.1.3`, with package and file hashes in its receipt.
Pass `-I out/cublas-syntax-headers-13.4.1.3/include` alongside the CUDA header root
when directly syntax-compiling `src/backend/cuda/gemm_plan.cpp`. Library stubs,
symlinks and package test scripts are not extracted or executed. This is not
installation of cuBLAS; actual CUDA builds still require the real toolkit and
link libraries. The header package targets the CUDA 13.2 version range.

The checker covers its explicit default translation-unit list: shared text tools,
Worker, HTTP surface, DeepSeek generation/model entrypoint, Qwen entrypoint and
converter, scheduler, core primitives, CUDA rank runtime/bootstrap and the
capability-only shared-expert adapter. It selects the native PP1 compile
definitions and produces `out/native-syntax-report.json`. This is not a CMake
build, whole-project check, linked binary, CUDA kernel compile, runtime test,
or release qualification. No success from this command closes those gates.

To include the 161 C++ model sources selected by CMake for the native DeepSeek
PP1 target, export the compilation database in a plugin-only build directory:

```text
cmake -S . -B out/build/preview-local -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
python tools/check_native_syntax.py --compiler PATH_TO_CLANG_PLUS_PLUS --deepseek-model-build out/build/preview-local
```

This checks the combined source list using the pinned Linux header environment.
The checker verifies the expected source-list hash and PP1 definitions before
compiling; it does not replay every CMake command or include CUDA sources.
Results and full diagnostics are saved as `out/native-syntax-closure-report.json`
and the adjacent `.log` file. A failing batch makes the command fail. The same
linking, CUDA, runtime and release limitations above still apply.

To also check all 24 capability-only C++ adapters in the native PP1 CUDA
backend, add `--deepseek-backend`. The source selection excludes DSpark, NCCL,
the direct physical-device probe and Kernel Pack contract sources, matching the
pinned CMake count/hash. It does not require CUDA headers or compile `.cu` files.

```text
python tools/check_native_syntax.py --compiler PATH_TO_CLANG_PLUS_PLUS --deepseek-backend
python tools/check_native_syntax.py --compiler PATH_TO_CLANG_PLUS_PLUS --deepseek-model-build out/build/preview-local --deepseek-backend
```

The report records the selected, deduplicated translation units; the first mode
writes `out/native-backend-syntax-report.json`, the combined mode writes the
closure report above. Both are syntax observations, not CUDA builds or tests.
The `native-syntax` job in `native-preview.yml` configures this combined check
without executing models. Editing the workflow does not mean it has run or passed.

The native-preview CI tokenizer job also compiles the actual PP1 model object,
the model-owned host CUDA capability-adapter object target, and the model-support
library with the Linux compiler. The adapters do not require CUDA headers or
toolkit activation. This job does not link
the CUDA model plugin or execute inference; CI configuration is not evidence
that a run has passed.

The explicit offline/artifact/supervisor source list and conversion variant can be checked with
`python tools/check_native_syntax.py --compiler PATH_TO_CLANG_PLUS_PLUS --artifact-tools`.
It writes `out/native-artifact-syntax-report.json` and a full diagnostic log.
This does not link the tool; the Linux `artifact-tool-build` CI job is configured
to build the actual executable with `PIH_BUILD_ARTIFACT_TOOLS=ON`.
