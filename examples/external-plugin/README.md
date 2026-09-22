# External plugin example

This standalone CMake project consumes the installed PIH C ABI SDK. It registers
and resolves one activation-scoped capability, then exercises the worker
lifecycle. It does not perform inference or establish GPU support.

From the PIH checkout on Linux, with CMake 3.26+, Ninja, a C++20 compiler and
OpenSSL 3 development headers:

```bash
cmake --preset phase1-native
cmake --build --preset phase1-native
cmake --install out/build/phase1-native --prefix "$PWD/out/sdk" --component pih-plugin-sdk
cmake -S examples/external-plugin -B out/external-plugin -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/out/sdk" \
  -DPIH_WORKER="$PWD/out/build/phase1-native/pih-worker"
cmake --build out/external-plugin
ctest --test-dir out/external-plugin --output-on-failure
cmake --build out/external-plugin --target smoke
```

The example directory can be copied outside this repository. Only the installed
SDK and a matching worker are needed; no PIH source directory or core library is
linked. The smoke generates a development Lock and loads the shared library.
The SDK package is preview version 0.1.0; build the worker and SDK from the same
revision. Exact package version matching is not a promise of cross-revision ABI
stability. `status_bridge.h` is an internal C++ adapter and is intentionally not
part of the standalone SDK component.

The contract test checks ABI rejection, duplicate entry, lifecycle ordering,
registration failure/retry and exception containment. A plugin instance serves
one activation only. A real plugin must add its own typed API contract and
resource ownership tests before being used by another plugin.
