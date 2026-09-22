# Plugin development
English | [中文](plugins.zh.md)

Build the worker first using [development](development.md). Then follow the [standalone external plugin example](../examples/external-plugin/README.md).

```bash
cmake --install out/build/phase1-native --prefix "$PWD/out/sdk" --component pih-plugin-sdk
```

Consumers use `find_package(PIHPluginSDK 0.1.0 EXACT CONFIG REQUIRED)` and link `PIH::plugin_sdk`. The SDK component contains the C ABI headers only; the internal C++ status bridge is excluded.

With `PIH_BUILD_MONOLITH=OFF`, ordinary installation also uses this explicit
header boundary: it does not publish the internal model/backend/core headers or
`status_bridge.h`. Use a fresh install prefix when migrating an older bundle;
CMake installation does not remove stale headers already present there. The
monolithic inference mode is rejected by the current top-level CMake build.
A previous local native installation produced the original ten public headers
and three SDK CMake files; the current SDK install rule now includes all nineteen
C capability contracts plus six ABI headers (25 total), including execution,
transport, device, memory, async, health, platform and artifact interfaces.
This expands external plugin authoring without exposing C++ internals.
The external example configured against that installed prefix and compiled/linked
with tests disabled. The local Windows cross-toolchain emitted a `.drectve`
linker warning, so DLL loading/exports and Linux runtime behavior remain unverified;
this is installation/build evidence, not a plugin execution qualification.

After installing into a fresh prefix, verify the boundary against the current
source revision with `python3 tools/verify_plugin_sdk_install.py /absolute/sdk`.
Without an install, `python3 tools/verify_plugin_sdk_install.py --source-only`
checks that the verifier list, CMake install rule, and external C probe agree.
The check requires the exact 25 headers, compares their bytes to source, checks
that their `pih/` includes stay inside the published set, and requires the three
CMake package files. Extra internal headers and stale public copies fail. It
rejects symlinks encountered inside the header tree but is not a signed-package
or hostile-filesystem verifier. Package-file presence does not validate their
semantics; retain the external consumer configure/build step. An earlier version
of the check passed against a local fresh install; the expanded 25-header
installation still needs a fresh configure/install/consumer build.

The external example also builds `sdk_c_header_probe`, a C11 object including
all 25 published headers from the installed SDK. It does not execute code or
link a model. Building `hello_plugin` depends on this probe, so the normal CI
consumer build checks C compatibility as well as the C++ example. The local
earlier probe compiled successfully; the expanded probe still needs a fresh
consumer build and is not a cross-compiler ABI layout or binary compatibility
qualification.

The example publishes an activation-scoped capability, resolves it during configuration, and advances register/configure/start/ready/drain/stop/dispose callbacks. It checks host ABI and contains exceptions at the C boundary. Registration failure must not commit lifecycle progress; shutdown must also work during startup rollback.

The [native preview workflow](../.github/workflows/native-preview.yml) copies this example outside the checkout before building and loading it. The development Lock is not a signed production deployment or a sandbox for untrusted native code.
