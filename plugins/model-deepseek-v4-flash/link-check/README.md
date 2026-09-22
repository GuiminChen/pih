# PP1 native link check

On Linux with a C++20 compiler, CMake, OpenSSL and ICU development libraries:

```sh
cmake -S plugins/model-deepseek-v4-flash/link-check -B out/deepseek-pp1-link-build
cmake --build out/deepseek-pp1-link-build --target pih-deepseek-pp1-link-check
```

This builds the actual model shared library and the prepare, verify and
generation-store executables from their production CMake definitions. The model
link uses `-z defs`; no provider stubs, Python inference, GPU execution or model
tests are involved. The CUDA capability adapters are ordinary C++ and need no
CUDA toolkit. This check does not build the SM89 Kernel Pack or a deployable
Worker bundle. Build `pih_kernels_deepseek_v4_sm89` in the normal CUDA build to
check the CUDA compiler, device images and strict pack link separately.

The PP1 rank resource owner accepts capability-backed leases, rank zero and
world size one. The old cross-process materialization/census/warm-seal API is
not part of this owner. Callers of that historical API need a separate
multi-process implementation; those declarations are not kept as unlinked
promises on the PP1 class.
