# Release contents and acceptance
English | [中文](release.zh.md)

Code/build acceptance requires all four production configurations to compile,
finally link, generate the correct CUDA device images and install; the native
capability and deployment graph must close without known substantive gaps.
Hardware execution is unverified and is not a blocker for this delivery.
Remote CI has not been executed locally or reported as passing.

## Three distinct distributions

1. Native bundles: configure a closed model profile, build `pih-release`, then
   install only component `pih-release` into a **new directory**. The positive
   target list is `cmake/PIHRelease.cmake`. It installs selected native commands,
   providers, model and Pack, Qwen external cubin/manifest pairs, deployment
   scripts and licensing. No weights, Python model runtime, qualification binary
   or reference backend is included. Runtime CUDA/ICU/NCCL/system libraries are
   external prerequisites, not copied from the build host.
2. Native source archive: `release/source-policy.json` is the positive inventory;
   `tools/package_release.py source --output /absolute/new-source.tar.gz` writes
   the exact member sizes/hashes into `RELEASE-MANIFEST.json`, scans selected
   textual files for credential patterns and local user paths, and reads the
   archive back to verify every member. Tests and the minimal SDK example remain
   development sources, not installed model executables. Historical design,
   implementation and qualification records are excluded from this archive.
3. Python wheel **and sdist**: `pih_client` HTTP client and host diagnostic only.
   `MANIFEST.in` and explicit setuptools package selection exclude the old
   `python/pih`, `_pih`, reference runtime and native build tree. The client does
   not import a model implementation.

The development-only DeepSeek V4.1 Python/TileLang reference package is absent
from the public repository and all three distributions. Native deployment has no
automatic fallback to it. Legacy Python runtime and binding sources, private
build evidence, delivery backups, `.git`, `out`, weights and generated binaries
are excluded from both the public repository and the native source archive.

```bash
python3 deploy/build-release.py qwen-4090d --build-dir /absolute/build-qwen --bundle /absolute/new-qwen
python3 tools/package_release.py native --root /absolute/new-qwen --output /absolute/new-qwen.tar.gz
python3 tools/package_release.py source --output /absolute/new-source.tar.gz
python3 -m build --wheel --sdist
```

Each native bundle contains the Apache-2.0 project license, third-party notice
and preserved DeepSeek MIT notice. Native source includes the adapted encoding
notice at its original path. Weights/tokenizers and third-party runtime libraries
are not redistributed. Pattern scans are bounded checks, not a claim that every
possible credential or licensing issue can be detected automatically.

The compile-only `native-release-build` workflow uses a maintainer-managed
Linux CUDA13.2/NCCL build host, is manually dispatched, and executes no GPU
program. Existing CPU contract workflows are separate. No automatic publish,
Git commit or push is part of this delivery.

The Docker preview is a separate CPU/minimal-plugin developer image, not a model bundle. Its final stage copies only the Worker, minimal-plugin files, Lock and license notices; the build context excludes the historical/reference runtime. Docker build/run has not been executed in this local environment.
