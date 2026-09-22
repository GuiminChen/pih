# Contributing to PIH
English | [中文](CONTRIBUTING.zh.md)

PIH is an early-stage, plugin-first LLM inference framework. Contributions are
welcome, especially when they preserve explicit capability contracts and keep
deployment bundles minimal.

## Before contributing

1. Open an issue for substantial API, ABI, plugin-contract or architecture
   changes before implementation.
2. Keep model weights, generated bundles, benchmark outputs and credentials out
   of the repository.
3. Do not describe a GPU/model profile as supported without the evidence and
   promotion gates required by its checked-in contract.

## Development workflow

Create a topic branch, make a focused change, and submit a pull request that
describes the affected contracts and compatibility impact. For the
dependency-light native path:

```bash
cmake --preset phase1-native
cmake --build --preset phase1-native
```

The Linux CPU contract workflow is the portable baseline for pull requests.
CUDA-, model- and hardware-specific validation is required only when a change
claims or alters those behaviors.

By submitting a contribution, you agree that it is licensed under the Apache
License 2.0 used by this repository.
