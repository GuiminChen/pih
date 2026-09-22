# Third-party notices

PIH is licensed under the Apache License 2.0. It builds against, interoperates
with, or documents the following third-party projects. Unless a future release
explicitly states otherwise (including the reference source below), their source code, binaries and model weights are
not redistributed in this repository.

## Build and runtime dependencies

| Project | Use | Upstream license or terms |
| --- | --- | --- |
| [OpenSSL](https://www.openssl.org/) | Cryptographic verification | [Apache License 2.0](https://www.openssl.org/source/license.html) |
| [ICU](https://icu.unicode.org/) | Native tokenizer Unicode and regular expressions | [Unicode License](https://github.com/unicode-org/icu/blob/main/LICENSE) |
| [GoogleTest](https://github.com/google/googletest) | Native tests | [BSD 3-Clause License](https://github.com/google/googletest/blob/main/LICENSE) |
| [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | Optional NVIDIA GPU build/runtime dependency | [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) |
| [NVIDIA NCCL](https://github.com/NVIDIA/nccl) | Optional multi-GPU communication dependency | [BSD 3-Clause License](https://github.com/NVIDIA/nccl/blob/master/LICENSE.txt) |

Dependencies are obtained separately from their upstream distributors. Users
must comply with the applicable upstream licenses and distribution terms.

## Reference implementations and model formats

PIH documentation and compatibility work refer to
[vLLM](https://github.com/vllm-project/vllm),
[SGLang](https://github.com/sgl-project/sglang),
[Transformers](https://github.com/huggingface/transformers),
[Safetensors](https://github.com/huggingface/safetensors),
[Qwen](https://github.com/QwenLM) and
[DeepSeek](https://github.com/deepseek-ai) repositories. These references
document interoperability, formats and independently implemented behavior;
they do not grant PIH rights to redistribute upstream code, datasets,
tokenizers or model weights. Model artifacts remain subject to the terms
published with the exact artifact selected by the user.

Pinned source links and revisions used for a particular contract are recorded
next to that contract. Before incorporating upstream source code into PIH,
contributors must preserve its copyright and license notices and document the
provenance in this file.

## Adapted DeepSeek V4-0731 encoding

`plugins/common/deepseek_encoding.{h,cpp}` adapts conversation rendering and
completion parsing from `encoding/encoding_dsv4.py` in
[DeepSeek-V4-Flash-0731 revision 9e165c30](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/9e165c30e2704aec5d9d593cce3eebd58bbef1cb/encoding/encoding_dsv4.py).
Copyright (c) 2023 DeepSeek, MIT; its license is preserved in
`plugins/common/DEEPSEEK_LICENSE`. Local changes include a stateless
complete-conversation API, bounded native JSON handling, stricter completed-turn
validation, and acceptance of an empty argument line for zero-parameter calls.
No tokenizer payloads or model weights are redistributed.

## DeepSeek V4.1 development reference

Development compared behavior with DeepSeek V4.1 revision
`517ef625df97ec57aadc91b67506a57c20fdc5bb`. That Python/TileLang reference
package is not included in this public repository or any default distribution,
and native execution has no fallback path to it. No model weights, tokenizer
payloads or copied reference implementation are redistributed here.
