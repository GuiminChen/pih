# DeepSeek V4.1 / B300
English | [中文](deepseek-v41-b300.zh.md)

The default delivery is the native text model, SM103 Pack, rank Worker,
NCCL provider/helper, supervisor and HTTP service. Production Linux/CUDA
compilation and final linking are complete; **hardware execution is unverified**.
The code profile has 40 layers, hidden size 5120, 384 routed experts and 2/4/8
ranks. Canonical TP1 is an offline conversion format. Vision, MTP, arbitrary
checkpoints and other GPU identities are outside this delivery. V4-0731
artifacts and execution contracts cannot substitute for V4.1 artifacts.

Follow the [deployment guide](deployment.md) for build/install/seal/serve and
the exact [supervisor schema](../plugins/model-deepseek-v41/SUPERVISOR_CONFIG.md).
Provide independently trusted config, weights, tokenizer/map, worker/helper/Lock
hashes and an explicitly delegated cgroup parent. Scripts do not invent weights
or budgets and do not create system delegation automatically.

The Python/TileLang reference used during development is not included in the
public repository, native bundles, native source distribution or client
wheel/sdist. Native execution never falls back to it, and reference behavior
does not establish native model support.
