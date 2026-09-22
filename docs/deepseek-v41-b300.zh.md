# DeepSeek V4.1 / B300
[English](deepseek-v41-b300.md) | 中文

默认交付为原生文本模型、SM103 Pack、rank Worker、NCCL provider/helper、supervisor 和 HTTP 服务。
完整生产 Linux/CUDA 编译与最终链接已完成；**硬件运行未验证**。
生产支持的代码形状为40层、hidden 5120、384 routed experts、2/4/8 ranks；
canonical TP1 是离线转换格式。视觉、MTP、任意 checkpoint 和其他 GPU 身份不在本次范围。
V4-0731 的权重布局与执行契约不能作为 V4.1 制品使用。

构建、安装、seal 和启动见[部署指南](deployment.zh.md)，
精确配置见[supervisor schema](../plugins/model-deepseek-v41/SUPERVISOR_CONFIG.md)。
必须提供独立可信的配置、权重、tokenizer、token-map、worker/helper/Lock 摘要及委派 cgroup 父目录。
脚本不生成虚构权重或预算，也不自动建立 cgroup 委派。

开发期使用的 Python/TileLang reference 不进入公开仓库、原生包、源码发行包或客户端
wheel/sdist，且没有自动回退可达路径。不要将 reference 的能力范围当作原生模型支持范围。
