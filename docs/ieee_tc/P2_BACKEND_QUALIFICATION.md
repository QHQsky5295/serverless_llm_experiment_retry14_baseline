# P2：共同 vLLM 后端资格（进行中）

本记录不代表新版后端已通过模型或性能资格。旧环境与旧结果不覆盖。

## 当前候选及本机条件

候选：[vLLM 0.30.0 官方发布](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)，
发布于 2026-09-22。官方默认 CUDA 13.0 路径；同版本用于 Prime 和独立 vLLM，
其他论文 baseline 保留其专用后端。

本机：4 × RTX 3090 / SM86 / 24 GiB，驱动 580.105.08，Linux x86_64。
[NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
要求 CUDA 13.x 驱动至少 580，但存在特性/PTX 限制。因此只能认为满足基本
驱动门槛，不能据此宣布所有 LoRA/JIT 算子已兼容，也不自动升级宿主驱动。

旧主环境保持：Python 3.12.12、vLLM 0.10.2、torch 2.8.0+cu128、Ray 2.54.0、
transformers 4.57.6、triton 3.4.0。

## 依赖解析证据

在 3/4 GiB、swap=0、CPU 2/26 的独立 build scope 中运行 pip dry-run：

- 显式 `--ignore-installed --only-binary=:all: --no-cache-dir`，官方 PyPI。
- 198 个二进制包解析成功，退出码 0，没有向旧环境安装任何包。
- 原始 report 位于 `results/ieee_tc/p2_backend_qualification/metadata_20260925/`。
- report SHA256：`feae96b521e953cd0670006e64354d800f16e4e3b4b6773caffb4e238405900a`。
- 完整版本/hash 锁：`paper_results/ieee_tc/p2_backend/vllm0300_py312_candidate_20260925.txt`。

| 包 | 解析版本 |
|---|---|
| vLLM | 0.30.0 |
| torch | 2.13.0 |
| torchaudio / torchvision | 2.11.0 / 0.28.0 |
| transformers / triton | 5.17.0 / 3.7.1 |
| cuda-toolkit / nvidia-cuda-nvcc | 13.0.3.0 / 13.4.92 |

后两 CUDA 包的版本差别是实际解析结果，不隐瞒、不先断言运行失败。后续
必须检查实际编译/加载路径；若失败，用具体错误与官方兼容矩阵决定版本适配，
不靠关闭错误检查或静默退回 CPU 来通过。

## 独立安装

目标：`/home/qhq/.venvs/primelora_vllm0300_tc_20260925`。
使用现有 Python 创建隔离 venv，拒绝已有目标目录；pip 使用 `--isolated`、
`--require-hashes`、`--only-binary=:all:`。不复制模型、LoRA、trace 或旧环境。
安装进程隐藏 GPU，实际运行前读回 cgroup 3/4 GiB、swap=0 和 CPU 2/26。

启动前保守登记新增峰值 80 GiB；需要至少 220 GiB 空间，实际约 339 GiB，
已有 `20260925_p2_install_preflight.json` 通过。这个 80 GiB 是 setup 空间上界
估计，不是 GPU 预算或论文性能数值；最终记录实际增量。

本次 setup 在专用 tmux server `tc-p2-0925-01` 的 `install` 会话运行，属于
`primelora-tc-build-b52bf775f5d34836860733f2764e13d5.scope`。不能重新启动同一目标
覆盖中间状态；先核查日志和最终 receipt。安装失败保留日志/环境，不盲重试。

日志、pip 原始报告不入 Git；curated 安装回执和版本/hash 锁入 Git。
安装成功、pip check 成功也仅证明依赖 setup，不证明 3B/7B 性能达标。

## 原生计量接入注意事项

现有 runner 的 `_derive_vllm_latency_metrics` 优先 finished timestamp，旧
`generate_prepared`/子进程路径和 `last_timing` 仍需逐一核对。不得把完成通知
延迟写入 IEEE 的 O，也不得用缺失 metrics 的零值生成优值。

当前官方 [per-request metrics 文档](https://docs.vllm.ai/en/latest/features/per_request_metrics/)
有显式开启开关，要求统计记录可用；API TTFT 边界从 scheduled 开始，并非
本项目计划到达。streaming metrics 在最后 usage chunk 返回。资格时必须核对
实际 0.30.0 源码、原生字段及边界，不能仅凭字段名称拼接 admission D/T/O。
每请求统计本身也可能增加 CPU 开销，标准监控和详细 profiling 分开验证。

仍需按计划测试：FP16 两模型、mixed rank/modules、动态 LoRA、adapter
identity、CUDA Graph、batch/cancel、原生 token、真实 slot/eviction、KV block、
正确释放以及 warm 指标。实际 worker 限制和监控握手通过前不启动模型比较。
