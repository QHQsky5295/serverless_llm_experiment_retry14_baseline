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

现有 runner 的 legacy `_derive_vllm_latency_metrics` 优先 finished timestamp。
新的 opt-in `timing_contract=ieee_tc_native_v1` 已接入 engine、子进程返回和
controller，严格按首末 token 分离完成通知开销；13 项无 GPU 测试通过，
详见 P1-D5。真实新版 stats/时钟/开销资格仍未完成，不能据此放行推理实验。

当前官方 [per-request metrics 文档](https://docs.vllm.ai/en/latest/features/per_request_metrics/)
有显式开启开关，要求统计记录可用；API TTFT 边界从 scheduled 开始，并非
本项目计划到达。streaming metrics 在最后 usage chunk 返回。资格时必须核对
实际 0.30.0 源码、原生字段及边界，不能仅凭字段名称拼接 admission D/T/O。
每请求统计本身也可能增加 CPU 开销，标准监控和详细 profiling 分开验证。

仍需按计划测试：FP16 两模型、mixed rank/modules、动态 LoRA、adapter
identity、CUDA Graph、batch/cancel、原生 token、真实 slot/eviction、KV block、
正确释放以及 warm 指标。实际 worker 限制和监控握手通过前不启动模型比较。

## 原生 worker 观察接口（已接线，真实模型调用待资格）

沿用 `gpu_monitor.py`，新增官方 `worker_extension_cls` 允许的只读扩展，
通过既有 engine collective RPC、dedicated worker 和 proxy 返回数据。
配置 `model.ieee_worker_observation=true` 才安装扩展；默认历史路径不变。
参考 [vLLM 0.30.0 WorkerWrapperBase 的 extension 接口](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/worker_base.py)。

读取：原生 worker PID/UID/cgroup/affinity、boot/time namespace、后端版本、
可见 GPU 与 local device、实际 CUDA mem-info、PyTorch allocated/reserved、
LoRA CPU registry、active GPU set 与 slot mapping。

LoRA A/B tensor 的 shape、dtype、view size 与底层 storage 分开记录，共享
storage 去重。不是用文件大小或 rank 估算显存，也不把整池分配再按多个 view
重复相加。当前仅接受已审计的 dense stacked A/B 表示；未知模块报错，不漏计。

默认不做 GPU synchronize。可显式 barrier 进行独立资格/剖析，但不把这种
同步放进每请求监控。无论是否 barrier，该快照没有 dispatch 引用或原子
reservation，始终标 `production_admission_snapshot=false`；不得直接冒充
IEEE confirmed tier 的租约证据。

五项 fake-tensor/实际 RPC 方法测试通过，覆盖 storage alias、CPU/GPU 区分、
slot 不一致、未知表示、同步标记、嵌套字段运输和错误不吞掉。它们没有加载
模型，也没有证明原生 worker 已满足资源包络。下一次真实模型资格必须读取
这些事实，验证后才能将其接入 owner；不能先把代理提示映射成 GPU-ready。

## 原生 GPU 引用接入（模型资格前的独立合同）

`model.ieee_gpu_references=true` 安装同一个 worker extension，并在真正的
worker CPU/GPU LRU 上维护请求引用；不是只修改 router 的 resident set。
同一份扩展的 observation 仍只读，不因启用引用功能而自动成为受保护快照。

- 当前限定每个 runtime TP=PP=1；多 runtime 横向扩容不等于 TP>1。
- snapshot 不持有引用；acquire 必须提交 worker incarnation 和 epoch。
- CPU-only、不再驻留或过期快照返回 conflict，不通过隐式 load 变成“原 GPU hit”。
- 第一个引用保护 native CPU/GPU cache；多个请求共享；最后一个引用只解除
  本 owner 的 pin，保留已有外部 pin。pin 不是本系统独创算法。
- 在 worker 当前 CUDA stream 上记录并等待 event，计入 acquisition 开销；
  不进行全设备同步。必须在真实资格中确认本机 dense LoRA copy/execute 的
  stream 和线程顺序，不能仅凭 fake event 测试宣称 readiness 已完成验证。
- 现有 engine/prepared/RPC 通道要求 generation 提交对应 adapter 的引用，
  并将原生 backend request ID 绑定到 lease。未见 native terminal 时不能释放。
  异常/取消不会假称 backend 已结束：保留引用直到明确终态或整 worker 回收。
- 显式卸载通过同一 owner；不能绕过 owner 调 remove、load_inplace 或 pin。
  正常 cache replacement 使用后端 LRU；检测到已有引用被旁路失效时，owner
  invalidated，停止使用该 worker，不把它重新解释为 ordinary cache miss。

这一步不实现 cold-load admission、active-request reservation、victim transaction
或慢层共享引用。没有将上述条件偷换为 True；回执明确
`request_admission_reserved=false`、`production_launch_authorized=false`。
后续真实模型资格用这些现有入口验证 hold/use/release/eviction，再完成控制器
快照与资源预约接入。没有使用本步骤数据声称 TTFT 或 GPU-s 已改善。
