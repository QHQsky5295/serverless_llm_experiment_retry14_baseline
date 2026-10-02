# D169：共同 GPU-ready warm 参考的执行资格

## 问题与边界

D168 已固定既有输入，尚未测量 warm 数值。D164 的 GPU-ready 条件组仍含
Prime 控制/排队，不能作公共阈值。本项沿现有 preflight、受限启动、专用
runtime、原生时间线和物理释放路径增加显式 `native_warm_reference` 模式；
不改 Full 服务实现、九个公式、冻结 V1，不恢复外部基线实验。

复用 D168 索引 SHA
`563abfbdeb74c1bc03e42a26c34d48d634c342ac4fa43bd00c5ff8dd82269793`，
逐项复核原请求、adapter、target、canonical/native prompt SHA。没有新权重、
新 trace、远端缓存准备。GPU-ready 参考允许在计时前从既有本地相同工件加载，
明确不是主实验的真实远程 cold/first-touch 路径。

## 独立参考配置及原始依据

| 项目 | 当前 Full | 本参考候选 | 原因 |
|---|---:|---:|---|
| max_num_seqs / runtime cap | 4 | 8 | 实现 V1 batch8，不由 Full cap 推断硬件不可能 |
| max_loras | 4 | 8 | 原批次最多 8 个不同 ID 必须都已 GPU-ready |
| max_lora_rank | 64 | 16 | 依据既有池实际最大 rank，不重新生成工件 |
| GPU memory fraction | 0.70 | 0.92 | 安装版本与官方 0.30 的原生默认，实际 KV 仍须核验 |
| TP / dtype / prefix cache | 1 / FP16 / off | 不变 | 同模型与输入执行合同 |
| max context / iteration tokens | 1024 / 1024 | 不变 | 不改长度或预填充工作量 |
| 主机资源限制 | 72/80 GiB、swap 2 GiB | 不变 | 同一服务外部包络 |

[vLLM 0.30 cache 配置](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/cache.py)
与本机安装源码均明确默认 0.92；不是从旧指南猜测 0.9。
[LoRA 配置](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/config/lora.py)
与[原生 worker 管理](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
用于核查 rank/slot 与装载。仅据源码不能宣布本机性能好或 batch 可行。

这是**参考专用候选**，不宣称 Full 已切换到相同内部配置，不按 Prime 胜负
挑参考。3B 尚未检查，跨模型公共 batch 与最终数值 manifest 仍未冻结。

## 执行与否证

1. probe 在每组原八请求批次中选择完整声明 KV block 总数最大、不同 adapter
   数最大批次，稳定首个 tie；两者相同只执行一次。不用时延或输出挑选。
2. 原 prompt 预处理与 GPU 引用全部建立后，检查实际 slot 与空闲 KV；必须
   能容纳该批完整上下文（实际 block 大小复核）。不足仅表示此保守筛查失败，
   不假称任意调度下都不可行，不静默降 batch。
3. 同时提交八次普通原生生成，不使用 Prime router/planner/pending admission。
   计量采用原生 dispatch→first 与 first→last；不使用代理调用总时间。
4. 原生 token 数、prompt hash、adapter 引用、终态、计量恒等式逐请求检查；
   批后 drain、终态确认及释放，不累积上一批队列。错误保留，不兜底重试。
5. probe 通过后才开展每组 256 请求×3 轮。每组先保留一批独立 warmup，
   不混入三轮均值。此时仍须核对全部样本与实际资源释放。

参考的加载、引用与 warmup 位于 warm 时延测量边界外，但完整持卡仍保留。
本项不证明正确 adapter 的逐 token 执行映射；D167 槽位内容证据不重复运行，
`semantic_full_pool_qualification`/正式 G1/G2 保持 false。

## 当前状态表

| 检查 | 状态 |
|---|---|
| 既有 preflight 72 项 | 已通过（合并测试中的 72 项） |
| 新 warm 输入/计量/批次测试 | 既有 Python 3.12 环境中 13/13 通过（0.459 s）；首次解释器错误保留 |
| 测试解释器处理 | 原生运行要求 Python 3.12；warm 测试换到既有 3.12 环境，不改生成实现 |
| 实际 GPU probe | 4/4 批次、32/32 请求原生合同通过；实际八请求解码重叠 |
| 1,536 个 warm 测量样本、共同阈值 | 待执行/未冻结 |
| 7B G1/G2、新指标旧 Prime 对比 | 未达标，继续主线 |

失败测试日志保留。该失败发生在 CPU 测试夹具，未发生 GPU 分配或推理失败。
本项先交付资格表，不将工程资格画作性能收益。后续先完成 7B 验收，再 3B，
最后恢复外部 baseline、主比较、消融和敏感性。

## 实际 probe 结果（单次容量资格，不是 SLO 阈值）

| 输入组 / 批次 | 请求 / 不同 adapter | 所需完整 KV block | 批前 / 批后空闲 block | 八请求共同 decode 区间 (s) | 原生均值 TTFT / TPOT (ms) |
|---|---:|---:|---:|---:|---:|
| 0 / 3 | 8 / 8 | 283 | 1022 / 1022 | 0.192102 | 710.066 / 28.842 |
| 0 / 19 | 8 / 4 | 336 | 1022 / 1022 | 1.590025 | 746.211 / 28.510 |
| 1 / 15 | 8 / 8 | 453 | 1022 / 1022 | 0.061170 | 924.509 / 49.248 |
| 1 / 24 | 8 / 6 | 496 | 1022 / 1022 | 2.270834 | 928.894 / 29.813 |

实测 block 大小 16 tokens；每批提交前所有目标 adapter 均在真实 GPU slot，
无遗留 native 请求/iteration/deferred-free，批后同样清空。32 个 prompt/token
hash、固定输出数、adapter 引用与原生时间恒等式通过。原生引用全部正常释放，
最后 adapter 缓存清空，服务进程与 GPU context 退出得到独立确认。

本次整体持卡 117.769191 GPU-s（含初始化/准备/诊断/清理），不是 Full 的
GPU-s/request。144 次外置资源采样，服务内存峰值 7,570,870,272 B，主机
最小 MemAvailable 106,752,929,792 B；观测 high/max/OOM/swap 均为 0。
service/watchdog 均返回 0。147 项历史保护检查通过。

这些批次按容量极值选择，不能充当随机时延样本或三轮均值；表中高组 TPOT
49.248 ms 的观测同样保留，不筛掉较慢点。无需再重复本 probe。下一步使用
同一参考配置和 D168 既定索引，采集 1,536 个正式 warm 测量调用，另保留
16 个 warmup 调用。最终数值仍需逐请求检验、三轮分组均值与独立 manifest；
跨模型公共 batch 和完整数值身份仍未验收，不能提前宣布 G1/G2 达标。

文件：`paper_results/ieee_tc/p2_backend/20261002_d169_warm_probe.{json,csv}`。
原始证据：`results/ieee_tc/p2_backend_qualification/d169_20261002/`。
按计划 §11 和 `academic-plotting` 使用容量/资格表，不画虚构收益图。
