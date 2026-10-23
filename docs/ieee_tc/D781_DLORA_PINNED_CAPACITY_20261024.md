# D781：dLoRA 3B 启动容量诊断

2026-10-24。使用 `run-experiment`/`monitor-experiment` 复用已有受限运行与观测，
按 `analyze-results` 区分实际记录、解释和下一步。不是完成的性能实验，不作
相对性能、均值或 CI；不修改论文、旧结果或既定评价协议。

## 1. 实测记录

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d781_20261024`。
配置仍为历史 DP2/TP1、PERIOD_MIG、完整 500 池、原 100 请求前缀，共同真实
远端及固定生成协议。使用 D778 源码视图、D779 环境和 D780 按需 Ray 启动。

| 项目 | 实际记录 |
|---|---:|
| 远端制品准备完成 | 500/500 |
| 两个原生 worker 的真实 PEFT CPU 池加载 | 均记录 500 完成 |
| 原生初始放置 | 501 份，两个引擎为 250/251 |
| 首个引擎 KV 配置日志 | GPU 1173 blocks、CPU 4681 blocks |
| ready 回执／合格推理结果 | 0／0 |
| 服务观测峰值 | 80,859,029,504 bytes（约 75.3 GiB） |
| memory.high 事件 | 611,536 |
| memory.max／OOM／OOM-kill 事件 | 0／0／0 |
| 结束方式 | 依据容量推算主动停止诊断，非 OOM、非保护超时 |
| 实际资源释放 | 所属服务域移除，NVML 无计算进程 |
| 合格物理生命周期计量 | incomplete，不生成性能 GPU-s 优值 |

原生 Ray 仅启动所需的少量 worker，未再出现 D780 的 40 个预启动 Python
worker；但这不足以解决完整池的内存容量问题。不能将 D780/D781 的不同
主动停止时刻当成性能改善对照。

## 2. 为什么名义张量字节低估实际需求

实际保留环境为 PyTorch `2.1.2+cu121`。其
[固定版本 pinned allocator 源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/aten/src/ATen/cuda/CachingHostAllocator.cpp)
在新分配时使用 `cudaHostAlloc(PowerOf2Ceil(size))`。原生 dLoRA 为每层、每组
A/B 分别创建 pinned 张量；CPU KV 也逐层分别分配 K/V，而非一个精确大小
的连续池。[官方 CPU KV 实现](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/worker/cache_engine.py)
与[官方 LoRA 分配路径](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/worker/lora_engine.py)
支持这一分配方式；3B GQA 的具体形状取自本地已经保留的适配源码。

这里需计量的是每个仍存活分配的容量，而非张量有效元素之和。按 FP16、
28 层、最大槽位 rank16、500 adapters、hidden3072、KV heads8、head_dim128 推导：

| 每个 worker | 有效张量／名义容量 | pinned 分配容量推算 |
|---|---:|---:|
| 完整 CPU LoRA 池 | 25.634765625 GiB | 31.5 GiB |
| CPU KV，4681 blocks | 约 8 GiB | 14 GiB |
| DP2 合计 | — | **91 GiB** |

例如单层单个 CPU K/V 张量为 `4681×8×128×16×2=153387008` bytes，
向上取整为 256 MiB，共 `28×2` 个，因而每 worker 为 14 GiB。
LoRA 的八个张量逐项计算保存在 `capacity_derivation.json`。

**91 GiB 是基于实际版本及形状的静态分配推算，不是观测到的内存峰值。**
它尚未包含运行时、Ray、其他缓存及驱动开销，已超过共同 80 GiB 硬上限。
本轮在达到该总量前停止；不能写成“发生 91 GiB OOM”，也不能据此断言
dLoRA 在其他硬件或所有合法配置下不可行。

## 3. 停止与证据边界

只向物理分配回执绑定的本次服务父进程发送 pidfd SIGTERM；原 launcher
负责所属进程树清理。实际 `launch3b.json` 存在，返回码 1，服务域已移除，
GPU 已释放。D780 的进程退出清理修正本轮未再抛出 census 身份竞争错误。

停止后的内部报告含 `ingress did not stop; outer service cleanup remains required`；
该错误与未合格 native lifetime 的记录全部保留。外层清理最终完成不能用来
伪补原生物理账目。远端只关闭本轮绑定的两个服务及监控，不触碰共享 k3s。

Ray 原生日志、回放失败、传输记录、实际资源采样和停止回执均保留。仅在
实际退出后按既有逐文件 SHA、原件、引用及打开文件检查，回收重复 objects；
不删除原始池、失败证据、用户修改或历史投稿。

## 4. 有界的下一步

先核查原有 CPU KV 配置如何在共同资源包络中容纳，包括 allocator 取整及
真实非 KV 开销；若需调整，仅调整必要的 CPU KV 容量并在新身份中固定。
不提高总包络、不重新预启动空闲 Ray worker、不改 GPU 放置/迁移/调度策略，
不缩减完整 adapter 池，不开基线性能搜索。新容量必须先有计算依据，不能
以一次又一次 OOM 试出可运行值；也不能把适配后结果称为 dLoRA 全局最优。

当前尚未证明完整原生初始化、独立数值正确性或新合同 W1/W2。后续仍需必要
GQA/所选层和 adapter 核查及限定 W1/W2，然后 Loquetier/HydraServe 3B、
PrimeLoRA 完整对照优化、同版本 Full、A1–A5、S1–S13、离线成本与审稿文档、
最终图表；本次诊断不删除任何实验，也不被用作 PrimeLoRA 的性能优势证据。
