# D776：dLoRA 私有运行域与原生 worker 生命周期接口

2026-10-24。PROGRESS；27 项 CPU 检查及原生环境导入通过，不是 GPU 性能
结果，不宣布完整复现或独立 LoRA 数值正确性。继续既定 dLoRA 3B W1/W2
必要适配；不搜索基线性能配置、不修改其他项目 k3s。

## 1. 为什么需要这一步

D774/D775 已补原生 token 时间线和共同输入。历史 dLoRA 结果仍缺实际 GPU
allocation/release；不能从模型 ready 时间倒推全部持卡时间，也不能把逻辑
Ray GPU 数量当作物理占用。原生 EngineManager 在 model init 之前先创建
workers 并初始化 NCCL，因此保护和分配必须先于这个阶段。

核查固定官方实现及保留环境：

- [dLoRA ray_utils](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/engine/ray_utils.py)
  与 [llm_engine](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/engine/llm_engine.py)
  用于确认分布式初始化顺序；本地保留的兼容版逐文件 SHA 另行绑定。
- [Ray 2.51.2 worker](https://raw.githubusercontent.com/ray-project/ray/ray-2.51.2/python/ray/_private/worker.py)
  及本机安装源码确认：显式 address=local 创建新节点，不因环境 RAY_ADDRESS
  连接既有集群；shutdown 关闭本 driver 所拥有的节点进程，不调用全局 ray stop。

本轮仅为既有 DP2/TP1 路径接入必要的安全、身份和测量接口；不改变原生
placement、迁移策略、调度评分、LoRA 运算或冻结的历史性能参数。

## 2. 实际修改

现有 prepare、measurement 和 replay 三个脚本小幅扩展：

1. source view 的可选 worker-lifecycle 模式在原生 RayWorker 构造时核查当前
   实际资源域、外置保护、live allocation owner 和两 GPU 分配记录。要求
   已有 canonical-input 模式；默认历史入口不变，保留 vendor 原树。
2. 每个原生 worker 可返回 PID/birth/cgroup、GPU 可见范围、静态 adapter
   整数映射、实际 GPU LoRA 容量及 KV blocks。原生放置会重新设置容量，
   因此不把输入 gpu_capacity=8 当作最终驻留容量。
3. 现有 runner 增加私有 Ray 生命周期上下文：DP2/TP1、单本机 Ray 节点、
   40 个逻辑 CPU、全服务共 8 GiB object store、独占 spill 目录。操作系统
   的 72/80 GiB、swap、affinity 和 watchdog 仍由既有保护接口核验；Ray 的
   逻辑声明不替代实际隔离。
4. 调用原生 EngineManager，初始化结束后核对真实 worker 回执。退出和构造
   失败均关闭所拥有的 Ray；不使用历史 wrapper 的全局清理命令。

重要边界：这仍是待接入的生命周期上下文，**本轮没有真实分配 GPU**。完整
HTTP service、parent Popen 绑定、实际 NVML worker 确认、等待退出后 release
尚待连接现有 TC 入口。上下文要求外层已创建真实 allocation；不能直接
运行它后声称完整生命周期计量合格。

## 3. 验证与资源

RAW：`results/ieee_tc/dlora_qualification/d776_20261024/`（原实验目录下）。

- 27 项 CPU 检查全部通过，0.105 秒；包括 D775 的 21 项和 6 项本轮检查。
  新增检查覆盖私有 Ray 参数、原生构造 hook、实际 worker receipt、原生
  构造失败/部分 Ray 初始化失败时的关闭、source-view 来源与旧树保护。
- 实际旧环境 `/home/qhq/anaconda3/envs/dlora_medusa_clone_20260519/bin/python`
  导入成功：Python 3.9.25、torch 2.1.2+cu121、Ray 2.51.2；加载本轮 source
  view 的 EngineManager/AsyncLLMEngine/RayWorker，构造参数与 SamplingParams。
- 导入过程 CUDA 未初始化、Ray 未初始化、模型未加载。它验证环境/API
  兼容，不验证 NCCL、GPU 内核、真实迁移或实际输出。
- 资格资源域内存峰值 684,384,256 bytes，high/max/OOM/swap 均为 0。进程
  已结束，空资源域已关闭；145 项旧投稿保护和三项用户 dirty SHA 未变。
- 无新权重、负载、完整解压池或远端服务启动；共享 k3s 未触碰。

## 4. 下一主线与交付范围

下一步把本轮上下文接到现有共同 HTTP、pre-ready ingress、远端全池准备
及 PhysicalGPUAllocation 接口；dLoRA 请求仍由 native EngineManager 选择
副本，不改成轮询。之后补必要的实际 LoRA 加载/GQA 检查、共同计量验证，
仅完成既定 3B W1/W2。数值正确性未知时继续明示，不以 HTTP 200 或 token
计数代替证明。

Loquetier/HydraServe 3B 对照、PrimeLoRA 完整比较及有证据优化、同版本 Full、
A1–A5、S1–S13、CE/Pareto、两轮审稿文档与最终图表全部保留。无基线参数
搜索，无无变化的广泛重验。本轮 run-experiment 技能约束资产/入口复用，
github-sync 技能用于显式路径检查点；MAIN 继续安全 backup ref，不 force
同步已分叉的 V2，不将备份分支称为最终 V2 交付。
