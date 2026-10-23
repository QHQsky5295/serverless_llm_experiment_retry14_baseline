# D777：dLoRA 接入共同请求与物理 GPU 生命周期

2026-10-24，PROGRESS。继 D776 后完成现有服务入口的必要连接，34 项 dLoRA
CPU 检查及 27 项共享服务回归通过；尚未启动真实 GPU、远端服务或性能回放。
本轮不是新的 PrimeLoRA 优化，不搜索 dLoRA 参数。

## 原因与保真边界

D774–D776 已有原生 token 时序、canonical 输入与私有 Ray 上下文，但还没有
共同 HTTP/parent allocation 连接。历史 3B 结果为 R2：可复用模型、权重、
环境和 DP2/TP1 参数，不能补标签变成当前 W1/W2。

本轮再次核对固定官方
[EngineManager](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/engine/engine_manager.py)
和 [API](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/entrypoints/api_server.py)：
请求在原生 select_lock 下选择 engine，然后进入原生生成流。此结构继续
保留；没有套用现有 resident runner 的 RoundRobin，也没有换放置/周期迁移
公式。原生 CPU engine actor 和 GPU worker 是不同对象，不假定 wrapper
暴露 model_config；vocab_size 从实际 GPU worker 的模型配置回执取得。

## 实际接入

复用 `replay_vllm_engine_trace.py` 的 resident 服务、现有 pre-ready ingress、
`prepare_slora_full_pool` 的统一已发布工件物化、外置 HTTP 回放及物理账目。
函数旧名不把 dLoRA 改标为 S-LoRA；使用独立 backend 和传输 request ID。

- 共同部署通知后物化全部既有 500 adapters；无未来热点、权重生成或本地
  frozen 回退。只新增小型 ID 索引，供保留的原生 subset loader 读取。
- CPU-only 准备不虚构 GPU-s。准备完成后，实际分配两 GPU，再启动唯一
  manager 子进程；worker config 绑定此前 acquire journal/lease。
- 新进程使用原生环境和项目私有 Ray；原生 EngineManager 初始化两个 TP1
  worker。准备/初始化期间业务仍按共同到达时刻排队，不平移业务时间。
- manager ready 回执包含两个 GPU worker。外层对照实际 NVML、PID/birth、
  分配 UUID、完整 adapter 映射和准备 SHA，再放行等待的请求。
- 前端只等待 manager，不选择 GPU 副本。保留原生 EngineManager 选择和
  迁移，实际选择结束记为 assignment；选择过程计入调度等待。另保留
  manager-entry/selection-start，首末 token 仍取原生生成事件。
- 迁移请求保留 native_engine_path 及其中所有 engine 的 GPU worker 身份；
  不把最终 worker 当成请求全生命周期唯一执行者。共同校验检查原生 IDs、
  prompt、adapter、时间顺序、TPOT、准备及 worker receipt。
- 完成后复用已有精确 Popen、kernel exit/NVML 清空、release 和终态汇总。
  初始化失败仍保留未合格资源账目，不伪造成功 release。
- MAIN 的物理终态校验只增加 dLoRA 原生 timing/token-source 对，原有全部
  完成条件和资源积分语义不变。

`ready_instances_at_enqueue` 在此 manager 入口表示 manager 是否已就绪，
不是两张 GPU 的占用数；物理资源与经过的 engine 分别由专门回执/账目给出。

## 已执行验证

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d777_20261024/`。

| 验证 | 实际范围 |
|---|---|
| 34 项 dLoRA 检查，0.417s | 原有 27 项，加 7 项配置/环境、pre-ready 等待、原生选择至共同统计/物理终态、错绑定拒绝、500-ID 准备接口、两 GPU worker 回执、CLI 分派 |
| 27 项共享服务回归，2.164s | S-LoRA 现有配置、HTTP/启动/清理路径，确保共用入口扩展未改变既有行为 |
| 原生方法 CPU 合成输出 | 实际调用保留的 select_engine/generate/tracker；第二 engine 成本低时确实选择它，非排序 adapter 池按原整数映射绑定 |
| 资源/保护 | 峰值63,643,648 bytes，high/max/OOM/swap为0；aux1已结束并关闭空域；145项旧投稿和三项用户dirty SHA不变 |

这些测试不调用实际 GPU 内核或 Ray 集群；共同终态测试里的成功是合成输出
的合同匹配，不是独立数值正确性。dLoRA 整套外层服务＋原生两 GPU 的实际
联通仍须由下一次必要资格验证，不能将上述共享服务回归泛化成完整 GPU 证明。

## 下一实际动作与停止条件

代码连接已完成，下一转真实 DP2/TP1 加载/必要 GQA 和 LoRA 检查以及共同
HTTP 资格；不再重复没有变化的接口检查或追加基线参数搜索。实际原生启动
若暴露具体兼容问题，只修复有证据的路径并保留失败。不能因 SLO 低而调整
dLoRA 策略或放宽公共要求。之后完成限定 3B W1/W2 各一次。

Loquetier/HydraServe 3B、PrimeLoRA 全对照与有证据优化、同版本 Full、
A1–A5、S1–S13、成本/CE、两轮审稿文档及最终图表全部保留。独立数值
正确性仍未知；整体目标未完成。run-experiment 约束入口/资源复用，
github-sync 保存显式检查点；不修改其他项目 k3s，远端服务本轮未启动。
