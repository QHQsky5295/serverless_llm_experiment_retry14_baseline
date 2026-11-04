# D1063：NoHOST 在途 staging 与 demand 接管

2026-11-04，接续 `D1063_NOHOST_FILE_OWNER_20261104.md`。本记录替代其
“尚未实现 staging 授权”这一局部状态，不删除其他待办。3B A1 性能点仍为
**10/11**；本轮没有 GPU 回放、没有启用 NoHOST，没有新的性能优胜结论。
用户已修改的 `EXECUTION_STATUS.md` 保持原字节，以此增量文档交接。

## 科学问题与已实现边界

NoHOST 关闭的是可独立服务的 HOST 缓存，不是 GPU 激活必需的 CPU backing
和仍在进行的准备操作。此前“所有未关闭计划的 targets”既用于注册对象的
退休保护，也延长 staging 存活，会把已完成目标留下的临时对象变成隐蔽缓存。
本轮将这两种生命周期分开，不改变论文公式、GPU/NVMe 容量或 demand 优先级。

- NoHOST 新的 file→CPU 准备即使注册缓存有空位，也先保持未注册 staging；
  必须属于同 name/content 的 pending GPU 目标。Full 保留原有注册策略。
- 每个新 staging 绑定创建 lease、plan、adapter、content SHA、source_id；
  异步构造另绑定 construction token。既有对象不重新标记来源。
- 同一 native worker 的现有 `demand_load_and_acquire` 内原子接管：先验证
  正常 owner/tier/source 冲突，再确认文件 witness、原构造回执、精确 staging
  对象和至少一个匹配的 pending plan；随后仍走 demand 的 pin、容量、预算、
  fence 和 GPU lease，不增加 RPC，不等待整个 proactive GPU admission。
- 创建 plan 已关闭时，另一个连续存活的匹配 pending plan 可以保持操作；
  完成目标不授予复用资格。最后 pending 结束且最后真实 staging 引用释放后
  回收对象；后来新 plan 不能复活旧 source_id。注册 backing 的 all-target
  退休保护暂保持已验证实现，不与上述 staging 规则混用。
- 回执保持实际 `source_tier_before_acquisition=staging`，额外记录
  `nohost_staging_join`，其 `independent_host_hit=false`。旧 CPU movement
  receipt 只证明构造来源，不冒充仍存活的 HOST 引用。真实 CPU lease 尚未释放
  时，join 后仍由原 owner 释放，GPU demand lease 继续保护 backing。
- `native_staged_source_reused` 仍为真，既有 profile 过滤语义不变；不能把
  共享 staging 当作完整 file load 或零成本 HOST 样本。

## 依据、复核、检查与失败保留

已完整重读计划、目标及 D1063 前三块，未重试零 HOST budget、扩大 GPU/NVMe
或等待整个 GPU movement 等被排除的方向。本轮再次核查官方 vLLM v0.30.0
[worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
与 [model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)：
CPU 对象加载/注册与 GPU 激活是相连但不同的生命周期。这是本机消融边界的
实现依据，不是上游已有本系统 staging 协议或本机性能提升的证据。

沿用 `ablation-planner` 的既有 reviewer 进行只读 provisional 复核；发现
旧 GPU 选择在到达 worker 前变成 staging 时，必须先返回正常来源失效，而
不能先要求文件 witness 导致异常。已调整检查顺序并加测试；不是独立外审。

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1063_20261104/staging_join/`。

| 检查 | 结果与含义 |
|---|---|
| staging_tests1 | 87 项，1 个测试断言失败；测试把新增 plan 前后快照错误相比较，原日志保留 |
| staging_tests2 | 88 项全部通过；修正上述断言，并增加旧 GPU→staging 来源冲突检查 |
| staging_integration1 | 1 项既有 Full arriving-demand/staging/proactive-recheck 集成检查通过 |

88 项覆盖新的 13 个 staging 用例及既有 native demand、protected-file demand、
private construction、file→HOST、NoHOST lifecycle 回归。两轮 source SHA
均前后一致；第一次失败不是性能失败，不能抹除。CUDA 不可见，CPU 2/3/26/27，
memory high/max 3/4 GiB、swap 0。前两轮峰值分别 609,165,312 与 609,120,256
bytes，high/max/OOM/kill/swap 均零。测试模型是 CPU contract double，不声称
验证真实 CUDA 数值正确性；主性能运行仍未增加。
独立 Full 集成检查峰值 610,852,864 bytes，同样无 high/max/OOM/kill/swap，
source SHA 与最终 88 项检查一致；其轻量文件服务是测试 fixture，不访问真实远端。

## 必须继续的同一主线

不能启用半实现 NoHOST。下列不是额外门槛，而是尚未关闭的实际使用路径：

1. 已注册、已失活且仅被旧 plan 保留的 CPU backing：禁止独立 HOST 复用；
   有实际 file witness 且无引用时 retire→真实 reload。实际引用冲突等待
   精确 release，不等待整个 plan、不忙轮询。`hold_host_source` 的一般 HOST
   授权仍未关闭，既有 lifecycle fixture 的 file_fallback 仍不是最终合同。
2. `_ieee_protect_selected_source`、`prepare_gpu_source` 不能因为任意 physical
   CPU row 存在就否决文件来源，必须检查 GPU 或 `host_reuse_allowed`，否则
   NoHOST 可重复重选。实验层开关尚未贯穿全部 owner。
3. file witness 允许旧 admission epoch；当前 capacity waiter 却要求等于
   拒绝 receipt epoch。等待时应绑定已验证的本次 owner/epoch/blockers，
   不能复用旧容量决策。join 后 `proactive_staging` HOST lease 的等待要绑定
   **CPU movement 实际 release ACK**，不能把整个 GPU plan 当释放证据。
4. NoHOST 每次 GPU replacement 都取实际 NVMe/remote fallback；文件替换的
   零损失后继须持真实 GPU lease，不以 CPU lease 换名。复用前块的计价实现。
5. 全链路一致后运行既定单个 3B W1 NoHOST，复用 D1047 Full；不补同键性能
   重复，不筛除负结果，不因未胜出改评价协议或对手。

## 保护、共享集群与余项

用户确认现有 k3s 不是本项目部署；保留共享配置和工作负载。HydraServe 的
隔离 rootless Podman 适配及 D807/D808/D809 的 3B W0/W1/W2 已完成，继续
复用，不重新部署。已测 pp1 不冒称验证 pipeline/consolidation。

本轮未操作远端服务/缓存、未生成权重或 trace、未覆盖旧结果。开工时磁盘
可用 162,947,997,696 bytes，约 151.76 GiB；保持原推理机磁盘与防 OOM 门槛，
下一重型任务前重新核查。两项用户 dirty 及 147 项旧投稿文件按原 SHA 保护。

剩余范围完整保留：NoHOST 3B、7B A1/两个累积点、受控 A2–A5、M2、S1–S13、
CE/价格/Pareto、B1/B2/C1/C2/C3、公式推导/新颖性/两轮审稿文档、IEEE 图表、
canonical 来源统一和 FaaSLoRA V2 非强制集成。反例与 3B 数值正确性尚未独立
证明的限制继续保留。按最新目标暂不为未达目标的 Full 生成展示图。

依据 `github-sync` 保存允许路径的测试检查点与 curated 证据；实际提交和
已核验远端 ref 以 RAW `backup.json` 为准。独立备份 ref 不冒称分叉的
`faaslora_origin/retry14_continuous_queue_v2` 已整合。
