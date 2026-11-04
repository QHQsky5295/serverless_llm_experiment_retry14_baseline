# D1063：NoHOST 的受管复用边界（只读复核，尚未实现）

2026-11-04。D1062 LastKnown 单轮 3B W1 正在测量。本页落实既定 A1 的
NoHOST，不新增矩阵、不修改当前服务源码，不是新性能结果。

## 要回答的问题与定义

在同一 GPU/NVMe 预算、路由、autoscaler、handoff 和 admission 下，移除
**独立受管 HOST 复用**，是否改变实际来源、准备代价、联合 SLO 和物理 GPU-s。
保留后端维持 GPU 副本必需的 CPU backing 及本次请求/计划的瞬时 staging。
不能将本消融表述为零 HOST 内存、零 CPU 张量、零 page cache，或完全消除
主机内存作用。Full 当前弱于 ElasticOnly/ElasticLocality 的反例保留；NoHOST
可能改善、退化或无影响，不以预期胜负决定准入。

## 复用的历史与新增核查

- D1049 已指出 `host_capacity_mb=0` 不会关闭 native tensor cache，native
  budget=0 又会破坏必要 staging；本次不重试这两个已排除方向。
- D41（`P1_FORMULA_IMPLEMENTATION.md`）区分受管文件、native CPU 张量和
  pinned allocator 留存。删除对象不等于物理字节已归还。
- `IEEEBackendGPUReferences._refresh` 要求 GPU-active IDs 是 CPU-registered
  IDs 的子集，且活跃引用继续 pin 两者；不能删除 GPU-active backing。
- `configure_host_budget` 的正预算及 demand/proactive workspace 继续保留。
- 新核查：`owned_preparation_inputs` 的来源构造路径将 GPU backing 同时
  列为 HOST 副本；`owned_gpu_planning_victims` 用它计算 GPU 驱逐后的损失。
  `proactive_host_prepare_and_acquire` 的 GPU 替换事务还检查原 CPU fallback
  保留。**只在事后清理 CPU-only 对象，会使 planner 先按不存在的长期
  HOST fallback 定价，因此不是足够的 NoHOST 实现。**

实际函数定位以当前源码为准：
`faaslora/memory/residency_manager.py` 的 `_refresh`、`configure_host_budget`、
`_collect_staged_host`、`release_host_source`、`close_preparation_plan`、
`source_snapshot`、`proactive_host_prepare_and_acquire`；
`faaslora/preloading/preloading_planner.py` 的来源构造及
`owned_preparation_inputs`、`owned_gpu_planning_victims`、
`owned_gpu_execution_objective`、`native_gpu_fallback_costs`。

2026-11-04 另核查官方 vLLM v0.30.0：
[worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
先在 CPU 加载，再注册并激活；
[model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
区分 registered/active cache，移除 registered adapter 会同时失活 GPU。
这支持保留必要 backing 的边界，而非证明任何消融收益。在线最初查询旧
`lora/models.py` 返回404，已改查实际 `lora/model_manager.py`，未以失效链接
作证。已安装文件还包含本项目已验证的 mapping/slot 修正，不能与官方字节
或原始实现完全等同。

## 最小必要实施范围（等待 D1062 完整结束后）

1. 显式、默认关闭的 NoHOST opt-in：禁独立 HOST 最终目标及长期保留，
   覆盖受管 HOST 文件和 native CPU-only cache；不能关闭整个 hierarchy。
2. GPU/NVMe 规划与 GPU 准备仍可执行；其必要 CPU 暂存须绑定具体请求/计划。
   依赖终止后回收，不能把暂存发布成其他请求可任意复用的 HOST hit。
3. 在原 owner 的串行生命周期/事务边界回收 GPU-inactive、无引用、无外部
   pin 和无在途依赖的 CPU 对象。不加周期全池扫描，不绕开 owner 直接 pop。
   GPU-active backing 保留并如实计入占用；source witness 随对象退役失效。
4. planner/owner 区分“物理仍在的 backing/staging”和“可长期复用来源”。
   GPU 替换后不再有 HOST fallback 时，以实际 NVMe/remote 来源和相同冻结
   需求/成本计算损失；保留原收益/损失公式，不只改 tier 标签或删除旧断言。
5. Full 默认路径及旧 HOST-fallback 测试不变。NoHOST 的替换事务同时检查
   新 fallback 可用性、相同 victim/slot、引用与 completion fence；不能
   通过破坏 GPU 缓存制造 HOST 消融的退化。
6. 保留 HOST 总包络、暂存峰值、真实 allocator 计量和取消 join；不把空出的
   HOST 预算转给 GPU/NVMe（后者属于 S13 另一设计点）。

实施前仍需逐调用核查 mode 在 owner 初始化、文件层和 planner 的透传；本页
不是已经完成的接口设计或“可安全直接运行”的证明。不要求先做整个后端的
CPU/GPU 对象分离，除非实际实施出现不能满足上述不变量的新证据。

## 针对性测试与机制证据

- GPU-active backing 保留；GPU 失效且最后依赖释放后 CPU-only 对象退役。
- demand/proactive staging 的完成、取消、重复释放；没有跨请求隐藏复用。
- 同一个 GPU 替换 fixture：Full 仍按原 HOST fallback；NoHOST 按真实
  NVMe/remote fallback 定价并一致执行。复用 `test_ieee_tc_gpu_references`。
- 复用现有 pin/lease、取消、capacity 和源码 identity 检查，不重做整池资格。
- planner 不选纯 HOST 最终目标，但 GPU/NVMe 与必要 staging 仍可达。
- 报 backing/staging/独立 HOST 的对象与字节、来源类型、回收/引用冲突及
  计量限制；物理预算与非目标配置的差异必须为零。

通过后只运行既定一个 3B W1 NoHOST 执行键，复用 D1047 Full，不添加无理由
GPU smoke 或同配置重复。原始输出变化、失败与负结果都保留。

## 复核身份、顺序与未完成范围

使用 `ablation-planner` 的审稿视角，由已有 reviewer 只读复核，主 agent 已
核对上述源码。技能示例中的旧模型不可用，本次是同系列 provisional 复核，
不冒称指定模型或独立外部评审；无代理测试、GPU、文件或服务修改。

先完成 D1062：资源释放、共同指标/新信息视图审计、解释、备份；再按本页
实施 NoHOST。3B A1 当前已完成9/11，D1062运行中，NoHOST未实现。7B A1及
两个累积节点、受控 A2–A5、M2、S1–S13、CE/价格/Pareto、B1/B2/C1/C2/C3
和两轮审稿文档、IEEE图表、最终同版本来源统一及FaaSLoRA V2集成均保留。

## 调用链补查（2026-11-04，D1062 测量期间只读）

- `scripts/run_all_experiments.py:_attach_ieee_host_budget` 在原生准备之前安装
  每 owner 的 HOST allowance，并校验共享 owner、workspace 和进程身份。
  NoHOST 生命周期模式应在这个已有安装边界冻结并读回；不能运行中切换，
  也不能以修改 profile 的内存上限代替模式声明。Full 默认合同保持原样。
- `freeze_owned_planning` 已将输入冻结传给 `owned_preparation_inputs`。
  模式必须同时覆盖未初始化 activation 和已初始化 native owner 的来源构造、
  HOST 目标筛除和 GPU victim 定价；仅在最终 selected 列表删除 HOST 会留下
  不一致的目标函数或中间来源。原生物理 footprint 仍保留，不将 backing 字节藏掉。
- 当前 GPU 准备控制路径只在存在 `staged` 对象时收集并传递
  `host_file_fallbacks`；普通已注册 HOST→GPU 路径不传。NoHOST 的 GPU 驱逐
  损失需要实际 NVMe/remote fallback，因此这一条件也必须覆盖 NoHOST，
  并复用现有 hold/release、取消 join 和内容核验，不能仅在 worker 换损失数字。
- `native_replacement_costs` 已在同一次新鲜 footprint 观测中计算 GPU/HOST
  两种损失；`_native_host_replacement_costs` 已验证实际文件 lease 或不可变
  remote fallback。可以复用这些校验与冻结成本，但对当前 GPU backing 的
  退役损失必须采用 GPU→合法后继来源的差值，不额外加一次 HOST 损失。
- 不在 `_refresh` 中无条件清理所有 CPU-only 对象：该函数也在 staging
  注册、GPU 激活之前调用，清理会破坏尚未完成的事务。回收应落在完整 owner
  事务结束、最后依赖释放和 plan close 等安全边界，保留外部 pin。

以上仍是实施依据，不是代码完成或测试通过。本次再次在线核对同版本官方
worker/model manager；尚未调整运行中 D1062 的任何 serving 文件。

13:14交接：D1062已完成并通过离线核验，结果见
`D1062_LASTKNOWN_OUTCOME_20261104.md`；3B A1进度为10/11。NoHOST仍仅完成
上述边界和调用链核查，没有新增生产实现或测试，不可直接启动NoHOST回放。
先备份D1062已核验产物，然后实施本页最小路径。磁盘维护已完成，下一次
重型启动仍重新检查实时余量，不把151.79 GiB历史值当永久保证。
