# D209 — 取得执行引用前的实时身份查询

2026-10-03。父版本 `a19365ef15f9d03885d1981b2e28c4fbbc445955`。
状态：CPU 资格完成，普通 Full 待执行；不宣称性能改善。

## 主线与可证伪问题

D203 普通 7B W0 完整回放已封存。平均 TTFT 1.741609 秒，其中推理前
等待 1.447967 秒（83.14%）；D204 分解的 source-to-handoff 均值
441.574187 ms。两者不能全部归因于盘点，不能据此估算本候选的收益。
G1/G2、共同 reference、旧/新 Prime 对照和正确性验收仍未完成。

D174/D178/D181/D184/D194/D202 已分别处理描述表、所选低层副本复核、
初始路由查询范围、pending 流水线、planner 观测及物理 HOST 表示，均不重复。
本次找到的是另一个消费者：`_acquire_runtime_gpu_reference.observe_source`。
初次文件来源加载、缓存冲突或容量等待后的重新观察只使用 live owner、
epoch、clock、工件整数/逻辑身份、source incarnation、tier 和 path；
所返回的完整 footprint 仅写入诊断字段，不参与该处选择或容量决策。

假设：对这一消费者复用已有身份查询，去掉重复且未消费的全缓存物理图，
保持实时查询次数、owner 事务及物理容量检查，可减少同步控制工作。
组件行为等价/查询次数不是端到端加速证据；需同配置普通 4000 请求验证。

## 官方依据与边界

- [vLLM 0.30 worker manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)
  `LRUCacheWorkerLoRAManager.add_adapter` 在单线程 core 内检查注册状态，
  再加载、处理容量与激活；身份查询与物理分配是不同责任。
- [vLLM 0.30 model manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/model_manager.py)
  的原生 CPU/GPU 缓存、packing 和 slot 操作继续使用，不移到非 owner 线程。
- [vLLM 0.30 CPU 资源说明](https://docs.vllm.ai/en/v0.30.0/configuration/optimization/#cpu-resources-for-gpu-deployments)
  说明 engine core 对 CPU 争用敏感；不借用其结论证明本机具体开销或收益。

原生 `IEEEBackendGPUReferences.demand_load_and_acquire` 仍在同一 owner 事务
中重查当前 epoch、source、slot/pin、完整 HOST tensor budget，分配前后
都调用原 allocation checker。GPU pool/copy contract、proactive admission、
planner、首次路由的实测 class 和所有九个公式保持不变。

不新增状态缓存、不复用旧 verdict、不放宽资源或超时、不增加并发。
身份观察使用已资格接口 `ieee_source_identities`，更新已有单调身份记录；
不覆盖实测 footprint，不把未测容量填写成 0。该 acquisition 诊断明确记录
`selected_source_identity` 与 `acquisition_observation_scope=native_identity_only_v1`；
移除该处原来的全缓存容量字段。完整物理证据仍在原来的分配/观测记录中。

## 资格与交付

复用 D178 精确父方法对照和 D202 受限 CPU runner，使用现有工件和负载。
检查五类来源、实时变动、容量等待、stale 重查、未知身份、取消、释放；
新测试要确认没有全图查询，并且不能绕过实际加载的预算拒绝。
用既有 D203 小投影核查目标调用是否真实发生，不重投影大日志。
定向测试 → 受影响回归 → 父/候选组件对照 → 状态表 → 校验备份。

按计划 §11 和 academic-plotting，本轮使用精确资格表，不绘制虚假 GPU
性能曲线。资格通过后仅做一次同 D203 配置的普通 Full；所有历史结果、
失败证据、用户修改保留。7B 达标后才推进 3B，然后恢复外部 baseline。
Remote 已发布缓存继续复用，不重新生成；此 CPU 资格不启动远端服务。

Plan SHA `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`；
MetricV1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`。

## 实际资格结果

| 检查 | 结果 | 解释边界 |
|---|---|---|
| 定向检查 | 222 项通过，2.862 s | 含新增 3 项：实际预算前后检查、预算拒绝、错时钟拒绝 |
| 受影响回归 | 1,144 项通过，157.105 s | 与定向检查重叠，不相加为独立样本 |
| 五类来源的精确 Git 父方法对照 | 10 条记录均通过 | 相同层级、加载事务、HOST 引用、释放结果 |
| D203 既有小投影 | 4,000 请求身份唯一、SHA 匹配 | 不重投影完整日志，不重跑请求 |
| 新 GPU/远端运行 | 0 | 未验证端到端改善，不作 G1/G2 或模型验收 |

| 来源 | 全缓存物理图：父版→候选 | 目标物理图：父版→候选 | 实时身份查询：父版→候选 | 实时查询总数 |
|---|---:|---:|---:|---:|
| GPU | 0→0 | 1→1 | 0→0 | 1→1 |
| 原生 HOST | 0→0 | 1→1 | 0→0 | 1→1 |
| HOST 文件 | 1→0 | 1→1 | 1→2 | 3→3 |
| NVMe | 1→0 | 1→1 | 1→2 | 3→3 |
| Remote | 1→0 | 1→1 | 1→2 | 3→3 |

此表是 CPU 小样例的调用路径计数，不是 GPU 性能。每条样例仍恰好执行
一次 demand-load/reference 事务，最终 GPU/HOST/request 引用均为零。
来源变化仍由实时身份和原子 owner 事务处理，不省略实际确认。

既有 D203 dispatch 来源为 GPU 1,189、原生 HOST 1,766、HOST 文件 36、
NVMe 873、Remote 136。后三类合计 1,045/4,000（26.125%），说明此次
消费者并非只在人工小样例中可达；这些来源计数不是实际 acquisition RPC
次数，也不能据它们推算能节省多少延迟。冲突重查还可能影响其他来源。

前两次定向检查各有 1 个旧测试字段错误：同一旧测试把身份记录当作完整
容量记录读取，两处断言分两次修正；原源码、日志、失败记录保留。生产
候选在三次定向检查之间未改变，未放宽断言、预算、超时或系统边界。
取消、stale、容量等待、HOST promotion、错身份和释放检查继续覆盖。
所有 CPU 任务使用 3/4 GiB、swap0、CPU2,3,26,27；无内存护栏事件。

接受结论仅为“保留候选进入一次同配置普通 Full”，不是性能接受。
若完整回放不改善目标或有副作用，则据实保留该结果并回到瓶颈分析。
输出差异、共同 warm/Resident、旧/新 Prime 新口径和正式重复仍未闭环。
