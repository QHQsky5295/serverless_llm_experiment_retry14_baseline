# D1066：零需求轮次的精确空规划消除

2026-11-05。D1065 为 PROGRESS；本轮为 PROGRESS，新增保留日志证据、
一个公式保持的候选及边界测试。目标 active；此页作为增量交接，保护用户
dirty 的 EXECUTION_STATUS.md，不把其 D998 历史头部当当前进度。

## 已完成证据和唯一假设

D1065 的 1130 个 completed/results=[] 不能直接等同零需求。
本轮复用 D1047 main_outcome、service_ingress 与实际冻结的 5 秒需求窗口，
将每次需求采样时刻限制在 epoch.started_at 与 planning_cpu captured 时刻
之间。对所有可能的 (t−5,t] 取并集；只有该并集中无 ingress 才认证为零。

| 既有 D1047 证据 | 数量 |
|---|---:|
| residency epochs | 2389 |
| 能关联规划 SHA 的 epochs | 2385 |
| 严格认证需求为零 | 260 |
| 已关联但需求未知 | 2125 |
| 未关联（取消） | 4 |
| 认证零需求且文件目标/执行结果均为零 | 260 |

这是保守触发下界，不是节省时长、资源或反事实时延。其余空执行可能涉及
容量、替换或已满足目标，不推断可跳过。分析目录：
`paper_results/ieee_tc/p2_backend/d1066_zero_demand/`。
首个分析 KeyError 和改正记录保留；最终分析峰值 131444736 B，
high/max/OOM/swap 均 0，正常退出。

候选：在当前控制轮、存在可规划副本时，读取权威 HotnessTracker 快照。
仅 total_arrivals=0 时省去该轮文件/native 状态读取、纯规划与空执行注册。
Eq.(4) 的 h_a=0，所有收益为零；Eq.(6–7) 不选任何正收益插入/替换。
不增加阈值、缓存 TTL、窗口、模型特判或未来信息。

- 既有准备任务继续排空，失败照常 reap；不取消正在进行的工作。
- 不发布虚假新状态、不注册计划或资源预留；独立 no-op journal 明确记录
  实际需求时刻、窗口、计数和本轮可参与副本。
- 请求路径 source/feasibility/admission、handoff、autoscaler 不改。
- 新需求在下一正常控制轮继续完整新鲜观察/原选择器；不会复用上一空计划。
- 这是当前采样的精确零收益简化，不保证与“先读取资源、稍后采样需求”的
  历史异步执行逐时刻相同。有需求在两者之间到达时可能改变准备时机，必须
  由完整 W1 验证净效应，不声称 260 轮全部 CPU 时间直接转成 E2E 改善。

没有实施“所有热 adapter 已 GPU-ready”扩展：它需额外位置证据和失效处理，
本轮未证明其独立触发/收益。没有重做 D163/D741 消费展开、D514/D516 查询
精简、D998 控制延后、跨 epoch 缓存、线程替代或 HOST 大小搜索。

参考官方 [vLLM v0.30.0 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)：
step 在无 scheduler work 时返回，同时 has_work 保留未完成 batch 工作。
借鉴的是“依据当前权威工作量避免空迭代，保留在途工作”的边界，不把其
无请求谓词等同本系统 trailing-window 需求，也不据此预测性能收益。

## 验证与完整回放登记

新增 5 项测试通过：零需求无 owner 读取/计划、半开窗口过期边界、新到达
恢复完整观察、在途任务及失败保留、禁用/无可用副本不采样、原选择器空集
等价（后两项按测试方法分组）。另运行 306 项通过，其中原 ControlRound
六项、已有零需求选择器一项、basic smoke 299 项。历史六项测试修正缺少
await 的调用以匹配已经异步的生产接口，不改其断言或生产选择逻辑。
终端 unittest 两次退出 0；没有把 CPU 测试称为 GPU 性能/数值正确性验证。
最初尝试用旧环境 pytest，因未安装该模块退出 1，测试未执行；未安装依赖，
改用现成 unittest。对应 scope 和失败记录另存。

完整性能执行只登记一次：

- RUN_ID=`d1066_20261105`；RUN_TAG=`d1066_3b_zero_demand_w1_full`。
- 原样复用 `configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml`，
  SHA `263bdcd6ff255108ddaffdd9ce46898d7f12be15948d7fe3d7ac27d5ea8828d2`。
- 沿用 D1000 入口，显式 W1、fresh_d747_v1；运行标签独占的 output/NVMe/HOST
  路径由现有入口覆盖，不写入旧配置路径、不重复建立源 trace 或工件池。
- 保留 D1031 profile、vLLM0.30、真实 D80 gzip 交付、60秒准备、4000请求、
  固定输出和共同资源包络。不增加前置短性能回放。
- 冻结指标 V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`，
  单轮 V2 SHA `193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`。
- 比较 D1047：95.900% 联合时延达成、6182.715971 GPU-s、1114.091001 ms
  P95 TTFT；W1 共同预算 6754.783565160236 GPU-s，既定要求 95%。
  也保留 ElasticOnly/ElasticLocality 已观测优势，不把胜过父版本当贡献证明。
- 核查 no-op 实际触发、完整输入/原生输出/资源释放与全部退化。单轮无 CI，
  不筛结果、不重跑同键。失败/负结果保留；n_correct=null 的独立算术正确性
  边界不因结构测试变为合格，不宣布正式 G1/G2 最优。

## 保护、备份与完整主线

启动前继续使用磁盘150GiB、空服务内存102GiB门槛；运行中100GiB停止线、
72/80GiB服务域、独立watchdog、整树释放不变。不并行压缩/清理或其他GPU工作。
147旧投稿项与两项dirty SHA不变，唯一资产/失败证据不删除。
按 github-sync 使用显式白名单检查、提交，推 origin V2 与 FaaSLoRA 独立
备份分支；分歧的实际 FaaSLoRA V2 不强推，备份不冒称已完成集成。

HydraServe 已采用隔离 rootless Podman 完成两模型既定比较；不操作共享k3s，
保留 pp1/未验证完整 pipeline/consolidation 收益的限制。
两模型外部比较与3B A1十一组均保留。仍待：Full净收益优化、7B A1十一组与
两个累积点、受控A2–A5、M2、S1–S13、CE/价格/Pareto、B1/B2/C1/C2/C3、
公式推导/新颖性/两轮审稿文档、IEEE图表、最终同版本来源统一及V2集成。
本页登记不是启动/完成回执，实际运行状态须由PID、launcher和终态文件确认。
