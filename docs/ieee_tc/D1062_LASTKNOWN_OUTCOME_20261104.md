# D1062：3B W1 LastKnown 单轮结果与状态传播证据边界

2026-11-04。唯一 D1062 回放于12:26启动、13:05完成资源收尾。4,000条请求
全部成功终态；未重跑 Full，未修改运行中的 serving 源码或远端服务。
本项完成后3B A1为10/11；NoHOST尚未实施，不把代码边界复核计作性能实验。

## 1. 原始比较

两组使用相同 source42/W1、4,000请求、adapter/prompt/target、固定输出合同、
共同 D648 阈值、D657 资源参考和V1/单轮V2。Full复用D1047；唯一配置差为
LastKnown来源视图的两个字段及三项本轮独占输出/缓存路径。

| 指标 | Full：D1047 | LastKnown：D1062 |
|---|---:|---:|
| 成功终态 / offered | 4000 / 4000 | 4000 / 4000 |
| TTFT达标 | 3972 | 3987 |
| TPOT达标 | 3858 | 3875 |
| 联合TTFT/TPOT达标 | 3836 | 3867 |
| 联合达成率 | 95.900% | 96.675% |
| 生命周期物理GPU-s | 6182.715971 | 6021.305073 |
| GPU-s / offered-request | 1.545679 | 1.505326 |
| 平均TTFT（ms） | 454.156048 | 439.532431 |
| P95 TTFT（ms） | 1114.091001 | 1008.015863 |
| 平均TPOT（ms） | 17.417030 | 17.686958 |
| P95 TPOT（ms） | 31.774061 | 31.976582 |
| 平均E2E（ms） | 2323.526111 | 2333.627120 |
| 远端获取次数 | 132 | 132 |
| 线上字节 | 306360162 | 306360162 |
| dispatch GPU / HOST / NVMe / Remote | 3327 / 2 / 538 / 133 | 3317 / 3 / 547 / 133 |

Full相对LastKnown：GPU时间高2.6807%，P95 TTFT高10.5232%，联合达成低
0.775个百分点；平均TPOT和E2E稍好。这是单轮取舍，不是Full获胜，亦不支持
LastKnown在所有指标或所有场景更好。两组观察值均过95%和共同资源预算，
不因此宣布正式G1/G2资格或统计显著性。

原始 `n_correct=null`、formal ranking=false保留。既有3B池是零更新结构
负载，原生长度/身份/引用链合格不等于独立非零LoRA算术已验证。资源分母
明确为offered，而非伪造correct-request。没有CI、独立留出或全局最优主张。

## 2. 新鲜度干预实际改变了什么

8,042个信息事件：4,000 arrival-prefetch、4,042 request-dispatch。
其中bootstrap674、hold10430、refresh8269、infeasible645是**候选行计数**，
不能当请求数或实验重复。全事件有14次纯视图选择分歧；与最终4000请求
逐一绑定的最终事件中，仅11次（0.275%）改变路由选择。

最终事件的9677个可行候选中，35行来源类发生分歧。保持年龄和底层来源年龄
必须分开报告：

| 时间量（秒） | 平均 | P95 | 最大 |
|---|---:|---:|---:|
| router额外保持年龄 | 0.023068 | 0.187519 | 0.497567 |
| 原生来源观测年龄 | 2.118194 | 5.300828 | 20.055392 |
| 实际来源所用观测年龄 | 1.816595 | 5.184717 | 20.055392 |

这说明0.5秒sample-and-hold不等于物理状态最多旧0.5秒。Full本身也复用
已确认历史观测，并在选中来源处重新保护/核验；它不是每次读取全局瞬时真值。
因此本项是**在既有来源验证不变时限制排名信息更新**，不是“完全无确认”
对“完美全局确认”，也不是移除层级缓存。

选中副本的排名GPU→稍后acquisition非GPU有13条，反向有12条；另保留42个
先前路由attempt。它们是不同时间点的观测，不是未选中副本的全局真值，也
不能全部归因为额外保持0.5秒。未执行候选的物理真值未知；没有shadow反事实
TTFT，未独立重放每次EWMA历史，不把这些限制补成零误差。

**论文判断：本自然W1观测未支撑confirmed-state propagation的端到端优势。**
干预对最终选择影响很小，且闭环扩缩容/批处理轨迹不同；不能把单轮差异直接
归因为陈旧性改善或损害。既定S2需要在请求和资源不变时区分“缓存是否存在、
缓存在哪个副本、router看到什么”，A3/S13继续保留。不为制造退化搜索更差
周期，不重跑此键挑胜负，也不据此断言状态协调在其他情形无用。

## 3. 其他机制与阶段证据

- LastKnown有2292个file plans、1136个GPU plans、708个native HOST准备、
  2281个residency epochs。684个类型化proactive admission决策均为admit；
  不据开启标记宣称此次软headroom策略带来收益。planner/handoff/admission
  未被此消融关闭，所有activation policy为full。
- 自然scale-up10、scale-down9，含initial共11个物理allocation，全部release。
  initial1/natural10均有first-dispatch和首token，unused/歧义均0；controlled0。
  样本定义不变，不为凑20追加同配置回放。
- Full平均TTFT高14.623617ms，其中arrival→admission差+12.966838ms，
  admission→acquired差+1.665090ms；其他阶段差均完整保存。阶段差是诊断，
  不是逐请求因果收益分解；两组自然轨迹不是固定相同的activation计划。
- 两组远端获取次数及字节相同，不能将差异解释为减少远端传输。Remote
  dispatch133与fetch132不必相等，dispatch状态和实际共享传输是不同事件。

## 4. 完整性、分析修正与资源

- 4,000个输入逐请求匹配，原生输出总数458,224，actual==target全通过。
  同后端输出hash3763条相同、237条不同，差异全部保留；没有无证据地归为
  纯浮点误差或错误adapter。独立数值正确性仍未知。
- 132次真实远端传输的ID/字节/内容hash双边一致；D80预发布gzip继续复用，
  无请求内打包。`existing_unmonitored`未连续测远端CPU/外部干扰，不声称
  全程远端无干扰。
- 2274次资源采样，无foreign GPU/escaped worker或保护中止；服务内存峰值
  22,170,759,168 bytes，high/max/OOM/kill/swap均0，实际GPU contexts释放。
  本轮HOST/NVMe工作缓存已空；147旧投稿文件及两项用户dirty SHA未变。
- 离线第一版14项测试通过，但真实记录揭示了分析语义错误：
  `routing_snapshot.selected_reference_acquired`在后续来源保护后会更新，
  不能强制它永远为false。真正不授予引用的是不可变ranking event。
  初次comparison失败及日志保留，不将此分析错误归为GPU实验失败。
- 修正后分别验证ranking无引用授权、实际来源保护标记与dispatch tier相符，
  新增本地/remote保护回归，共16项通过。collector不重复、仅重做离线比较，
  全4000排名/来源和最终事件journal关联通过；没有修改原始记录或serving配置。

## 5. 来源、备份及下一项

执行提交：`d0c608c1cc25960c81b19760530cf093ea3934f0`。
汇总：`paper_results/ieee_tc/ablation/d1062_3b_last_known_w1/`，包括
completion、comparison、analysis_qualification及各自SHA manifest。
原始回放和失败分析在
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1062_20261104/`。
运行后仅无损压缩已关闭的D1050/D1052–D1056六份main-outcome；各自两次
解压SHA均等于原文件，才移除未压缩副本。实际回收2,430,480,384 bytes，
完成时可用162,984,353,792 bytes（约151.79 GiB）。回执为
`paper_results/ieee_tc/storage/20261104_d1062_closed_ablations.json`；原路径旁
保留压缩/恢复回执，未来读取时使用gzip或按记录恢复，不改原研究身份。
Full D1047、当前D1062及所有请求JSON保持原文件，未覆盖旧投稿目录。

本检查点待按已建立方式提交至origin V2及FaaSLoRA专用backup分支；实际
推送成功与远端commit核验以原始目录`backup_completion.json`为准。
不把备份分支冒称已合并到存在分叉的实际FaaSLoRA V2。

下一主线是按 `D1063_NOHOST_IMPLEMENTATION_BOUNDARY_20261104.md` 实施NoHOST，
保留GPU backing和请求/计划暂存，关闭独立HOST复用并一致定价真实fallback。
之后综合10项已测反例/取舍和NoHOST证据提出一个公式保持的底层优化；不能
删除Full核心机制或改评价尺子冒称优化完成。Full尚未优于简化组，暂不画图。

未完成范围不缩减：7B十一配置和两个累积节点、受控A2–A5、M2工作点、
S1–S13、离线CE/价格/Pareto、B1/B2/C1/C2/C3与两轮审稿文档、IEEE图表、
最终同版本来源统一与实际FaaSLoRA V2集成。共享k3s不操作，HydraServe独立
适配和3B D807–D809直接复用，不重启已关闭的基础适配。
