# D679：在线控制测量范围与缩容必要条件审计

2026-10-18。只读复用D673/D678已封存投影，新增GPU运行0；不改配置、生产
代码、共同SLO、九公式或历史结果。先核验事实，再决定下一优化候选，不扫描
阈值、不通过删除冷启动样本使控制器看起来更好。

## 已核实的来源

- IEEE PDF SHA `22d4272e78255f393d4d4f071670e7ca43f43771805784af75d9123a4e3912d1`，
  §III-B规定queue、active saturation、observed P95 TTFT归一化最大值扩容，
  三者持续低于各自下阈、无pending、保留minimum时才缩容。不能将TTFT
  直接替换为GPU利用率、平均值或去掉这一项而声称公式/思想未改。
- D48提交`faf5c855b0cbb2e13d841ff711ac7aaead43a5ae`接入当前控制律；
  当前runner将成功完成请求的`overall_ttft_ms`按实际收到时刻加入窗口。
  pending不能冒充ready；未知TTFT不能当作零。D549初始真实到达控制保留。
- D88 §Step 4的219.518056ms来自旧source-only GPU-ready acquisition→first
  P95。文档本来就标为开发启发式，而不是端到端SLO或最优值。D645只更新
  服务/准备profile，明确未重新标定这个控制参数；配置一直继承该值。
- D648已冻结共同评价阈值，原生dispatch→token warm定义也不等于完整
  controller TTFT。因此不能不加解释地用warm P95替换219.518并宣称口径一致。

## 本次要证伪的问题

D678已证明两版TTFT低条件始终不成立。本次检查：保持实际queue、active、
pending、minimum及48秒cooldown，是否存在其他缩容必要条件连续成立的区间？
这不是移除生产TTFT gate，只是确定“即使修正TTFT配置，是否还被其他条件
完全阻断”。只统计连续控制采样，不声称采样间没有瞬态；不计算反事实资源
或延迟。另复用原生首末token，对短输出违约量化实际decode超过共同许可时间
的大小，以区分资源回退与未完成的服务质量问题。

## 同类原始资料：可借鉴与不可照搬

使用research-lookup作有界核查；Parallel CLI 0.7.1未认证，使用可用网页工具。

- [Knative官方说明](https://knative.dev/docs/serving/autoscaling/kpa-specific/)
  明确对concurrency/RPS采用不同时间窗；panic期间不缩容。与PrimeLoRA的
  observed TTFT P95不是同一量，不能直接搬其默认秒数或目标值。
- [Knative官方实现](https://raw.githubusercontent.com/knative/serving/main/pkg/autoscaler/scaling/autoscaler.go)
  直接用observed stable/panic值除以同量纲TargetValue；缺测返回无有效决策，
  down-delay保留近期推荐的上包络。借鉴点是信号、目标、时窗的一致性；
  本次没有移植其比例扩容/双窗口，也不据此证明PrimeLoRA的参数最佳。
- [Llumnix OSDI 2024](https://www.usenix.org/system/files/osdi24-sun-biao.pdf)，
  §4.4及§6.5在freeness同一量纲内设置上下界，并用资源—尾延迟曲线评价。
  它有KV迁移和不同信号，不能给本项目毫秒阈值背书。

## 离线结果及可证伪边界

| 观测 | D672 | D677 |
|---|---:|---:|
| 到达窗口内控制采样 | 1888 | 1889 |
| 除TTFT外必要条件连续满足≥48s的区间 | 12 | 12 |
| 最长上述区间(s) | 123.261213 | 172.608358 |
| 上述区间中跨过48s后的控制采样数 | 123 | 195 |
| 实际三项全低并跨过cooldown | 0 | 0 |
| 短输出（目标≤31）请求总数 | 1106 | 1106 |
| 其中TPOT违约 | 122 | 141 |
| 短输出违约的decode超额时长P50/P95(ms) | 90.747/328.658 | 72.590/428.278 |
| 短输出违约且dispatch为GPU | 61 | 70 |

decode超额为原生`last−first−(n−1)×共同TPOT阈值`，不是按每条请求重新定义
TPOT，也不是已测到的可移除CPU成本。D677短输出TPOT违约141条中18条同时
TTFT违约；GPU子集超额P50为73.365ms。完整4000条均保留，未改短请求分母。

结论：不存在“其他所有缩容必要条件永远无法持续48秒”的阻断；实际日志
中TTFT gate确实另行阻止了这些区间的缩容。但这不是移除gate后能节省多少
GPU-s的预测，因为缩容会改变路由、并发和后续TTFT。更不能据12段区间直接
承诺G1/G2达标，或从这些区间反推一个有利下阈。

**测量范围不一致是配置依据需要重新审视，不是自动证明219.518数值错误。**
共同Resident-vLLM D653在同一3B W0中P95 TTFT170.577488ms，联合99.975%，
表明同一硬件/输入下存在较低完整TTFT观测；它不带PrimeLoRA全套控制，不能
当作两者差值全部可删除的因果证明。D648 warm仅覆盖原生路径，直接用它
补一个更宽松控制阈值仍不能解释Full稳态管理开销。该参考已由D657封存，
本次只引用，不重新测量或用于改变评价尺度。

下一主线是有界核对保留的稳态执行证据：在尚未重新标定控制参数前，先
定位Full的admission→engine等待及短decode干扰由哪些已观测控制工作造成，
对照D512–D635/D670–D675历史避免重复已做的优化；选一个同时可能帮助
服务质量和物理资源的可证伪候选。不是继续多做一次HOST清点缓存微优化，
也不是再次部署HydraServe或追加baseline性能搜索。

## 执行与封存

复用D612的summary/读写helper与现有Type-1分位数、预检查/资源关闭入口。
新增仅为连续必要条件和decode超额分析；无需重读大outcome或重建负载。
check-only边界检查及唯一离线执行通过，资源域递归空关闭；峰值104321024B，
high/max/OOM/kill/swap均0。GPU/远端未启动，生产源码与配置未改。
diagnosis SHA `cb445b838ceb0f5975376d3b315ea92489c79bdfa06b60fb05ea201b476a48fd`。
原始目录`results/ieee_tc/p2_backend_qualification/d679_20261018/`。

D678已推送`5f18f6727accd93ec201c8f82066b10237135e95`且远端一致，publication
域递归空关闭。D679本次结果备份待执行，不将准备状态写成已推送。
3B G1/G2及其他既定场景/基线/dLoRA、同版本7B、A1–A5/S1–S13、成本/CE、
审稿文档与最终图表全部仍ACTIVE。
