# 从论文贡献推导评价，而不是从希望的排名选择指标

2026-10-14，补充用户最新澄清。与`SLO_BUDGET_RATIONALE_20261014.md`合用。
本轮重新核对用户提供的IEEE版本LaTeX（附件0b113e57）Introduction及Design。
不修改论文。历史CE结论、SGLang实现描述不冒充新实验已通过结论。

## 同类系统为什么选这些指标

| 系统与核心问题 | 核心机制 | 因此选择的指标与动机 | 对PrimeLoRA的可迁移部分 |
|---|---|---|---|
| ServerlessLLM：模型启动、迁移阻塞 | 分层快速checkpoint加载、推理迁移、启动感知调度 | 启动/暂停时延直接衡量其缩短的路径；不能只靠稳态token吞吐证明快启动 | 单列adapter准备与扩容首次服务，不把所有排队混叫模型启动 |
| HydraServe：serverless冷启动，同时控制并行启动的资源代价 | 取模型与初始化重叠、分布式获取、流水线合并回单端点 | TTFT/SLO对应冷启动可见延迟，TPOT揭示流水线代价，memory-time揭示资源回收效果 | 同时报首次服务、尾部与生命周期资源，接受真实取舍；不照搬其显存计费为整卡成本 |
| DistServe：prefill/decode相互干扰，满足双时延要求的容量不足 | 分离两阶段，联合分配资源与并行度 | 在TTFT/TPOT约束下最大化per-GPU goodput，避免吞吐提升以违约为代价 | 联合质量约束合理；本项目固定offered trace不能冒充已测最大goodput |
| Sarathi-Serve：大batch提高吞吐却暂停decode | 分块prefill与不停顿调度 | P99 TBT约束下容量及median TTFT，直接检查decode是否被阻塞 | admission需要争用证据；请求平均TPOT不能证明每个token都平滑，需S3事件/逐token证据 |
| Llumnix：跨副本不均衡、碎片及优先级隔离 | 运行中迁移与动态调度 | 尾延迟、优先级效果、相近尾部下平均实例代价，检查资源和服务的关系 | 同质量省资源/同预算减尾部，比单个自定义乘积更清楚 |
| dLoRA：adapter合并与批处理取舍、请求/adapter位置不均衡 | 动态merge/unmerge与共同迁移 | 服务时延/吞吐和总E2E除以总输出token，评价共同编排收益 | 不能假设“联合编排”本身新颖；需要隔离confirmed state与准备时机的增量 |

依据：[ServerlessLLM §7.1](https://www.usenix.org/system/files/osdi24-fu.pdf)、
[HydraServe §8](https://www.usenix.org/system/files/nsdi26-lou.pdf)、
[DistServe §2/6](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)、
[Sarathi-Serve §5](https://www.usenix.org/system/files/osdi24-agrawal.pdf)、
[Llumnix §6.5](https://www.usenix.org/system/files/osdi24-sun-biao.pdf)、
[dLoRA §7.1](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)。
不是把这些论文的新颖性或实验结论直接当成本项目证据。

## PrimeLoRA贡献与结果证据的对应

IEEE版本目标是让新激活副本更早对当前adapter需求可用，同时保留弹性回收，
并限制准备对已有推理的干扰。这包含两种成本：准备不足造成等待，过度准备
造成资源占用/争用。不能只最小化其中一个再宣称全部目标实现。

| 论文机制 | 应先证明的直接效果 | 关联的系统结果 | 可能反向代价 |
|---|---|---|---|
| confirmed per-replica state与observed-cost routing | false-ready/missed-ready、状态年龄、分歧及选中副本的dispatch前层级 | churn下TTFT/P95及联合SLO | 查询/同步阻塞decode，热点副本排队 |
| activation期间demand-weighted handoff | trigger→ready与first-dispatch→first-token分开；已准备且被实际使用比例 | burst/scale-out首次服务与W1尾部 | 未使用准备、提前持卡、传输争用 |
| GPU/HOST/NVMe层级驻留 | 真实remote获取次数/字节、重用、有效来源与加载时间 | 达标工作负载的GPU-s和TTFT | HOST/CPU占用、迁移/驱逐控制开销 |
| 依据KV/headroom/load的执行期admission | 候选/接纳/延后、真正加载干扰与后续延期代价 | TPOT/SLO保护，同时避免TTFT转移退化 | 将准备推迟到请求路径，队列与管理成本 |

这支持G1/G2，但不从机制推出95%或75%。论文收益函数估计需求加权准备
时间节省，是局部规划目标；它不等于全局GPU-s或P95的解析最优解。
因此必须测最终结果，不能把满足背包预算或有tier命中当作性能最优。

## 论文评价主次的客观建议

1. 主贡献叙述为“服务质量约束下的弹性资源效率，以及扩容/churn下的尾部服务”，
   不是所有原始时延都最快，也不是CE最大就算成功。
2. 主表一起列联合SLO、物理GPU-s/request与TTFT/TPOT尾部；G1和G2回答不同
   问题。保留冻结95%/75%点，但并列展示事先规定的多档结果和Pareto边界。
3. 机制实验用A/A、LastKnown、固定residency/admission、ElasticOnly、handoff
   延迟及S3分解建立因果链。ElasticLocality仅是复合消融，不是已发表baseline。
4. 各系统在相同输入/真实远程/完整生命周期下比较；新vLLM基线排除后端升级
   独享收益。baseline核心机制未触发必须注明，不能把适配范围写成原论文极限。
5. 预期优势来自减少无效等待/准备和更及时释放，不来自某个任意数值。管理成本
   可以抵消收益，D512已有这一方向的证据。持续优化必须解决代价而非隐藏它。

当前97.425%只证明通过既定95%的时延门槛，不证明满足99%。如果后续结果
只在某一质量档位领先，论文结论应限定档位；这比事后寻找能让PrimeLoRA
第一的门槛更能回应两轮审稿对CE与成本假设的质疑。独立数值正确性与完整
W1/W2、3B证据缺口仍单列，不由指标文献核查消除。

## 实验章节可采用的说明草稿（不是已完成实验的声明）

> PrimeLoRA旨在减少弹性扩容中尚未完成的adapter准备，同时限制准备与已有
> 推理的争用。仅比较原始首token延迟会忽略常驻资源的代价，而仅比较资源
> 消耗又可能奖励违约服务。因此，我们借鉴DistServe在TTFT/TPOT服务约束
> 下评价有效服务能力的方法，以及Llumnix同时考察服务尾部和资源代价的
> 方法，分别评价达标服务的生命周期GPU时间与共同预算内的尾延迟。
> 对缺乏应用SLA的回放，我们参考HydraServe的warm请求标定方法设置时延
> 尺度；长度分组与联合判定由本研究明确规定，不声称与其设置完全相同。

> 我们保留预先冻结的95%联合达成要求作为主参考点，即所有请求正确完成
> 后，最多5%可违反时延要求；并检查90%、95%、99%三个服务档位。
> 共同资源预算取完整常驻参考GPU时间的50%、75%、100%，其中75%用于
> 检验至少25%资源节省条件下的服务表现。这些是研究档位而非行业统一SLA。
> 我们同时呈现原始达成率、时延分布和质量—资源关系，避免结论依赖单一
> 阈值或成本折扣。每个冻结配置保留一次有效运行；敏感性复用同一配置的
> 观测时，不将其表述为该系统在每个服务档位重新优化后的能力上界。

写入最终论文前须核对：独立正确性证据是否补齐、W1/W2预算参考是否完成、
所有敏感性点是否确实已测/可由日志恢复。未完成部分改为限制，不使用完成时态。
引用建议为DistServe §2/6、Llumnix §6.5、HydraServe §8.3；涉及逐token
平滑性与争用实验时再引用Sarathi-Serve §5，不能用它替代本项目TPOT定义。
这些引用支持评价动机和方法，不用来替95%/75%伪造直接出处。
