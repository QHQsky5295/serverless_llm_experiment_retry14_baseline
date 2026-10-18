# GPU 资源指标的文献依据与本项目边界

2026-10-18。回应用户关于“GPU占用是否有同类论文常用评估方式”的问题。
本文件是解释性文献核查，不修改冻结 V1、单轮 V2、SLO、预算、执行配置或历史结果。
核查为有界原文阅读，不声称全领域普查；不新增 baseline 或性能实验。

## 结论

累计 GPU 时间不是 PrimeLoRA 独创的复合评分，具有直接的 serverless LLM
论文先例；但不存在所有论文统一采用同一资源单位、计时边界和 SLO 门槛的规范。
共同范式是同时考察服务质量和资源代价，而非只看 GPU utilization 或只看延迟。

| 原始来源 | 实际评估方式 | 与本项目的关系和限制 |
|---|---|---|
| FaaScale，MLSys 2026，§7.4、Fig.10/11 | GPU 分配时间线、累计 GPU 时间、TTFT 分布 | 最直接的 GPU-time 先例；不能据此声称其准备/清理计时边界与本项目相同 |
| Llumnix，OSDI 2024，§6.5、Fig.14/15 | 平均实例数作为资源成本，并展示 P99 prefill latency–实例数关系 | 支持质量–资源取舍；只有每实例 GPU 数固定、时间窗口一致且资源归属一致时，才可换算为 GPU 时间 |
| DistServe，OSDI 2024，§1/§6.2 | 满足 TTFT/TPOT 达成要求的最大每 GPU 请求率，即 per-GPU goodput；主要90%，补充99% | 支持共同 SLO 下的资源效率；其“最大可承受速率”不是本项目固定回放的 N_good/U |
| HydraServe，NSDI 2026，§8、Fig.13 | TTFT/TPOT 与达成率；成本与 GPU memory-time product 成正比 | 显存×时间适合分析其共享/流水线资源模型；按实际已用显存计时不等于整卡占用时间 |
| ServerlessLLM，OSDI 2024，§7.4、Fig.12 | 改变每节点 GPU 数并测时延，以及固定 GPU 下增加模型数 | 直接使用资源–性能曲线，并非一律使用 GPU-s/request |

原文链接：

- [FaaScale 会议原文](https://proceedings.mlsys.org/paper_files/paper/2026/file/6e32c247076c2c0fb381e022c02d2c78-Paper-Conference.pdf)
- [Llumnix](https://www.usenix.org/system/files/osdi24-sun-biao.pdf)
- [DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)
- [HydraServe](https://www.usenix.org/system/files/nsdi26-lou.pdf)
- [ServerlessLLM](https://www.usenix.org/system/files/osdi24-fu.pdf)

## 哪些是数学关系，哪些不是文献共同规定

本项目的物理持卡时间为 U=∫G(t)dt=Σ_d∫a_d(t)dt，单位 GPU-s；
R=U/N_correct，单位 GPU-s/correct request。G(t)为服务实际持有的物理卡数，
不是逻辑副本数、利用率或显存比例。同卡多进程按区间并集计量。

在同一计时区间长度 T 下，平均持卡数为 U/T；静态 g 卡部署为 U=gT。
这是数学恒等式，不是声称任何论文采用了我们的全部计量合同。
不同 drain/准备/释放边界不能忽略；实例跨卡数变化时也不能仅数实例。

GPU-s 和 GPU-hour 仅差3600倍。正确请求工作量固定时，U 与 R 的排序一致。
GPU utilization 用于瓶颈诊断：高利用率可能来自有效批处理，也可能伴随排队，
不能单独证明更低资源成本或更好服务。GPU-s 不是能耗，能耗需积分功率。

若所有卡型号相同且按整卡保留、单价统一，GPU-only cost=pU/3600；
不需要 idle-ready 折扣即可比较物理资源。CPU/HOST/远端另报，不能将其称为
完整云账单。若未来采用可实际复用的分卡/共享、异构 GPU 或按显存计费，应
另列相应资源模型，不能沿用整卡公式后声称已经覆盖这些条件。

显存时间 J=∫M(t)dt：仅在 M(t)=M_card G(t) 的固定整卡保留容量定义下
有 J=M_card U；按实际 used memory 计量一般不等价。少用一半显存但未释放
该卡，不自动节省一半整卡占用，也不能抹去共享系统可兑现的容量收益。

## 对 PrimeLoRA 评估的判断

保留 G1/G2：全部正确完成、共同服务要求下减少生命周期 GPU 时间；共同
预算和服务要求下比较扩容/churn 尾延迟与观测联合 SLO。报告原始 TTFT、TPOT、
达成率、总 U、R、平均持卡数及资源–质量关系，避免单一合成分数掩盖取舍。
N_good/U 是可补充的达标请求/GPU-s，但不称为 DistServe 原文定义完全相同的
最大 per-GPU goodput；未扫描容量上限，不声称测得最大吞吐。

95%联合达成、75%参考预算、60秒准备和真实释放边界属于本项目已冻结协议，
不是上述论文共同规定。维持90/95/99%及0.5/0.75/1预算敏感性。未达95%的
baseline仍展示；它只是未满足该档服务约束，不等于不允许纳入比较。
单轮仍仅作观测性结论，不外推全局最优或统计显著。

论文推荐称呼：累计物理 GPU 占用时间（cumulative allocated GPU time）及
每正确请求 GPU 时间；不用含糊的“GPU占用率”代指。论文需说明：从共同部署
通知后的实际持卡起点至真正释放，覆盖启动、驻留、业务、排空和清理；不能
把全部边界归于 FaaScale，也不能把 CPU-only 准备虚计为持卡。

## 与旧 CE 讨论的关系

[ServerlessLoRA 2025公开稿](https://arxiv.org/html/2505.14468v1) 的§2.2脚注3、
§6.1明确给出1/(E2E_latency×Monetary_Cost)，§6.4按云计价比较。
因此不能说类似 CE 完全没有先例；这里核查的是公开稿，不据此增补发表资格。
该先例也不意味着此复合指标能替代服务质量约束、价格敏感性或本项目的物理
资源证据。保留CE为补充，不恢复为唯一主指标。

检索方式：research-lookup 的来源核查原则；Parallel CLI 0.7.1未认证，
改用网页工具读取会议/作者原文，未安装依赖、提供新凭据或上传项目私有数据。
运行中的D677保持不变，本次没有重跑、画图、调参或修改计量口径。
