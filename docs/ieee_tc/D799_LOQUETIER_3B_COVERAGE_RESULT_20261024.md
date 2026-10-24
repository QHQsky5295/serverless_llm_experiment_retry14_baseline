# D799：Loquetier 3B完整500-ID功能覆盖结果

## 本轮回答的问题与结果

这是已登记的`adapter_coverage_v1`功能视图：复用原trace前500条输入、到达与
目标长度，将其映射到500个既有逻辑adapter。不是新生成的主性能负载，也不是
W0。四个静态分片各125个adapter，保留原生SMLM、batch4、PEFT owners和顺序
初始化；使用D797模型专用内核、D798有限数值检查及D799已备份的绑定实现。

| 核查项 | 本轮观测 |
|---|---:|
| planned / arrived / submitted / terminal / response | 各500 |
| 请求失败 | 0 |
| 实际服务的逻辑adapter | 500 |
| 逐请求原生生成、绑定与计时重验 | 500/500 |
| 原生输出token | 68,925 |
| 实际就绪静态worker | 4 |
| E2E、TPOT与原生计时最大重算误差 | 0 ms |
| 生命周期物理GPU时间 | 2,246.453670 GPU-s |
| 远端获取次数 / 线上字节 | 500 / 1,160,474,851 B |
| 核验后的本地物化字节 | 19,482,573,296 B |
| 请求路径动态打包 | 0 |
| 服务内存峰值 | 35.924 GiB |
| 主机最小可用内存 | 88.956 GiB |
| high / max / OOM / OOM kill | 全部0 |

500份工件在部署通知之后从同一D80已发布只读gzip服务获取；没有本地remote
fallback。实际链路启动前为1000Mb/s/full，未注入sleep。服务/监控返回0，
四张GPU真实释放、服务资源域移除；本次精确身份的远端服务和监控已停止。

## 保留的时延诊断，不作为W0排名

| 指标 | 观测值 |
|---|---:|
| 平均 / P95 TTFT | 206.407 / 475.252 s |
| 平均E2E | 213.729 s |
| 平均提交滞后 | 6.699 ms |
| 平均dispatch前等待 | 132.497 s |
| 平均service TTFT（含原生调度等待） | 73.903 s |
| 平均decode时间 | 7.317 s |
| 平均 / P95 TPOT | 53.641 / 61.104 ms |
| TTFT / TPOT条件 / 联合SLO达成 | 16.4% / 0% / 0% |

SLO使用冻结3B D648共同阈值：两输入长度组的TPOT分别29.922/36.216 ms。
较长TTFT同时包含共同通知后的完整池准备、顺序初始化和服务队列等待；
service TTFT不能全部称为prefill计算。所有阶段保留真实时间，不将其从结果
中扣除，也不把本次功能视图的长等待外推为原论文所有部署的机制缺陷。

低SLO不是基线排除条件或继续配置搜索的理由。本轮排除了已检查路径上的
模型几何/RoPE误用、无法部署完整池、错误adapter绑定及少生成token等问题；
下一步沿同一冻结配置完成W0/W1/W2，观察实际主负载，不增加100/1000门禁。

## 正确性边界与来源

`workload_passed=true`指生成/绑定/计量合同通过，**不等于全池独立数值证明**。
原始`independent_lora_numerical_correctness=false`、`n_correct=null`等未知
字段保留。D798仅对两个真实prompt、两个同payload逻辑adapter作六项有界
logits/token对照；本轮500-ID成功不能把它扩大为500份独立训练权重的数值验证。
这也不是正式性能资格、完整复现资格或因果性能改善证据。

本轮使用`analyze-results`区分观测、解释与结论边界；单轮、不补CI、不画图，
不与4000请求PrimeLoRA结果计算相对收益。原始和curated数据分别位于：

- `results/ieee_tc/loquetier_qualification/d799_20261024/`
- `paper_results/ieee_tc/baseline_audit/d799_loquetier_3b_coverage/`

既有analyzer要求`http_3b_500.json`，实际执行配置名为
`http_3b_coverage500.json`。仅在真实释放后创建逐字节相同的分析别名，
保存共同SHA与原因；没有改动执行配置或重跑GPU。沿用原analyzer，无统计修补。

清理只针对本轮已证明重复、无活跃引用的下载对象，保留原池、传输/绑定回执、
原始请求和监控。代码此前已分别备份到BASE `8594922`和MAIN `482ff4d`；
结果检查点另存不可变manifest，不覆盖此前的静态绑定检查点。

下一主线：Loquetier 3B W0/W1/W2 → HydraServe 3B（复用已验证的隔离路径，
不改他项目k3s）→ 全基线对比与PrimeLoRA优化 → 同版Full/A1–A5/S1–S13、
成本/CE/Pareto、两轮审稿文档、最终图表与V2交付。所有未完成问题继续保留。
