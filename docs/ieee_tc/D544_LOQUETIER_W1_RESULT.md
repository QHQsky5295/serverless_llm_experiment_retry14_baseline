# D544：Loquetier 7B W1 完整终态与失败诊断

2026-10-15。预登记见 D544_LOQUETIER_W1.md。本轮只运行一次；不重跑、
不加大超时、不修改原生策略。以下是固定已知负载的开发观测，不是独立留出评估。
实际执行 MAIN `fa65e1619be810bb862dac91d1178cd25e01ac40`，BASE
`bde215460acb9e850efa479a89bda6fc39a2bd7d`，均已在启动前推送。

## 结果及分母

| 指标 | 观测值 |
|---|---:|
| 计划／到达／提交／终态 | 4000／4000／4000／4000 |
| 原生绑定、数量、计时核验通过的响应 | 3844（96.10%） |
| 失败 | 156（3.90%），全部 TimeoutError |
| 成功输出 token | 428276（目标总量458224） |
| TTFT／联合达成 | 0／4000，0% |
| TPOT 条件达成 | 2863／3844，74.4797% |
| 成功请求平均／P95 TTFT | 1098265.744／1725784.244 ms |
| 成功请求平均／P95 TPOT | 62.254／67.623 ms |
| 成功请求平均 E2E | 1105125.931 ms |
| 完整生命周期物理 GPU-s | 14803.43732188805 |
| GPU-s／offered 请求 | 3.7008593304720123 |
| GPU-s／绑定有效响应 | 3.8510502918543312 |

失败全部留在4000的联合SLO分母中。时延分位数只有3844条成功请求，存在失败
截断，不能与全部成功系统直接推断尾延迟优势。独立数值正确性仍未证明，
`n_correct=null`；绑定、内容和原生执行核验不代替独立数值证明。
本轮 `measurement_complete=true`，但 `workload_passed=false`，
`performance_ranking_eligible=false`。不进入G1/G2可行排名，不把更短到达窗口
或超时造成的U变化称为相对W0的效率改善。

互斥GPU时间：pre-arrival 0、arrival 7609.67812881805、drain
7153.190300580114、cleanup 40.56889248988591 GPU-s，合计严格等于U。
所有4个真实lease释放，进程退出和原生GPU空闲证据均已核验。

## 为什么这么慢：可支持与不可支持的解释

同原W0的500逻辑ID、132实际请求ID、同prompt/target/源序列，仅使用预定W1
到达变换。完整4000输入与D536 vLLM逐请求对应；sealed D533/D537/D542
证据联结PrimeLoRA、ServerlessLLM、S-LoRA。相同输入不意味着各自原生版本、
初态、slot数、编译缓存或部署适配相同。

成功请求的阶段分解：

- router队列平均39739.940ms；原生队列入队至注册平均1058274.986ms，
  后者占平均TTFT的96.3587%。不能将它称为纯GPU计算或远端传输。
- 注册至首token平均210.962ms，decode平均6853.553ms。
- 所有8个500请求phase的联合达成均为0，未删掉启动阶段或挑后期片段。

| 分片 | 逻辑adapter | offered | 成功 | 失败 | ready相对业务开始（秒） |
|---|---:|---:|---:|---:|---:|
| 0 | 125 | 1008 | 1004 | 4 | 275.508 |
| 1 | 125 | 1076 | 999 | 77 | 367.334 |
| 2 | 125 | 1048 | 973 | 75 | 458.725 |
| 3 | 125 | 868 | 868 | 0 | 550.097 |

请求超时平均1800.001975秒、P95为1800.003274秒，与冻结的1800秒保护一致。
原生记录：4000入队、3845完成、155取消；其中一条原生完成发生于客户端超时
后0.119418秒，仍计失败，不能补为成功。155条取消均已注册并记录11279个
部分输出token；这些不计有效吞吐，所消耗GPU时间保留。没有native/registration/
cleanup错误事件，没有失去终态记录的失败请求。

当前证据支持：冻结的SMLM/FP16/batch2/context1024、四个125-ID静态分片和
顺序启动部署，在W1压缩到达下持续积压。静态字节均衡不是逐时负载均衡。
这是该适配工作点的局限，不是原论文所有配置或统一训练/推理机制的性能上界。
未用单次观察证明分片、启动或batch上限各自的因果贡献。

源码核查：D544启动前24个Loquetier相关函数/类与D445执行版本AST相同，39项
既有CPU路径测试通过。官方[run_inference工作流](https://raw.githubusercontent.com/NJUDeepEngine/Loquetier/aae33baeeb19777129c1ccbff99a898d4a0e2c63/src/loquetier_src/model_workflow.py)
按batch容量登记请求并调用原生generate；当前online FIFO保留这一核心调用关系。
静态分片与顺序启动仍是披露的本机适配，不冒称官方自适应集群调度。
对照D445，注册至首token约210ms、TPOT约62ms相近，主要恶化发生于注册前队列。
这不是隔离其他因素的因果实验；未发现新的适配缺陷证据，不据此扩大batch、
改分片或对baseline搜参数。保留异常结果及边界，继续既定主线。

## 环境与完整性

完整500对象在通知后准备93.442315秒，500次远端fetch、420740410线上字节、
12985984450物化字节；请求内fetch及打包均为0。双端transfer、内容SHA、
worker receipt、实际forward与可靠计时复核通过，E2E/TPOT/native重算误差0。
复用既有D78/D80只读gzip发布，不改权重、负载或生成契约。
500逻辑ID不等于500独立训练权重：498零／2非零ID的限制继续披露。

服务内存峰值43098591232B，high/max/OOM/kill/swap均0；没有安全中止。
物理释放与本机资源域移除后才停止3个身份匹配的远端服务，全部inactive、
MainPID0、端口消失。正式推理期间未执行测试、构建、清理、压缩或远端修改。
历史轮询曾只查看http_error而遗漏http_request_failed；已在运行结束前纠正并
告知用户，本报告基于全部事件枚举。该监控错误不修改原始记录、到达或超时。

## 离线分析修正（不是服务性能优化）

第一次严格分析被拒绝：原有complete_with_failures只识别ServerlessLLM错误
格式。原生resident runner对失败请求既报告workload错误，也报告
`physical or native workload measurement incomplete`；后一句并不意味着
本轮物理测量缺失。不能只改错误字符串或忽略异常放行。

现有analyzer增加按scheduler/schema区分的显式失败诊断：要求完整4000终态、
响应/失败计数一致、独立物理summary完整、无open lease、无interrupted、
native完成数与成功响应一致、无额外释放错误。后续来源、原生绑定、双端交付、
实际释放与GPU积分检查全部保留。成功模式仍拒绝任何失败；错误模式仅产出
不可排名的诊断。58项CPU测试通过，覆盖不同原生schema、计数不符、未知错误、
缺失释放/测量和不可将失败改名成功。首次失败日志原样保留，只重算受影响统计。

原始路径：`results/ieee_tc/loquetier_qualification/d544_20261015`（原results根）。
curated：`paper_results/ieee_tc/main_comparison/d544_loquetier_w1/result1/`。
保存audit、输入对应、阶段/失败诊断、代码/来源SHA和小型证据包；原始大日志不入Git。

## 主线状态

7B W1已具备PrimeLoRA、vLLM、ServerlessLLM、S-LoRA、Loquetier单轮记录；
五者均未达到共同95%联合SLO，Loquetier另有156条失败。PrimeLoRA为93.725%，
不能因对手更差宣布G1/G2完成。下一步安全回收已验证冗余缓存后，复用HydraServe
已完成的隔离适配执行W1，不重做其W0/基础资格，不修改其他项目k3s。
7B W2/G2、PrimeLoRA改进、3B全部比较、dLoRA B6、A1–A5、S1–S13、
成本/CE与审稿文档、最终图表继续保留。尚未接近完整目标，当前不画图。
