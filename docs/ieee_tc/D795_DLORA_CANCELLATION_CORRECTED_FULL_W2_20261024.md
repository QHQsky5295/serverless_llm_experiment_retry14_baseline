# D795：取消生命周期修正后的 dLoRA 3B 完整 W2

2026-10-24。4000条请求均到达、提交并进入终态；3122条绑定/原生计量有效
响应，878条TimeoutError。完整失败观测保留，不是G1/G2可行点。本轮未再
出现旧迁移断言异常，但不由此推断全部超时原因已知或完整数值正确性已证明。

## 配置、身份与复用

与D794 W1使用同一D793修正版本：DP2/TP1/PP1、原生EngineManager、
PERIOD_MIG、500-ID部署、max-seqs4、max-batched-tokens1024、GPU利用
配置0.92、GPU LoRA容量8、swap8。native配置仅独有ray_root路径不同；
HTTP输入逐字节复用D792 W2。没有新参数搜索、新权重、整池复制或短GPU
资格。共同1800秒保护、真实远端和资源包络保持不变。

- MAIN执行HEAD：`28381d8e2f4f176069c27f6df103008f74b40f92`。
- BASE执行HEAD：`32c8d0fc4735165306ca0b9f5f6d5a6ce23cc586`。
- HTTP SHA：`7a7264bd9c88244c5eb0a9ee63fc1ee303bb59c943b1c2b5b6b903a4227286d6`。
- W2 view SHA：`5558de43ede0e2cf49c7478dcfb94640e7a8516406f53766fabfdf524bf91174`。
- 模型源码视图SHA：`fd0916a07bc7365d35b78ac6b54b0d5481b849f25056927019f78471f82c6ca1`。
- 共同3B SLO SHA：`d15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8`。

W2维持原arrival/prompt/token序列，只采用既定rotation100映射；实际offered
497个adapter ID，部署准备500，目标458224输出tokens。源seed42、开发身份41；
不是新的独立留出trace。复用D793的55项CPU回归及D790两组既有prompt/adapter
在prefill与两步decode的6/6 greedy一致证据，后者不是4000请求独立数值证明。
原始`n_correct=null`、formal_run=false等身份不追改。

## 完整测量

| 观测量 | 本轮数值 |
|---|---:|
| 计划/到达/提交/终态请求 | 均4000 |
| 绑定/原生计量有效响应 | 3122（78.05%） |
| 失败 | 878（21.95%），全部TimeoutError |
| 有效响应原生输出tokens | 351924，逐请求与目标一致 |
| 实际服务adapter ID | 380；offered497；部署500 |
| TTFT达标 | 41/4000＝1.025% |
| 联合TTFT/TPOT达标 | 26/4000＝0.65% |
| 条件TPOT达标 | 2283/3122＝73.126201% |
| 物理生命周期GPU占用 | 9086.517108698376 GPU-s |
| 条件平均/P95 TTFT | 386545.496681 / 1566152.884935 ms |
| 条件平均/P95 TPOT | 306.975874 / 206.221501 ms |
| 条件平均/P95 E2E | 424927.482929 / 1648597.987556 ms |
| 原生计时/E2E/TPOT最大重算误差 | 均0 ms |

时延样本为3122条有效响应，失败仍在4000的offered分母中。GPU-s/offered
为2.271629277174594，GPU-s/绑定有效响应为2.910479535137212；均不称为
已证明数值正确请求的资源效率。无完整正确完成、共同联合SLO亦不满足，
不能以资源总量较小宣布胜过正确服务点。不作CI、显著性或全局最优结论。

GPU互斥窗（GPU-s）：pre-arrival0、arrival7740.434715338051、
drain1322.1366175338626、cleanup23.945775826461613，和等于U。
service/replay退出码均1、watchdog0；实际GPU释放和服务资源域消失已核验。

## 阶段与机制解释

有效响应平均E2E分项（ms）：提交3.766700、提交后dispatch等待20247.714951、
service TTFT366294.015031、decode38366.595846、完成通知15.390402。
主要等待发生在进入原生服务之后；service TTFT包含原生排队、调度、加载和
prefill，不能称为纯计算，也不能将整体延迟归结为网络或客户端等待。
序号3000–3499的500条均响应，router queue均值6.861027ms、service TTFT
均值479974.830011ms。这是事后描述切片，不是无启动独立主结果。

按完整worker日志行首匹配：64次ILP求解、累计42.840753316879s、最大
3.660435914993s；69条move决策打印，312次请求提及。后两者不是312个
独立成功迁移回执；不从用户延迟减去累计求解时间。Finished3122、Aborted878
的条数与终态相符，但不把计数对应当作每次取消和迁移的完整所有权证明。
未匹配Traceback、AssertionError、RuntimeError或Task exception；末尾
12:43:46的SIGTERM对应终态后的受控关闭，不改写为推理崩溃或全部成功。

修复前D792保留为取消缺陷影响的诊断；不择优使用其结果，也不把不同响应
子集的前后差值写成单因素性能因果结论。D794/D795已完成同D793配置的
限定W1/W2；没有新的具体异常证据要求继续基线性能搜索。低SLO本身不触发
重跑或放宽阈值，下一按主线处理3B Loquetier必要正确性适配。

## 真实远端、资源和保存

复用D80部署通知前的只读gzip发布；500次部署准备fetch，线上1160474851
bytes，物化19482573296bytes，请求内动态打包0。必要传输与本地准备仍在
实际时间线上，不事后扣除。运行期间未维护远端、清理、压缩或启动另一实验。

服务内存峰值77310349312bytes，high2896、max/OOM/kill均0，swap峰值
1971212288bytes；host available最低35597443072bytes，disk free最低
141864009728bytes。软限制回收和swap是本轮性能条件，不能宣称无影响。
离线分析high21429单独记录，不混入服务性能；无OOM，分析资源域已实际关闭。

维护已关闭身份匹配的本地与远端服务，保存54项Ray日志、远端传输/监控和
resolver回执，未复制权重。RAW：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d795_20261024`。
完整主证据为audit_full4000_w2.json、launch3b.json、worker_0.log及SHA清单。
本轮重复objects回收与备份以实际回执为准，不提前宣布完成，不改他项目k3s。

12:51回收完成：仅删除4000个经过内容、保留原副本及引用核查的重复objects
文件，释放19497304064bytes，磁盘可用161200472064bytes（约150.13GiB）。
147项保护和用户dirty SHA不变。不可读同UID进程2628、3360275、3361057
如实记录，不冒称全系统FD可见。storage资源域已实际关闭，备份另看回执。

后续3B Loquetier/HydraServe必要适配与完整矩阵、全基线比较和PrimeLoRA
优化、同版本Full、A1–A5、S1–S13、成本/Pareto、两轮审稿文档、图表和
最终V2交付全部保留。当前结果不等于完整目标已经实现。
