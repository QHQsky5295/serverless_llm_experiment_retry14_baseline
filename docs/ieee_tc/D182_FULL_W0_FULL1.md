# D182：D181 请求范围测量的 7B 完整回放

状态：2026-10-03 07:08+08 完整回放正常结束，日志报告 4000/4000 完成、0 失败。
本机与远端收尾、请求级数量合同与完整指标已核验；G1/G2 仍未达标，详见下文。
运行版本 `a4f4c0630534ed2c4b6f20f4f84e1f262884904f` 已推送备份。

唯一变化是 D181 的新鲜完整 identity＋当前请求 HOST footprint 观察，
以及与其配套的 scope-aware in-flight sharing 和一致性检查。
沿用 D180 依赖证据及 D181 974 项回归资格；不叠加新优化或重新调参。

与 D179 使用相同的 7B/cap4、D157 profiles、source42 W0 4000 请求、
500 adapter subset、fixed-length greedy、60 秒共同通知与 1800 秒保护。
新配置仅更换输出、NVMe、HOST 三个独占路径；不生成新工件或 trace。
真实远端复用 D78/D80 已发布缓存，不按请求打包、不注入等待。
两端 NIC 启动前均为 1000 Mbps/full；不在推理中维护远端。

## 待验证问题

减少无关 HOST tensor 遍历能否降低用户 TTFT/TPOT 与等待？按 adapter
分开的读取可能增加 RPC/collection，必须连同总等待一起衡量，不能把
组件扫描减少比例当端到端加速比。planner/admission/物理预算观察不变。

这是 n=1 开发回放，无置信区间或显著性声明。全部原生 token 数量匹配
仍不替代数值 adapter 身份验证；数值正确性、共同参考、旧 Prime 新指标
对照、G1/G2 继续待完成。3B 和外部基线仍暂停。

完成后依次：真实资源释放→匹配远端记录→复用有界分析→完整诊断表→
解释净效应与剩余差距→备份。失败、不利结果和所有旧原始数据保留。

原始目录：`results/ieee_tc/p2_backend_qualification/d182_20261003/`。
启动前验证 SHA：`af0127bb119c34b8b3c25f855f0ce660c59742713f0dc4522d79f14e73e4fb6f`。
385 source refs、147 protected entries 及 Plan/V1 校验通过。

## 本次运行身份

- tmux：`tc_d182_full1`。
- service scope：`primelora-tc-svc-1897397af148439584458b834565894d.scope`，
  InvocationID `f9e875290aa24ef08c527c7d1954f02e`，72/80 GiB、swap 2 GiB。
- auxiliary scope：`primelora-tc-aux-778f05587c94420c9cd0211b707bbfd6.scope`，
  InvocationID `e6c17cc1fea24d3c803dc7a86a9d6422`，3/4 GiB、swap 0。
- 远端 3B/7B 服务 InvocationID：`d988110286864a36a2a0a2edae14677d` /
  `a94375ea624d4b4498d080ab8da59ee5`；仅 LoRA 制品服务，无 3B 推理。
- 远端 monitor：`primelora-artifact-monitor-d182full1.service`，
  InvocationID `1d9769a39e244649a5f6dad9bb48c6ef`。
- 7B 远端时钟：`remote-process-monotonic:dd8b98388c4347c4a1d67143c8a41b2d`。
- 监测日志：远端 `/home/lab14/primelora_remote/tc/d182_20261003/remote_monitor_7b_full_full1.log`。
- CPU prelaunch/health 已结束，scope 自动消失；InvocationID 分别为
  `cffb6b2440ea41b5a3ab5c579a6c0d6c`、`d06279524ee445fdb95008575192433d`。

## 运行结束与收尾（尚未封存分析结果）

- 07:08 最终 launch 记录：service/replay/watchdog 返回值均为 0，
  native GPU context release confirmed=true，service path removed=true。
- 07:10:05 独立检查：service/auxiliary scope 均 inactive、路径消失；
  已记录的 replay PID 3456721 不存在，GPU 进程实际普查为空。
- 4048 个资源采样保留待汇总；最后一条是 service_domain_gone 事件，
  不是资源采样。资源域自动移除后，不伪造新的最终 cgroup event 读数。
- 07:10:15 在本机释放后，按本次 InvocationID/PID 停止三个远端服务，
  均为 inactive、PID 0、Result success；推理期间未进行远端维护。
- 远端 journal 经本次 health clock 匹配，下载前后 SHA 一致：
  `332bb49bfcd50b63c988a03e600bd4512e8de7bfe72a46cfcd31b97cade1276e`。
- 远端 monitor 下载前后 SHA 一致：
  `065f5ea0d374761410cd2ebd4d7d126e6c60935c66a7b4986d8bf3c4ce1344c9`。
- 复用 D179 收尾脚本并绑定本次实际身份；没有复制历史 PID 作为本次证据。
  下一步复用有界分析流程，旧 D179 请求投影直接复用，不重新解析旧大文件。

运行完成与统计合格分开记录。终端旧 5000 ms SLO、CE、逻辑 GPU 汇总
不替代冻结 V1 的共同阈值、正确性和物理生命周期资源指标。

## 完整结果与精确对照表

状态：完整回放、释放、请求审计和离线分析完成；证据校验/封存另存回执。
以下全部指标来自同一完整比较 CSV，保留 11 项，不挑选有利指标。
依据 academic-plotting 的数据来源规则和 Plan §11.2，用表呈现精确多指标；
两版本各 n=1，不画伪 CI，不宣称统计显著或已分离的因果效应。

| 指标 | D179 | D182 | 低优指标相对下降 |
|---|---:|---:|---:|
| 平均 TTFT (s) | 3.117377 | 2.730194 | 12.420% |
| P95 TTFT (s) | 7.071532 | 6.375893 | 9.837% |
| P99 TTFT (s) | 9.855402 | 8.681467 | 11.912% |
| 平均内部 E2E (s) | 8.030663 | 7.389124 | 7.989% |
| 平均 dispatch/admission wait (s) | 1.932886 | 1.735268 | 10.224% |
| 平均 service TTFT (s) | 1.184491 | 0.994926 | 16.004% |
| 平均 native TTFT (ms) | 377.321400 | 340.622429 | 9.726% |
| 平均 TPOT (ms) | 42.412898 | 39.631695 | 6.557% |
| P95 TPOT (ms) | 75.252008 | 68.328108 | 9.201% |
| 平均 response pickup (ms) | 473.087758 | 437.978163 | 7.421% |
| 物理生命周期 GPU 占用 (GPU-s) | 15940.368568 | 15928.394194 | 0.075% |

本次观察到平均/P95/P99 TTFT、TPOT 与内部 E2E 均下降，支持继续保留这项
实现作为开发候选，但单次对照不足以声称稳定收益。物理 GPU 占用仅下降
0.075%，不足以认定资源节省；四个 runtime 均真实持卡至释放。
不能用终端逻辑 GPU/旧 CE 数值替代物理 15928.394194 GPU-s。

### 完成、身份、远端与资源核验

| 项目 | D182 观察 | 资格边界 |
|---|---|---|
| planned/submitted/started/terminal | 均 4000 | 无漏请求，失败 0 |
| 原生输出数量 | 4000/4000 与目标一致 | 不等于 adapter 数值正确 |
| 对照 D179 | 15 个非输出字段逐请求相同 | request、arrival、prompt、adapter、target 等一致 |
| 输出 hash | 3895 相同、105 改变 | 保留审计，不宣称逐 token 等价 |
| n_correct | unknown | GPU-s/correct-request 不填值 |
| dispatch 前 tier | GPU1300/HOST1746/NVMe818/remote136 | 4000 完整，冲突 0 |
| GPU allocation/release | 4/4；无 open lease | 不等于 G1/G2 合格 |
| remote 获取 | 132 UUID 对，全部 published | 与 selected-remote136 不是同一集合 |
| 线上字节 | client/server 均 131480060 | 真实传输保留，按请求打包 0 |
| 已核验逻辑字节 | 3300789780 | 不冒充线上字节 |
| 资源采样 | 4048；峰值20562866176 bytes | high/max/OOM/swap/warning均0 |
| 最低主机可用内存 | 93393641472 bytes | 无安全中止 |
| 源状态重试 | 90 请求、94 次、最多2次 | 原请求时间线包含重试 |

所有 4000 请求的 TTFT、TPOT、dispatch 和服务阶段恒等式通过 1 ms 规则，
本次所检字段最大重算误差为 0 ms。内部控制器完成与外层终态均不是客户端
收到响应；没有客户端接收事件时不把内部 E2E 冒称其精确边界。

### 同一候选时限下的缺口

仍用 D170 候选，不修改参考：input≤616/>616 的 TTFT 阈值分别为
2.990091890/4.635325166 秒，TPOT 为0.059646198/0.085345274秒。
参考 batch8/slots8/seqs8/rank16/GPU0.92 与 Full cap4/slots4/rank64/GPU0.70
不同；共同参考尚未最终冻结，数值正确性仍待验收。

| 候选时延分类 | 请求数 |
|---|---:|
| 两项通过 | 3200 |
| 仅 TTFT 未通过 | 670 |
| 仅 TPOT 未通过 | 85 |
| 两项均未通过 | 45 |
| 失败 | 0 |

联合时延上界从 D179 的73.525%上升到80.000%，增加6.475个百分点。
离3800请求门槛仍差600请求。该值是暂定共同参考下的 timing-only 上界，
不是正式联合 SLO、不是正确率，也不构成 G1/G2 达标。

### 延迟和工作量的正反两面

平均首 token 分解：

| 阶段 | 平均秒数 |
|---|---:|
| arrival→admission | 1.735268 |
| admission→native dispatch | 0.654304 |
| engine entry | 0.109680 |
| native queue | 0.016733 |
| native prefill | 0.214209 |
| 合计 TTFT | 2.730194 |

进入引擎前平均2.389572秒，占TTFT约87.52%；575请求在进入引擎前
已超过候选TTFT阈值。首要缺口仍在引擎前，不能只凭网络或GPU型号归因。
arrival→gate为0.626012秒，gate→source admission为1.109256秒。
native→末token为4.493419秒，末token→控制器完成为0.506133秒，
控制器完成→外层终态为0.234767秒；不将重叠诊断区间重复加到账单。

1764次控制采样中766次存在队列，其中737次active小于ready容量。
采样不是连续GPU空闲证据，ready容量也不意味着每个adapter立即可接纳。

源状态读取请求4578→4586，collection2138→4249，RPC8487→16930，
共享等待2440→337，stale rejection485→492。按请求范围分离减少了
无关HOST遍历，也减少了不同目标之间的在途合并；不能说调用次数下降。
尽管本次完整延迟改善，未来优化仍需同时衡量总调用、总等待和安全语义。

planner共2052记录：初始化2个、owned execution epoch2050个，
均completed；worker CPU累计1165.539755秒，D179为1164.128076秒。
目标计算均值87.223547ms/P95919.628469ms，不能据局部时长认定总CPU收益。
本次无取消记录；D179的取消原始证据保留，不追改。

### 分析失败、类型追溯与修复范围

首次 timing 分析退出1：严格 helper 要求 Python int，但保存的原生计数为
759.0/152.0这类精确整数值。首次脚本、日志、scope与失败返回码全部保留。
检查原始D179/D182文件均存在该表示；运行代码的RPC timing归一化
（scripts/run_all_experiments.py 的 _attach_parent_rpc_breakdown 与
SubprocessInferenceEngineProxy.generate）明确将数字转换为float。
本机jq1.6将152.0重写为152，conda jq1.7.1保留152.0；未记录首次投影
可执行文件绝对路径，因此不追称已确定历史进程使用哪个jq版本。

只修正本轮离线入口：有限、正值、严格整数、<2^53，并逐请求与已保存的
整数input_tokens/completion_tokens核对；不四舍五入、不截断、不用expected
填actual，不修改原始结果、严格helper、V1或服务源码。两个正表示和10个
非法值（含bool、字符串、非整数、非有限值）通过检查，4000/4000两个字段
均精确一致。第二次分析成功，第一次失败不删除。不重跑GPU、不再投影大文件。

首次封存校验还发现分析脚本复制时误替换了投影scope的历史身份字符串；
该次校验中止，未生成成功回执。对照实际scope原始文件修正，保留失败脚本/
日志，第二次用独立资源域核验。未改实验结果或降低身份检查；失败校验的
scope已自动消失，退出前未保存最终内存事件，不补造零值。

### 来源、封存与下一主线

有界metadata87811719 bytes（128MiB门槛不变）；请求投影34441471 bytes，
SHA `d548f108f6b3872ccfaa4152c65b4899aa2035595225481159d5492a42e2c7ab`。
D179直接复用既有34MB投影/curated，不重新解析旧完整结果。
本次curated SHA：
`19c0aa6d4853b3dbb7730823a12a6efc9f9e93f72cf3350a545e005d34ad6c99`。
所有分析使用3/4GiB、swap0、CPU2,3,26,27；各scope实际身份、完成与移除
保留回执，最终旧投稿147项保护文件再次核对。没有新GPU/远端服务运行。

下一步仍为Prime7B：从已观测到的引擎前等待与控制CPU支出定位一个新的、
可证伪的瓶颈；结合历史修改和原始论文/官方实现再进行最小验证。
本轮不叠加第二项优化，也不重跑相同假设寻求更好数字。
旧Prime按新指标的对照、数值adapter正确性、共同warm/Resident参考与
G1/G2仍未完成；不能只胜过D179慢候选就关闭7B。随后才推进3B与外部基线。
