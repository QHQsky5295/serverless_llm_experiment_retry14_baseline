# D1043：复用静态词法路径，保留全部实时文件观察

2026-11-03，3B PrimeLoRA 的单一开发候选。父源码
`0c7779926e8941a48cdc98300d4720bdbb54f0cc`。用户修改的 EXECUTION_STATUS.md
保持原字节，以本文件记录接续状态。尚未接受系统性能收益。

## 问题、历史和可证伪假设

D1033 完整 W2 联合达成3772/4000（94.30%），U=8155.283736 GPU-s；
距离既定95%还差28请求，独立数值正确性未知，不能进入最终排名。D1042
实际525次 preparation_snapshot 平均持锁54.581739ms、调用线程CPU
54.476792ms；扣除直接子调用包络后29.744612ms。这是诊断，不能从请求
时延直接减去持锁时间，也不证明该路径解释所有违约。

当前 D189 已保存静态目标字符串，但每次遍历500候选仍重新执行1000次
`root / adapter_id`，并首次解析/哈希所得 Path。D1043 的既有微型 fixture
测量9.771513、9.645572、9.572515ms（不带 profiler），独立 cProfile
显示1007次路径拼接及大量首次解析。其累计时间相互包含，不相加，也不
把微型4个现存adapter、496个缺席名称的结果外推为实际146副本缓存。

唯一候选：让已有内容/subset/root/分配粒度绑定的静态模板同时保存不可变
Path，规划循环直接查用。只删除重复词法构造；每轮仍完整实时清点与二次
stat、源签名/epoch/预算/预留/替换计算，执行前重检查不变。首次模板构造
和额外元数据内存仍在被测服务内计量，不新增参数或免费预处理。

历史区别：D188/189 是内容摘要与目标描述；D457 是单次观察索引；D606 是
inventory 内规范字符串；D610 是同次 extent 结算。本候选只处理规划候选
循环剩余的静态 Path。没有恢复 D844/D856 原生stat批处理、已撤回的
D860/D861 来源投影、D628 SHA移锁、D850通知提前、D851–855线程迁移，
也没有用 D329 已否定的 owner/epoch 代替物理事实。D1037仲裁修正已存在，
尚缺完整性能验证；后续与D1033比较须披露这一共同源码差异。

## 官方依据与语义边界

- [vLLM CPU开销分析](https://vllm.ai/blog/2024-09-05-perf-update) 支持基于
  实测检查对象管理成本，不提供本系统收益或因果证明。
- [Python 3.12 pathlib](https://docs.python.org/3.12/library/pathlib.html)
  区分纯词法操作与文件I/O，路径为不可变可哈希对象。实际LLM环境的
  Python3.12源码也缓存词法字符串和hash；这不是文件状态缓存。

九公式、输入工件、源trace、远端交付、路由与admission、缓存容量、
SLO/预算和资源包络完全保持。binding变化替换整个模板；输出仍为独立
字符串/字典，不向外泄露模板，不缓存 stat 或确认状态。

## 有界验证与下一步

先保留父版新测试的失败：已编译模板的496个缺席候选不应再次产生992次
路径拼接；同时检验每轮仍一次新清点、输出逐字段一致（仅忽略捕获时刻）。
实现后复用现有静态描述/实时文件观察测试，覆盖内容/subset/root/几何
重绑定、源变化撤回、二次stat竞争、pending增长、返回值隔离。
仅做一次三对交替的组件对照，保留所有值；不是三轮系统性能或CI。
若无局部收益则关闭，不做参数扫描；若通过，备份后进入一次普通完整
4000请求W2，不额外做短Full或重复诊断。正式采用由完整结果决定。

raw：`results/ieee_tc/p2_backend_qualification/d1043_20261103/`。
当前新增profile_snapshot.py仅在raw内执行，未被生产导入；生产候选只涉及
residency_manager.py，测试扩展现有test_ieee_tc_static_preparation.py。
CPU资格复用3/4GiB、swap0、CPU2/3/26/27、CUDA隐藏；保护147项历史文件及
两项用户修改，不修改远端、不接管共享k3s、不生成图。

同版本两模型W0/W1/W2、G1/G2、独立数值正确性、最终外部比较、A1–A5、
S1–S13、CE/Pareto/价格、两轮审稿及B1/B2/C1/C2/C3文档、最终IEEE图表
和V2集成仍待完成。已有HydraServe隔离适配与baseline观测不重新执行。

## CPU资格完成：进入唯一完整W2，尚未采纳性能收益

父版计数测试按预期失败（992 != 0）；第一版32项通过。复核发现应明确
保留当前 owner 的 tier 迭代顺序，不能让静态映射顺序成为输出顺序来源；
因此最终版使用当前 roots 的顺序查用静态Path，并增加顺序回归测试。
最终33项全部通过，包括内容/子集/根/几何重绑定、返回值隔离、文件变化
撤回、二次stat变化拒绝及pending增长。两版结果全部保留，不选较快版。

最终版有界组件对照（ms，三对交替调用，不是独立系统实验或CI）：

| 已有500-name微型fixture | 第1次 | 第2次 | 第3次 |
|---|---:|---:|---:|
| 父版热模板快照 | 10.480053 | 9.703790 | 9.607512 |
| 最终候选热模板快照 | 3.103515 | 2.504337 | 2.743632 |
| 父版首次模板快照 | 17.767280 | 17.432843 | 27.640102 |
| 最终候选首次模板快照 | 13.098008 | 12.608726 | 12.135511 |

12份冷/热输出逐字段一致，仅忽略捕获时间；热模板组件均值减少约71.97%，
不是系统TTFT或GPU-s减少71.97%。实际缓存的扫描量不同，不推算整轮收益。
首次组件启动使用了33位而非32位scope标识，在任何fixture/测量前失败；
保留错误记录，只修正启动身份，没有产生被舍弃的性能数字。

所有已完成辅助scope已移除；147项保护文件及两项用户修改SHA不变。
资格、各次原值、来源SHA及失败回执位于
`paper_results/ieee_tc/p2_backend/d1043_static_lexical_paths/prelaunch/`。
不追加短Full、原生资格矩阵或调参扫描。

唯一完整执行：配置
`configs/ieee_tc/20261103_d1043_3b_static_lexical_paths_w2.yaml`，相对D1033
仅三个独占结果/工作缓存路径变化；已逐字段断言其余完全一致。
RUN_ID=d1043_20261103，RUN_TAG=d1043_3b_static_lexical_paths_w2_full。
复用原D1000 runner、实际W2、D747 fresh descriptor；显式关闭prefix和所有
详细profiler及进程file owner。4000既有请求、500逻辑ID、真实远端已发布
归档、60秒通知、单轮V2、32槽/c8/4096 token/1–3副本和资源保护不变。

共同D648阈值：prompt≤506/>506时，TTFT1466.050801102/2539.648668191ms，
TPOT29.921872367/36.215515193ms；联合至少3800，G2另要求
U≤12083.063870 GPU-s。独立数值正确性未知仍须明确，不能用token数冒充。
PLAN/V1/V2/SLO SHA见qualification.json；不换评价尺子、不反复运行同键。
完整结果若不支持改善则保留并据实际阶段决定，不把组件收益写作系统优势。

备份首attempt在提交前停止：常规diff检查只报告新YAML及两份精确脚本快照
末尾空行（3项），没有内容、语法或测试错误。为保持已核验来源SHA与快照
原字节，后续只对这3个确切告警允许blank-at-eof，并继续检查其余所有空白
错误、凭据、allowlist及源SHA；没有放宽实验正确性或资源门槛。失败回执保留。

## 20:05运行中交接：已启动，不得重复启动

20:03:54在tmux `tc-d1043-full` 启动唯一完整W2。源码检查点
`d85e3fec81ec3f7bb2d44aa4eb16f2fd4138f1ac` 已提交并ls-remote核验：
origin/retry14_continuous_queue_v2及faaslora_origin/
backup/d1043-static-lexical-paths-20261103。实际FaaSLoRA V2仍有历史分歧，
未force、未声称最终集成；raw/backup.json保存核验。备份在GPU启动前完成。

raw根：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1043_20261103/`；
运行目录 `d1043_3b_static_lexical_paths_w2_full/`。launch.json须等待终态，
当前exec receipt、watchdog、回放与终态记录在launch.launch/，不能将缺少
launch.json误判为未启动。

- service scope：primelora-tc-svc-cde833376df547c6ab735e6b2998b93d.scope；
  InvocationID c26adf3869ce4e379ca40cbff882d110。
- aux scope：primelora-tc-aux-83f0b5ca00f04860ba8ec6f8eb37bc8e.scope。
- controller PID955220；watchdog PID955222/start_ticks321468439；
  external replay PID955180/start_ticks321468269；初始native GPU PID956557。
- exec receipt确认count/source_count=4000、W2、同trace/map SHA及60秒准备；
  不是prefix，tiny_witness=false。服务72/80GiB、swap2；外置3/4GiB、swap0。
- sample85：4条request terminal、服务峰值12.114GiB、host available93.396GiB、
  磁盘152.329GiB，high/max/OOM/kill/swap、foreign compute及escape均0。
  这是运行中安全观察，不是完整运行通过或性能结论。

下一动作只监视同一运行；不得并行测试、构建、压缩、推送或远端重启。
结束后先确认真实GPU及资源域释放，再复用D1042/D1033既有完整请求、D648
联合SLO、物理U、阶段与远端transfer分析；不套用诊断持锁数据，也不
把局部71.97%当系统收益。完整目标保持active，本文前述未完成范围不变。
本段运行中交接尚未推送，终态后随实际结果备份；被测源码已安全备份。

20:06:42–20:12:55继续同一回放，属于verified wait：直接读取/proc确认
controller/watchdog/replay分别仍为PID955220/955222/955180，start_ticks
321468420/321468439/321468269均未变化。资源sample234→531，终态记录
92→398；launch.json尚未生成，不是终态或重启理由。轮询回执为raw/
monitor_20261103_200642.json，保留所有本轮定时观察，不产生性能比较。
最后可用内存92.152GiB、服务峰值16.582GiB、磁盘149.744GiB；所查累计
high/max/OOM/kill与服务swap均0，无foreign compute/escape或保护告警。
磁盘已低于下一重型任务启动150GiB，但高于当前停止100GiB，不在性能
测量中压缩或清理；结束释放工作缓存后再决定是否需要维护。没有新代码、
配置、额外实验、远端操作或推送；上一目标turn为实际实现/验证/备份/启动
的progress，本turn继续已核验的活任务等待。完整目标及全部剩余范围不变。

20:14:00–20:21:25继续同一回放；上轮及本轮均为verified wait。未发生上下文
压缩或切换任务，上一轮完整阅读的PLAN本轮SHA复核不变；目标文件重新完整
读取。8次只读轮询的实际PID/start_ticks全部相同，watchdog记录年龄0–1秒，
sample646→1034、终态486→879。raw/monitor_20261103_201400.json保留这些
检查。最后服务峰值20.577GiB、host可用92.250GiB、磁盘147.655GiB，累计
high/max/OOM/kill和服务swap均0，所查无foreign/escape/保护告警。持卡数
有1/2/3卡变化，仅为观察，不在运行中外推物理U或判定完整性能。
launch.json仍未生成，三个实际进程仍live；不重启，不并行新实验/维护。
继续等完整终态与真实资源释放后再分析。所有未完成主比较、消融、机制、
敏感性、独立正确性、文档及图表范围继续保留；没有缩减完整目标。

20:22:31–20:30:55继续同一完整W2；前后goal turn均为verified wait，而非
阻塞。目标文件重新读取、PLAN SHA未变；当前10次直接PID/start_ticks
及实时watchdog核查均确认原三个进程live，记录年龄0–1秒，sample
1138→1597、终态1017→1525。回执raw/monitor_20261103_202231.json。
最后host可用91.956GiB、服务峰值21.725GiB、磁盘144.671GiB，
累计内存保护事件及服务swap仍0，所查无foreign/escape/保护告警。
没有终态launch.json，不重启、不追加GPU实验、代码修改、维护或备份。
后半程及最终释放仍需等待，未把部分样本计为G1/G2结果；下一动作仍为
监视当前同一运行，结束后按完整合同分析，所有剩余实验/交付范围不变。

20:31:56–20:40:32继续同一完整回放，前后turn均为verified wait。已重读目标、
monitor-experiment技能，PLAN SHA仍一致；10次实际/proc身份及新鲜日志
检查确认PID/start_ticks未变，sample1670→2166、终态
1617→2334，watchdog记录年龄0–1秒。回执raw/monitor_20261103_203156.json。
最后host可用91.777GiB、服务峰值23.048GiB、磁盘141.275GiB；
累计high/max/OOM/kill与服务swap0，所查无foreign/escape/保护告警。
实际进程仍live且无终态launch.json；不将过半样本作为完整性能结果，
不重启、不改变源码/配置、不并行实验或维护。下一动作仍为等待当前回放
完整终态及资源释放，再按冻结合同分析；全部后续范围及active目标保持。

20:41:37–20:50:05继续同一完整W2，仍为verified wait。上下文压缩后重新完整
读取目标、PLAN及monitor-experiment技能，核对AGENTS、用户状态文档和本交接。
8次实际/proc身份与新鲜日志核查确认三个PID/start_ticks不变，sample
2230→2730、终态2413→3031，日志年龄0–1秒。回执raw/
monitor_20261103_204122.json。最后host可用91.639GiB、服务峰值26.493GiB、
磁盘138.392GiB；累计high/max/OOM/kill及服务swap均0，所查无foreign、
escape或保护告警。没有终态launch.json，实际回放仍在运行，不重启。
本轮没有新实现、实验、性能结论、维护或推送；磁盘低于下一任务启动门槛，
但未触及当前100GiB停止底线，继续避免测量期清理。下一动作仍为等完整
终态与资源释放后执行既有完整审计；不把3031条终态当作正确数或SLO达标数。
用户EXECUTION_STATUS原字节不动，以此文件接续。全部未完成范围和active目标不变。

## 完整W2已结束：平均时延改善，但共同SLO与资源目标未改善

20:50:52–21:11:17持续核对同一进程，终态3066→4000；回执
raw/monitor_20261103_205052.json。随后launch.json正式写出，service、replay、
watchdog退出码均0；实际三个PID、tmux及两个scope均已退出，GPU无计算进程。
完整生命周期审计确认7次物理分配全部释放，不是用一次GPU空闲采样补造释放。
本turn从verified wait推进至完整终态、离线分析和结论归档；没有重复执行性能点。

复用D703原生计量、profile绑定、全请求阶段和D1033闭合证据收集脚本，
只适配本轮身份/路径。CPU分析在3/4GiB、swap0、CPU2/3/26/27资源域中执行。
4000请求的原生终态、adapter内容/请求绑定、fixed-output数量、源trace/map、
60秒通知、共同D648阈值和物理时间完整核验，TTFT/TPOT/E2E重算及六阶段
恒等式误差均0。独立数值正确性仍未合格，n_correct=null、ranking_eligible=false；
原生数量与内容绑定通过不能替代独立数值正确性。

| 3B W2，单轮开发观测 | 参考D1033 | D1043 | 变化 |
|---|---:|---:|---:|
| 联合SLO | 3772/4000，94.30% | 3759/4000，93.975% | -13请求，-0.325百分点 |
| TTFT达标数 | 3850 | 3831 | -19 |
| TPOT达标数 | 3913 | 3909 | -4 |
| 物理GPU-s | 8155.283736 | 8378.454562 | +223.170826，+2.737% |
| GPU-s/offered-request | 2.038821 | 2.094614 | +0.055793 |
| 平均TTFT，ms | 765.199131 | 635.541989 | -16.944% |
| P95 TTFT，ms | 1956.042105 | 1998.920931 | +2.192% |
| 平均TPOT，ms | 15.610976 | 15.711299 | +0.100323 |
| P95 TPOT，ms | 27.056324 | 27.380893 | +0.324569 |
| 平均E2E，ms | 2481.904323 | 2358.738980 | -123.165342 |

本轮仍差41个联合达标请求，不能进入G1/G2达标排名；虽然物理U低于既定
12083.063870GPU-s预算，预算通过不替代95%和正确性要求。局部组件约71.97%
加速没有转化为共同SLO/资源目标改善，不据平均TTFT或CE宣布成功。
上述差值是单轮描述，不是显著性或纯路径对象复用的因果效应。

完整八段联合达标数：D1033为465/479/484/476/477/466/468/457，
D1043为491/415/488/480/478/475/470/462。第二段（ordinal500–999，
第501–1000请求）减少64，其余七段合计增加51，净减少13。所有区段都保留，
不删除第二段、重算“去掉异常段”的主结果或按相同配置重跑寻找赢家。
241次联合违约中，TTFT-only150、TPOT-only72、两者19。169次TTFT违约的
最长阶段为arrival→admission102、admission→source acquired62、
executable→engine4、engine entry→queue1；这是阶段关联，不能直接称因果瓶颈。
全请求平均TTFT下降129.657142ms，其中arrival→admission均值下降126.692493ms；
同时该阶段P95从866.830126升至1029.951541ms，均值与尾部确有不同方向。

资源增加主要在共同到达窗口：+223.804973GPU-s；准备-0.099266、drain+0.506247、
清理-1.041127GPU-s，四个互斥窗口之和与总差一致。实际6次自然scale-up、
5次scale-down；在offset705.503s释放一张卡后，于744.968s重新分配，
另有不同释放/扩容轨迹。上述事实提示应优先对齐第二段的控制与等待时间线，
不凭时间重叠宣布缩容是已证实原因。D1037仲裁修正在本轮源码内，D1033没有，
不能把两个源码差异和动态轨迹的总变化都算作D1043路径优化的单因素效果。

远端505次实际交付，共1172100920线上字节；两端ID、字节和SHA均匹配，
请求期无打包。没有连续远端CPU/外部干扰监测，保留这一限制，不虚构排除。
4047份资源样本内high/max/OOM/kill和服务swap均0，foreign compute/escape为0；
服务峰值31.531685GiB，147项旧投稿保护文件及两项用户修改SHA不变。
工作缓存已由原runner清空，未删原始结果；本轮没有维护压缩、远端重启或新图。

结论：D1043不升级为性能参考，不接受“已改善G1/G2”的主张；D1033历史结果
完整保留。本轮已测代码与33项测试保留在检查点，不在此时盲目叠加新优化。
下一动作是复用第二段的控制事件、请求阶段和物理分配日志，核对何时缩容、
何时负载增加、何时请求排队及再次激活；对照D1037及历史已否定方向，再决定
一个新的可证伪方向。不得直接加固定冷却、常驻副本、放宽阈值或扩大缓存兜底。

完整产物：paper_results/ieee_tc/p2_backend/d1043_static_lexical_paths/completion/，
含full4000、profile_bindings、native_descriptor、stages、paired_d1033、
delivery_and_resources、closure与逐文件SHA manifest。复用收集器精确快照也已保存。
剩余同版本两模型/三场景、外部最终比较、独立正确性、A1–A5、S1–S13、
成本/Pareto/审稿文档/IEEE图表/V2最终集成均未关闭；完整目标继续active。
