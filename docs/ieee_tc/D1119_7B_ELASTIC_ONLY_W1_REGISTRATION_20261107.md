# D1119：7B 当前版本 ElasticOnly W1 复合消融

## 最新状态：分析完成

见 `D1119_7B_ELASTIC_ONLY_W1_RESULT_20261107.md`。4000原生合同，联合
99.675%，U8575.626704086084，P95TTFT1035.5770019814372ms；与D1115
完整输入匹配，179输出摘要差异保留。Full节省11.3316%资源但时延/SLO更差，
按资源—服务质量取舍解释。机制关闭与真实selector均通过执行记录核验。
本键封存，不再运行；下文旧“正在分析/运行”均为历史状态。

## 当前状态：2026-11-07 22:56 +08 完成，正在分析

同一panePID1953868已终止且exit0；4000成功，全部11次物理分配释放，
GPU计算进程为空。service/replay/watchdog均0；完整占用8575.626704086084
GPU-s。尚未完成冻结联合SLO及Full比较，不由此预设胜负。
对应clock的远端journal已下载到本轮raw/remote_transfers.jsonl，双端SHA
`93f4ec19a36fd8b7a8a305f344e94d119200bf39102841f2735a4c201e7381ff`。
仅离线扩展现有统计入口的显式scenario并保留默认Full；不改运行数据。
下文启动和监测状态保留为历史，不作为再次启动指令。

## 当前状态：2026-11-07 22:18 +08 已启动一次

22:20:41复查：同一panePID1953868存活，76到达、72完成且成功、0失败；
已观察三次自然扩容，当前4runtimes。watchdog sample142无告警/中止，
high/max/OOM/swap全0；available99.958477GiB、disk160.037388GiB。
这是局部进度，不是全量SLO或资源排名；不使用legacy5000ms显示替代冻结评价。

tmux `tc-d1119-elastic-only-7b-w1`，panePID1953868，22:19复查仍存活。
启动前检查点 `02bccc884f155c295208557eed31238910431a28` 已推送到
faaslora_origin/retry14_continuous_queue_v2，远端HEAD完全一致。六个明确
文件提交；用户dirty manifest和历史状态文件未stage，144旧投稿文件未变。

实际service为8f3642ca72f942588447b751d6ab3950，invocation
a16fd363b6f1482a8f60d2a3ce51db9f；aux为b5f3fc5c3706466589f987e0d70237f2。
通知3568345.783442678，业务起点3568405.783442678，间隔60秒。
W1viewSHA103ac4add35a43c4c2859cb1d9118ea18971186e7332ba8f025eb2754e98d9a3，
sourceSHAefb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c。
实际日志场景为tc_ieee_elastic_only，不是Full。

远端本次InvocationID3a5aa97080a94984ad26ddb9265c6d25、PID1158555；
clock为remote-process-monotonic:69ab6e9a4e864acda9c3607846d854a5。
初次health在服务就绪前失败，随后published-gzip/timing-v2验证通过，
全部发生在部署通知前。两端1Gbps/full，没有网络调整或动态打包。

启动前available119014620KiB，disk172372598784B，GPU计算进程为空。
sample54时available115941040128B、disk172370706432B、服务high/max/OOM/
swap全0，无告警或中止。尚未完成性能实验，不能据此给出胜负。
接下来只监测同一个运行；结束后真实释放、校验、分析，不重复启动。
运行交接仅本地轻量更新；下一备份在测量结束后。下文“尚未运行”为历史登记。

2026-11-07，启动前登记；尚未运行。延续已批准 PLAN §8 A1/A5，不新增
baseline、负载或优化候选。D1117/W1 与 D1118/W2 已关闭并封存，上一目标
阶段有实际结果进展；本次转入当前版本 7B A1，不重复 Full 或五个基线。

## 科学问题、对照及终止条件

相同后端、弹性控制和资源配置下，主动 adapter 管理整体是否减少 GPU
生命周期占用并维持共同服务要求？这是复合消融，不是 confirmed-state
propagation 的单因素证明；后者仍由 LastKnown/A-A/shadow/S2 回答。

Full 固定复用 D1115 W1：4000 原生合同、联合98.55%、U7603.874677798245
GPU-s、P95 TTFT2331.3295198604465ms。其配置是 D1100/max4，不使用另一
工作点 D1117/max3 的成本或延迟拼接。Full 原始结果 SHA 为
`99574e407543545937b93d0277c7c70a4d3c28eb8107290d79db65aa1777e577`。

若主动管理有效，预期在相同工作量与共同 SLO 下降低完整 U，并由远端访问、
准备重叠、选中副本 tier 和队列等机制事实解释；不预设必须改善的百分比，
不把命中率、CPU fixture 或开关存在等同于最终收益。双方均报告联合 SLO、
GPU-s/offered request、P95 TTFT/TPOT、失败和阶段分解，按冻结 G1/G2 解读。

本键第一次有效完整观测即封存，不因输赢重跑。缺失/失败保留；执行错误只在
确证后修复受影响项。若 Full 不占优，先检查非目标差异及机制触发，再按
实际瓶颈判断是否有可证伪的改进；不加延迟、不削减对照资源或正常缓存。
3B D1057/D1085 的既有反例及资源—时延取舍保留，不混成本模型比较。

## 唯一目标变化与公共条件

新配置：`configs/ieee_tc/20261107_d1119_7b_elastic_only_w1.yaml`，SHA
`2796e6d03e257ccfb69bd76be36058c966ef2ac6c978fc7ff6061413da000a74`。
来源 Full YAML SHA：
`386a34c84ec70e5d0d366f6d3bcaf0891536be503ba1359d583a93bd5ecf303f`。

- 目标开关复用 D1057：`elastic_only_v1`、`load_only`、`no_handoff`、
  普通 `demand_lru`，关闭 coordination、hierarchical residency、主动
  effective-capacity admission、forwarding 和 HOST hit promotion。
- load-only 为既有 Q 元组（admitted、pending loads、GPU utilization），
  再按 last-dispatch/replica ID 稳定决胜。不使用 tier/D+T+O 排序，亦不
  改成另一条 legacy least-connections 路径。
- 保留物理可行性、原子 reservation、引用/取消、原生按需加载和缓存。
  只在 run 开始使用新缓存，不在每个 phase 清空。未给 7B 新增 3B 专用
  arrival-prefetch、模型 profile 或初始化策略。
- vLLM0.30.0、FP16/TP1、14 slots/seqs、1024 batch tokens、1–4 replicas、
  相同预测伸缩规律、45秒历史/38秒cooldown、D362 profiles 和预算不变。
- 4000 个既有 W1 请求、500 逻辑adapter、seed42 source、8×500/30秒间隔、
  固定greedy原生token合同、60秒通知、真实远端已发布gzip交付全部不变。
- 只有批准开关、准确 scenario 名称和三处独占结果/缓存路径不同；完整
  YAML 归一化相等测试同时覆盖历史3B与本次7B。

仅扩展现有 D1100 runner 的第七个显式 scenario 参数，默认 Full 和旧
0/5/6参数调用不变。实际执行身份 `tc_ieee_elastic_only`，不改原始JSON
名称冒充Full。生产算法源码本次未修改，IEEE九公式保持原语义。

## 历史、官方依据与最小验证

已核对 D1057 entry/implementation、D1083 semantics、D1085 及当前实际
router、variant contract、owned qualification、scale-out 入口。复用经
执行的机制，不增加500-ID总资格、100/1000请求回放或新测试框架。

再次读取 [vLLM0.30.0官方按需LRU](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
及 [dLoRA原论文](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)：
关闭主动管理不等于取消正常缓存；联合request/adapter管理应与弹性控制
分开检验。这些来源只界定设计和归因问题，不提供本实验的性能结论。

使用 ablation-planner 做已批准设计的本地复核；其指定外部模型不可用，
标记 `[pending external review]`，不声称独立审阅完成。run-experiment
用于既有 tmux 运行入口；github-sync 用于提交前明确文件检查及备份。

17 项 CPU 测试通过：非目标配置相等、Full/ElasticLocality回归、实际
ElasticOnly owner/场景、保持扩容且无主动准备、无handoff仍可激活、
load-only排序与物理可行性。shell语法及diff检查通过。测试资源域3/4GiB、
swap0、CPU2,3,26,27；未运行GPU性能测试。
首次测试命令遗漏 WorkingDirectory，因导入 tests 失败（无测试实际执行）；
保存 tests.txt，修正启动目录后 tests2.txt 通过，不覆盖失败记录。
路径：`results/ieee_tc/p2_backend_qualification/d1119_20261107/prelaunch/`
位于 `/home/qhq/serverless_llm_experiment/`。实际机制触发仍须完成后核验。

## 执行与安全

先备份配置/入口/测试/本文，再由既有入口新鲜检查≥102GiB可用内存、
≥max(150,100+1.5×32)GiB磁盘、GPU计算进程为空及真实远端身份。
保持72/80GiB服务、3/4GiB辅助、swap2/0和固定CPU隔离；两端链路核查。
远端只管理本项目7B服务的实际InvocationID，不碰共享k3s或其他作业。

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1119_20261107/7b_elastic_only_w1 \
  d1119_7b_elastic_only_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1119_20261107/7b_nvme_elastic_only_w1 \
  /dev/shm/tc-d1119-7b-elastic-only-w1 \
  configs/ieee_tc/20261107_d1119_7b_elastic_only_w1.yaml \
  tc_ieee_elastic_only
```

tmux名 `tc-d1119-elastic-only-7b-w1`。实际启动后另记PID/时间/资源域，
此命令本身不证明运行已开始。测量期间不并行重型任务、压缩、哈希扫描、
测试、备份或远端调整。保护旧投稿与用户dirty manifest，不stage无关文件。

终态后先核验真实释放，再收集对应clock的远端journal。复用D1113 collector、
D299 review和已有A1机制分析；7B统计入口尚硬编码Full，应仅增加显式scenario
并保留默认值/旧协议及原始身份，不改原始JSON、不复制大结果、不改指标。
逐请求与D1115核对完整输入及输出摘要差异；routing4000、handoff/主动
file/GPU/HOST/residency计划0等必须来自执行记录，不由配置直接推定。

PLAN SHA `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`；
冻结V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`；
单轮V2 SHA `193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`。
以实际文件SHA核验为准；不改协议字节。D394共同SLO/D594预算引用保持不变。
已知开发trace、单轮、n_correct=null/独立数值正确性未证明等限制不隐藏。
本次只推进7BA1；其他A1–A5/S1–S13、成本/CE/Pareto、两轮审稿文档、图表
及最终V2集成全部仍在完整目标中，不据一项消融宣告完成或全局最优。
