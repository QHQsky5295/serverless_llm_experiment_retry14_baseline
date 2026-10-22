# D721：短输出解码与其他请求的重叠——完整日志复用

2026-10-22。主线为3B PrimeLoRA的共同联合SLO缺口。本次没有新的GPU
实验、生产优化、配置选择或图表；复用D703和D719两个完整4000请求日志。
九公式、冻结V1／单轮V2、真实远端和旧数据均不改。当前D719仍为91.60%
联合达成、13508.300779 GPU-s，独立数值正确性未知，不能进入G1达标排名。

## 1. 为什么做这个有界分析

D719已消除重复相同否定来源的重试，但336条联合违约中320条没有前序
路由重试。继续优化罕见分支不能回应主要缺口。另一方面，D716显示短输出
TPOT违约较多，尚未把“与其他请求prefill重叠”和“没有这种重叠”分开。

本次检验的是：**将短输出违约统一归因为prefill干扰，是否符合现有证据？**
不是估算删去某段时间后能新增多少达标请求，也不是再分析已闭合的请求总数
问题D720。短输出沿用D716的2–64 tokens分组，完整总体和长输出同时保留。

复用现有`audit_d703_3b_full.py`及D716小型CSV。每条请求重新根据同一owner、
worker、replica和时钟中的实际区间计算交叠，排除自身和其他owner；交叠区间
取并集，不重复累加。prefill区间是原生scheduled至first-token的包络，
不是纯GPU计算；按需加载只计实际调用native load的区间。GPU命中没有加载
区间时保留null，不补零，也不把主动准备／状态查询偷偷计入按需加载。

## 2. 全部短输出的互斥分组

两轮各有1539条短输出，表中格式为“请求数／其中TPOT违约数”。

| 解码期间的同owner重叠 | D703 | D719 |
|---|---:|---:|
| 两者均无 | 1096／36 | 1035／32 |
| 只有其他请求prefill | 262／30 | 312／44 |
| 只有其他请求按需加载 | 39／12 | 47／16 |
| 两者都有 | 142／33 | 145／40 |
| 合计 | 1539／111 | 1539／132 |

全部输出长度的TPOT违约分别为116和144条；对应四个互斥组分别为
`37/30/12/37`与`38/46/16/44`，没有删除长输出反例。

### 观察、解释和边界

1. D719短输出违约中84/132与prefill包络重叠，48/132没有该重叠；D703为
   63/111与48/111。因此“全部短输出问题都是prefill造成”不成立。
2. D719有prefill重叠的457条短输出中84条违约；没有该重叠的1082条中48条
   违约。这是条件关联，不是随机对照；请求长度、负载、准备和控制工作可能
   共同变化，不能把这两组达成差称为prefill的因果贡献。
3. D719的32条“两者均无”短输出违约仍可能遇到主动准备、状态清点、CPU
   调度或其他等待；现有包络没有包含这些活动。不能用“无此类重叠”断言
   完全没有竞争，也不能将它们无证据地归为GPU算力不足。
4. D719全部4000请求中，最大TTFT分段为arrival→admission的有2896条，
   admission→acquired有699条，自身native prefill只有162条。这不是违约
   专属统计，也不证明每段的根因；但明确不支持仅盯prefill忽略服务前等待。
5. 即使把全部TPOT违约假想消除，D719原有TTFT达成仍只有3791/4000；这只是
   两个已观测边际的逻辑关系，不是可执行优化或反事实性能预测。必须同时
   处理TTFT与TPOT缺口，不以TPOT单项改善宣布联合95%达标。

两个运行是已有开发单轮，不能合并为8000个独立实验、输出CI，或据差值认定
D717身份传播修正独自造成全部变化。输出90个分组（每轮45），空组均值为null。

## 3. 代码历史、官方依据与方向裁决

已核对D691/D692以及D684、D674、D685–D689：

- 原子控制调用之后的模型step交错已生效，不能重做“限制每轮一个utility”。
- D514的GPU准备identity-only替换、同owner准备串行、批量pin查询、HOST描述
  物化和scale-down阈值均有既有反例／拒绝记录；本次不恢复或改名重试。
- 每个控制采样触发residency epoch不自动等于冗余。需求窗口、来源、容量与
  pending状态都可能变化；本次没有证明相邻epoch输入等价，不能凭周期2秒
  直接减少规划频率或缓存旧物理快照。
- 当前3B已启用chunked prefill，`max_num_batched_tokens=4096`、seq上限8、
  最长实际prompt约760；仅“开启chunked prefill”不是新优化。现有runner
  没有传入单请求`long_prefill_token_threshold`，本机0.30.0默认0关闭该cap。
  是否实际分块还由当次剩余token预算决定，不能仅看开关宣布已限制干扰。

本轮核查的原始资料：

- [vLLM 0.30优化文档](https://docs.vllm.ai/en/v0.30.0/configuration/optimization/#chunked-prefill)：
  decode优先后用剩余token预算安排prefill；小预算和大预算存在ITL／TTFT取舍。
  文档中的通用数值不是本机实测最优值，不直接采用。
- [vLLM 0.30 scheduler源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/scheduler.py)：
  running和waiting路径均处理显式long-prefill cap及剩余token预算。本机文件
  SHA为`c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884`；
  scheduler config SHA为`822179da62be4bca0619c8590082ae09123f5c4415f9cc2037be673748239edb`。
- [Sarathi-Serve，OSDI 2024，§4.3](https://www.usenix.org/system/files/osdi24-agrawal.pdf)：
  token预算需要权衡decode时延与过细分块的额外开销。其逐token尾延迟与本项目
  逐请求TPOT并不相同，不照搬其512/2048设置或宣称相同收益。

**裁决：允许进入一个有界的原生prefill/decode干扰资格，不直接改变Full。**
下一步复用既有原生profile／iteration观察入口及旧请求、工件，先检验实际
prefill token数与混合迭代耗时，并分开控制调用时间。它针对新问题，不再重跑
没有新观察量的D684前缀。只有实际证据支持，才从测得的迭代开销与既定服务
要求导出一个模型级执行预算，按原计划最多两轮最小验证；不能把包络时长
当作GPU计算速度来直接推一个常量。

若改变后端调度配置，需新的配置身份及匹配的service/preparation profile；
不重用旧profile冒充同配置，不重标定共同SLO来放宽评价。完整W0验证后再
判断是否接受，并保留TTFT、TPOT、GPU-s副作用。此处尚未选定新token预算、
实施生产候选或承诺该方向足以达到95%。若资格不支持，停止此方向而非扫参。

## 4. 校验、失败和资源

114项离线检查通过，含新增跨owner排除、互斥计数、空组、缺时间／非有限值、
加载空值语义和错误重叠拒绝。已有4000条stage恒等式与封存计数重新一致。

分析准备发生两次错误，均在产生结果之前停止并保留：

1. 第一启动只请求AllowedCPUs，实际affinity检查拒绝；以显式taskset纠正，
   不把cgroup属性当作有效CPU隔离。原初小型只读计数的affinity没有核验，
   已明确标注，不作为最终受限分析回执。
2. 第二次错误要求GPU命中也有加载时间；补null语义与两项负例后重算。该次
   父进程和测试子进程同时导入运行库，产生2876次memory.high、无max/OOM／
   swap；最终把大型分析导入放在测试退出之后，未提高4GiB包络。

最终实际CPU为2/3/26/27，3/4GiB high/max、swap0；峰值2422059008B，
high/max/OOM均0。三次分析与最初计数均无GPU运行；结束后核对空进程并关闭
各自scope。144个旧投稿文件、用户manifest、PLAN/V1/V2与生产代码／配置
保持原SHA，没有清理唯一数据、复制工件或操作远端／非项目k3s。

产物：`paper_results/ieee_tc/p2_backend/d721_decode_peer_exposure/analysis3/`。
主统计脚本新增opt-in `--diagnose-decode-overlap OUTPUT --stage-audit PATH`，
不覆盖旧目录。原始分析及失败在`results/ieee_tc/p2_backend_qualification/d721_20261022/`。
按analyze-results保留观察／解释／反例；vLLM技能建议须经本机原始实现核验，
未用其通用吞吐或SLO数字。research-lookup默认服务无认证，使用官方网页
替代；github-sync只备份显式路径，不stage用户manifest或强推分歧分支。

7B已完成的基线及隔离HydraServe继续复用。3B W1/W2、其余外部基线/dLoRA、
同最终版本跨场景确认、A1–A5、S1–S13、CE、两轮审稿文档与最终图表仍开放。
本次为诊断进展，不是完整计划完成或已取得最优。
