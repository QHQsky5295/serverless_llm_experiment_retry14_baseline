# PrimeLoRA IEEE TC：冻结评价指标协议 V1

生效日期：2026-09-28。协议 ID：`primelora_tc_metrics_v1`。
用途：所有后续 PrimeLoRA/基线对比、配置选择、消融统计和论文主张核查。
在每次比较前完整读取本文件及 PrimeLoRA-PLAN.md，不以终端旧显示替代本协议。

**冻结状态：本文件冻结定义、选择目标、计算规则和失败处理；尚未测出的共同
warm 阈值、Resident 预算参考和各系统工作点不填虚构数值。正式比较须完成
这些数值标定并另存不可变 manifest，未完成不能宣布正式 SLO 合格。**

本文件落实已批准计划 §4、§7、§8、§10.2；不修改 IEEE 九个算法公式。
旧指标协议 tex 和历史论文结果保留，不原地覆盖。差异分析见
[HydraServe 指标核查](HYDRASERVE_METRIC_ALIGNMENT_20260928.md)。
对文献的采用以适用性为依据，不因其已发表就将不同资源模型视为相同。

## 1. 最终决定与目标优先级

1. 正确执行和计量有效性先于性能。不能用高 CE、低成本抵消错误 adapter、
   少生成 token、漏请求、资源越界或缺失的关键计量。
2. **G1：全部正确完成且联合 TTFT/TPOT 达成率≥95%，比较生命周期 GPU-s/request，越低越好。**
3. **G2：全部正确完成、共同 GPU 时间预算且联合达成率≥95%，比较扩容/churn
   场景的 P95 TTFT，越低越好；联合达成率是否非劣另行检验。**
4. TTFT 边际达成率、TPOT 条件达成率、联合达成率始终同时报告。
   原始平均/P95/P99 TTFT、TPOT、E2E、完成率、吞吐和资源总量不隐藏。
5. CE、美元成本、goodput、good requests/$、good requests/GPU-s 为补充。
   不用一个加权总分生成跨不同约束的唯一冠军。

理由：只比 TTFT 会忽略常驻资源代价；只比 GPU-s 会奖励不服务；CE 允许
延迟与价格互相补偿。G1/G2 分别检验“达标地节省资源”和“同预算更好服务”。
减少重复传输、提前准备、准确信息传播和避免加载干扰均有直接对应的结果指标，
但其收益必须由实际数据与机制对照证明，不能由指标选择预先保证。

## 2. 比较对象、输入和执行身份

- 模型仅已有 7B/3B；工件、源 trace、prompt/token 工作量复用，不新增整池。
- 主系统显示名：PrimeLoRA、vLLM、S-LoRA、Serverless、HydraServe、Loquetier、
  ElasticLocality。dLoRA 仅既定 3B W1/W2 范围。未合格的系统不填造性能点。
- `Serverless` 保留内部 `serverlessllm_new`/版本/补丁身份；不显示 `-new`。
- W0=原4000请求/500逻辑adapter/Zipf1.0/rotation500；W1=同序列8×500，
  phase内到达间隔减半、phase间30秒；W2=原arrival/prompt/token、rotation100。
  以已有参数变换和索引实现，不重新抽样到达。W1不等实际drain，不强求8次扩容。
- 正式主实验共同部署通知在业务前60秒；准备期间持卡计入资源。
  ready前请求排队，不按每系统ready时刻平移trace，不清空每个phase的合法缓存。
- 所有cold/first-touch使用同一真实远端已发布只读交付对象；实际传输、必要
  读取和本地准备计入请求路径，按请求临时打包不得进入正式路径。
  允许各系统原生缓存与合法预取，不强求相同miss/字节/下载次数。
- 执行键含代码/后端/配置、输入和工件SHA、generation、初态、timeout、控制
  规则、CPU/内存/swap/监控、远端协议与运行块。执行键相同可跨图复用。
- 当前 D90 是内部100请求诊断，不具备上述全部主协议条件，不能转作M1/M2。

## 3. 正确完成、失败与分母

每轮保留 `N_plan, N_arrived, N_submitted, N_terminal, N_correct, N_good`。
主运行 `N=N_plan=4000`。正确完成指示 z_r=1 要求正常终态、规定生成数量、
正确请求/adapter身份和完整原生计量；无静默base-model或本地remote fallback。

固定生成协议 `fixed_length_greedy_v1`：

\[
n_r^{target}=\min(n_r^{source\ expected},256).
\]

temperature=0、top_p=1、ignore_eos=true、无stop；相同canonical prompt与
tokenizer/特殊token规则，内容最多759token，最终上下文合法。原生实际数量
必须逐请求等于target。Prime/vLLM取token_ids，S-LoRA取SSE整数token.id；
文本重分词和expected数量不能替代实际计数。

重试属于原request，不新增offered分母，不重置到达时间，额外计算/持卡保留。
跨后端不要求逐token相同；同后端输出变化须审计。500逻辑ID不等于500种
独立训练权重；静态权重SHA与数值可区分性分别披露，未通过的数值身份验证
不能因名称正确而宣称通过。

对中止运行保留原计划分母；未到达不伪作timeout。只报告观察到的资源 U_obs
与条件时延，不进入完整达标排名。系统失败、外部干扰、启动/测量错误分别
分类；不按胜负剔除。修复后新attempt保留原记录。不得补跑成功数洗掉失败。
零正确完成的per-correct成本/资源为N/A，不为0；未执行为not_run，不为0。

## 4. 请求时间线及分位数

同一可靠时钟中：a=计划到达，e=实际提交，d=后端dispatch，f/l=首/末token，
c=完成通知；n=原生输出token数。全部内部计算以秒和未舍入数值进行。

\[
TTFT_r=f_r-a_r,\quad
TPOT_r=(l_r-f_r)/(n_r-1)\quad(n_r\ge2),\quad E2E_r=c_r-a_r.
\]

\[
E2E_r=(e_r-a_r)+(d_r-e_r)+(f_r-d_r)+(l_r-f_r)+(c_r-l_r).
\]

Dispatch Wait若定义为d−a，不再加一次e−a。service TTFT=f−d另报，不能冒充
用户TTFT。提交迟到、客户端排队不扣除。单token的TPOT为N/A，不补0。
跨主机未校准时钟不直接相减；服务器与客户端重叠span不累计伪作总延迟。
同可靠时钟的分解及TPOT重算误差上限1ms；不能添加事后SLO容差。

每轮在正确完成且有可靠该指标的集合中计算时延，并公开样本数与失败数。
缺失必需计量使正式计量不合格，不能只删除缺字段的成功请求。
Type-1分位数：排序x_(1)…x_(m)，Q_p=x_(ceil(pm))，0<p≤1；空集N/A。
不把所有块请求合并后称为“平均运行P95”。partial的低尾延迟不能说明完整服务好。

## 5. 共同 SLO 标定及判定

每模型使用最终同版本vLLM、共同资源包络、真正GPU-ready adapter、固定prefix
cache；按已有输入长度四分位分组，重复边界合并。不用主表排队延迟作为warm基准。
默认batch8；若不可行，在所有共同参考条件中统一选择最大可行batch≤8并冻结。
每组每轮256请求、三轮，batch间不累积前轮排队；TPOT参考至少2输出token。

对模型μ、输入组κ，各轮先计算请求算术均值，再对三轮均值等权平均：

\[
T^{warm}_{\mu\kappa}=\tfrac13\sum_j \overline{TTFT}_{\mu\kappa j},\qquad
P^{warm}_{\mu\kappa}=\tfrac13\sum_j \overline{TPOT}_{\mu\kappa j}.
\]

\[
\theta_r^T=5T^{warm}_{\mu\kappa(r)},\qquad
\theta_r^P=2P^{warm}_{\mu\kappa(r)}.
\]

5/2倍率参考HydraServe；共同后端、长度分组、样本数及95%联合要求是本项目
协议，不称行业标准或与其全部设置完全一致。已有变长工作负载不强改为1024。
当前开发5000ms并非最终标定阈值。标定文件保存原始样本、单位、组边界、
样本数、执行身份和SHA；对所有系统共享，不能用每系统自身慢参考放松阈值。

先判失败：z_r=0时I_r=0；其余按下式：

\[
I_r=\begin{cases}
\mathbf1[TTFT_r\le\theta_r^T],&n_r=1,\\
\mathbf1[TTFT_r\le\theta_r^T]\mathbf1[TPOT_r\le\theta_r^P],&n_r\ge2.
\end{cases}
\]

\[
A_T=N^{-1}\sum_r z_r\mathbf1[TTFT_r\le\theta_r^T],\qquad
A_{joint}=N^{-1}\sum_r I_r,\quad N_{good}=\sum_r I_r.
\]

A_P^cond在正确完成且n≥2集合计算，明确条件分母，同时列失败数和联合值。
联合达成不是两个边际的平均、乘积或最小值。全部4000正确且至少3800联合
达标才满足主可行性；3799/4000即便显示95.0%也不达标。

SLO敏感性保持既定S4–S13/§10.2规定：90/95/99%要求及阈值倍率；纯离线
改评价可复用日志，影响控制或重新选工作点则是新执行键，不能改标签冒充重跑。

## 6. 生命周期 GPU 占用、成本与 goodput

\[
U=\sum_d\int a_d(t)dt,\qquad R=U/N_{correct}.
\]

a_d是服务实际持有物理GPU的指示，不是utilization、逻辑副本数或显存比例。
多进程同卡按时间区间并集计一次，TP多卡逐卡计；CPU-only准备不虚构GPU时间。
计时覆盖共同部署准备至实际释放，后台常驻仍持卡继续计费。
使用真实allocation/release证据；请求完成、engine.shutdown或一次idle采样
不自动证明释放。缺release保持右删失，不补造时间。

互斥分项：业务到达前、共同到达窗口、窗口结束至全部终态、终态至资源释放；
终态边界不早于到达窗口末，四段之和必须等于U。启动/加载/生成可重叠的活动
只能另画时间线，不直接相加。10s持卡中的6s重叠准备仍是10GPU-s。

GPU-only美元成本：C=pU/3600，per-correct=C/N_correct，p单位$/GPU-hour。
统一单价缩放不改变资源排名；idle折扣不能冒称物理GPU释放。
CPU/HOST/远端另报，不称上述成本为完整云账单。

HydraServe的GPU memory-time可作补充，但不是本机整卡持有的主替代：
以固定整卡保留容量M计，J=M×U；按实际已用显存计一般不等价。
除非证明可执行的GPU共享和资源归属，否则半显存不等于半张卡成本。

完整运行的T_obs=max(共同末次计划到达,最后请求终态)−业务开始：

\[
Throughput=N_{correct}/T_{obs},\quad
Goodput=N_{good}/T_{obs},\quad
TokenThroughput=\sum_{z_r=1}n_r/T_{obs}.
\]

\[
GoodRequestsPerDollar=N_{good}/C,\qquad
GoodRequestsPerGPUSecond=N_{good}/U.
\]

观察窗口含drain，不含终态后的清理；清理仍计入U/C。失败生成token另报为
消耗，不算有效吞吐。offered rate=N/T_arrival不冒充完成吞吐或最大容量。
不能以goodput除以总成本声称requests/$；正确关系需相同窗口的成本速率。
上述分母为0时N/A并标原因，不生成∞或0的伪优值。

## 7. 工作点选择与“最优”的范围

仅开发验证期间选择参数；每模型固定工作点跨场景和敏感性复用。
公开候选范围和合理优化机会，不只优化Prime。正式结果反馈开发后，冻结新
版本完整评估所有规定块，保留旧结果，不筛测试块或无限补跑至显著。

### G1 / M1

可行配置在全部验证场景W0/W1/W2、全部预登记验证块均N_correct=N且
A_joint≥0.95。目标是三个场景等权的平均R，不按请求时间长短改变权重。
平局按参考归一化P95 TTFT，再按稳定配置ID；选择结果命名resource_opt。

无可行点时：有全部正确点则最大化最差块SLO，再最小化GPU-s/offered；
否则最大化最差完成比例、再最差SLO、再GPU-s/offered及稳定ID。
这些均标diagnostic_nonfeasible；无合法路径标qualification_failure。
不得把未达标系统的条件成本当“达标最低资源”。

### G2 / M2

Resident-vLLM共同后端/包络/准备协议、四卡常驻，每模型每场景三次完整正确
验证，定义U_ref为三次完整U均值，B=0.75U_ref。不能短trace线性外推。
G2配置须满足正确性、U≤B、A_joint≥0.95，以及共同四卡峰值和CPU/HOST上限。
W1/W2等权最小化参考归一化P95 TTFT，命名budget_latency_opt。

归一化的固定分母明确取上述同模型同场景Resident验证的运行级P95 TTFT三轮
均值Q_ref；必须有限且>0。每轮q=Q95/Q_ref。它只用于跨场景等权选择，
绝对P95仍为主表数值，不能用各系统自己的分母。M1平局也用同一参考。
这是对计划“参考归一化”的具体化，不改变G1/G2目标；参考未完成不得填值。

共同预算为上限，不要求实际消费恰好相等；报告U、U/B及平均持卡数。
不拼接M1成本与M2延迟。未合格点完整展示，不填0SLO或∞TTFT。
正式测试任一块不满足约束时保留该块并标未持续达标，不删除后继续称全部达标。
结论限定为已测配置、硬件、场景和合同，不能称理论全局最优。

## 8. 重复、区间、多重比较与非劣

source_trace_seed=42；41开发、42 smoke、43–47首批正式块。块号不是新真实
trace；同一冻结回放的重复只反映运行变异。主比较5块、支持消融3块按计划，
自然A4扩充规则单独遵循原计划。系统顺序预先平衡，状态按共同协议重置。

每轮先求指标Y_j，再求均值和95% t-CI：

\[
\bar Y=J^{-1}\sum_jY_j,\quad
CI=\bar Y\pm t_{0.975,J-1}s_Y/\sqrt J.
\]

低优配对差Δ_j=Y_baseline,j−Y_Prime,j；高优反向；实际输入、区组和条件
真正配对才使用配对区间。CI跨0为未分辨，不代表相同；J<2不出t-CI。
小样本t解释依赖运行独立和差值分布近似；保存原始点，不把4000请求当4000重复。
百分比改善先每对计算再汇总；reference=0则N/A。SLO差用百分点，不混相对百分比。

预登记主要比较族按模型×工作点目标×指定场景/汇总范围定义，所有该范围内
Prime对基线的优越性检验均入同族，K在看正式结果前由资格登记确定；不可行项
单独报告，不以看见结果后删项缩小家族。Holm双侧p排序p_(1)…p_(K)，依序
与0.05/(K−i+1)比较，首次不通过后停止拒绝。未经校正的95%CI不称联合95%保证。
支持“下降”还须实际差值方向正确。对不同模型/场景的全局“都更好”主张另
预登记覆盖全部相关比较的家族，不能跨家族拼凑全局显著性。

G2非劣检验用D_j=A_joint,Prime,j−A_joint,baseline,j，容忍度0。
报告单侧95%下界mean(D)−t_(0.95,J−1)s_D/√J；下界≥0才支持该比较的非劣，
多基线同时主张仍进行预登记家族校正。方差为0的小样本仅作观测性判定，
不能据此保证未来无退化。此判据不是删除运行的门槛。
“共同95%达标”不自动等于“相对98%的基线非劣”，两种主张分别给证据。

## 9. 机制证据如何解释主指标

- readiness快照必须在reserve后、resolve前；不能下载后改写成事先命中。
- initial、natural_scaleout、controlled互斥；first-service绑定epoch第一条
  实际dispatch，失败不换下一条成功请求补齐。引擎ready、请求TTFT和
  activation-to-first-token三种时间分开。
- 用共同trace预定义cold/churn/burst请求集合进行跨系统比较；按每系统自身
  路由形成的GPU-hit/cold集合仅为条件诊断，不冒充共同请求因果对照。
- 报fetch数量、线上/解包字节、各层驻留与使用、准备完成但未使用、状态年龄、
  false-ready、admission延后及后续成本。命中率增加不自动意味着G1/G2提升。
- native token事件与SSE chunk间隔不同；不能从多token chunk反推单token停顿。
- LastKnown只改变信息新鲜度，其他可行性和资源合同保持不变；无触发不归因。

## 10. CE、计费敏感性与历史复用

保留旧CE的L定义和单位：CE=1/(L×C)，L按既有平均E2E协议；不偷换为TTFT。
ln(CE_P/CE_B)=ln(L_B/L_P)+ln(C_B/C_P)。广义CE采用共同参考归一化，
α,β∈{0.5,1,2}，idle因子{0,0.238095,0.5,0.75,1}；成本与CE break-even
分别计算。cost/token使用实际正确输出/输入计数，定义分母及单位1M。
零或缺失不能用于对数/除法。CE仅辅助解释，不用于主配置选择。

R0同合同字段全可直接复用；R1执行正确仅统计错误可离线重算；R2合同不同
只复用资产/诊断；R3执行错误或关键事实缺失则重跑受影响键。旧local-sim、
旧无限制运行、旧不匹配输出不混入真实远端主表。Full主表/消融相同条件的
展示引用同一canonical run-set，独立条件重测必须另标，不手工对齐数字。

## 11. 每次系统对比的强制核查

1. 完整读取计划、执行状态、本文件；记录协议ID/SHA。
2. 确认运行是开发、资格还是正式；正式必须具备冻结warm阈值/预算/工作点。
3. 比对trace、subset、prompt、target、generation、远端交付、初态、资源合同。
4. 检查全部offered身份、正确完成、失败/重试、原生token与可靠时间线。
5. 检查GPU真实分配/释放、重叠积分、内存事件、外部干扰和收尾。
6. 先生成完整状态表；任何不完整不能进入达标资源/尾延迟排名。
7. 逐run计算，再配对、CI和预登记校正；所有失败/退化/无触发保留。
8. 图表均保存输入SHA、run-set、脚本版本；Serverless命名；IEEE单栏/TNR规则。
9. 发布前再次核对本文件；记录与协议不符项，不使用兜底缺省值通过验收。

freeze manifest必须包含metric_protocol_sha256、slo_runtime_hash、
slo_eval_hash、warm_reference_sha、resource_reference_sha、generation_hash、
execution_key与analysis_key。纯价格/显示等离线变化只改analysis_key；
SLO在线控制/timeout或工作点变化改execution_key。修订另建V2并说明影响，
不原地改V1、不追改旧运行身份；同一正式系列不得中途换口径。

超时遵循计划：资格/验证1800秒；正式按所有参测系统合法初始化/验证证据
冻结H_μ=max(600,2T_init,2T_val)，每请求Δ_timeout=max(H_μ,
10[θ_T+(n_max−1)θ_P])。缺证据不填0，SLO违约不立即取消；正式不按胜负修改。

## 12. 自适应优化不得改变评价尺子

除论文行间公式/核心思想，允许基于实际队列、完成事件、KV/slot、footprint、
传输字节/耗时、生命周期和压力优化底层执行。先提出可证伪假设，对照历史和
同类原始论文/官方实现，最小验证后完整回放，再接受或撤销。
避免无依据magic number，不用另一批自由系数包装“自适应”。物理上限、
安全阈值、60秒/30秒/95%/75%等公开实验常量继续明确保留。
不同模型可以有验证后冻结的特定配置；不得看正式结果换公式/阈值/分母。

本协议承诺可核验和公平比较，不承诺Prime必然领先。观察领先、统计支持、
正确实现和实验完成是四个不同状态，分别记录。
