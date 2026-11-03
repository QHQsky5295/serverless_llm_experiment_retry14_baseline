# D1029：可达 active-LoRA 执行图的完整 3B W2 验证

2026-11-03。测前登记：D1028 画像已结束，不重跑、不追加短 Full。
本文件承接当前主线，不覆盖用户 dirty EXECUTION_STATUS.md。完整目标 active。

## 科学问题、已有证据与唯一候选

D1027 完整 4000 请求原生合同完成，联合 SLO 3784/4000（94.60%），物理
生命周期 8316.260703600012 GPU-s，P95 TTFT 1970.868099015 ms；差16条
达到冻结要求。其初始编译缓存与 D1022 不同，不能将改善全部归因文件校验。
D1027 的6个worker仍重复捕获88/32个图描述，编译已热的扩容初始化34–37秒。

D1028 根据实际 vLLM 0.30.0 V2 scheduler 的 max_num_seqs=8，保留32槽、
token轴、0及dummy=33，只裁掉不可能出现的中间16/32活跃LoRA图分支。
完整推导、历史否决方向和官方源码依据见 D1028_REACHABLE_LORA_CAPTURE_20261103.md。
30项相关CPU检查与一次423请求原生画像已经完成；实测服务图66/24，32槽
不变，KV空闲block增140。该画像不证明主工作负载收益或独立数值正确性。

唯一待检验假设：这一实现及其必要重新测量的初始化画像，能否在完整W2
改善联合SLO、尾延迟与GPU生命周期？不能把它写成纯启动单因素消融，不能
离线扣启动时间，不调整扩缩容阈值、并发或加载策略，不重开文件校验线程扫描。

## 冻结执行与比较合同

配置：configs/ieee_tc/20261103_d1029_3b_reachable_lora_w2.yaml；SHA
8a4dbd1945b04fcf4a8829c16e3b83930df2851454be9b19f930351e28c33b65。
父配置为D1027，结构比较仅14处差异：3个独占输出/缓存路径、1个capture
策略、2个completed-length身份、两份新画像各4项路径/SHA/上下文身份。
没有其他字段变化。32槽、并发8、max_num_seqs8、FP16/TP1、1–3实例、
D747 descriptor、stdlib gzip、真实已发布remote、全部IEEE公式与控制保持。

D1028 manifest SHA：83a9c988443ee005ca6586a030a50ea61cee136b797e6cdf111a34c6da2be12f。
service SHA：5d31393cbdfc4e6b32b652e776a0dfb9f076d66539bbdd1e8ab493058e278ef8。
preparation SHA：fda596df99900379bcfeba8bf91346f14b16b72a8bd394d49d93862dc40bbf20。
使用实际 dedicated subprocess 配置解析和原有严格loader，CPU加载验证160个
service类与96个preparation类通过；启动用fastsafetensors策略仍明确记录。
没有移除画像身份约束或使用缺类兜底。

输入复用seed42的4000请求、500逻辑adapter、W2 rotation100；arrival、prompt、
target和fixed_length_greedy_v1不变。60秒共同部署窗口，准备与收尾持卡均计量。
source trace SHA：4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d。
W2 map SHA：28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23。
这是已知负载的开发候选，不重命名为独立留出数据。

PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
V1 SHA 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22；
单轮V2 SHA 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。
D648 SLO不变：prompt≤506/>506的TTFT分别1466.050801102/2539.648668191ms，
TPOT分别29.921872367/36.215515193ms。G1要求全部正确且联合≥3800，再比较
物理GPU-s；G2另要求U≤12083.063870，再比较W2 P95及本次观测SLO差。
当前独立数值正确性未闭合，n_correct=null，ranking_eligible=false，不能由
binding/token或画像通过补造。没有CI、重复同键、挑阶段或按胜负删点。

## 测前检查、入口与保护

14处结构差异和严格画像加载在独立CPU2/3/26/27、3/4GiB、swap0中通过，
scope为primelora-tc-aux-d1029c0f00004f07989c33a15f8480001.scope，退出0。
未保存退出前全部资源计数，不能声称这项CPU检查全部资源事件为0。
147项旧投稿保护verify通过，回执raw/protected_before.json SHA为
5a88ff8a6eb65454973364d55797e9d5445c66b2d6b589fe532224faffe7b2f7。
用户manifest SHA 3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df，
EXECUTION_STATUS SHA 41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374不变。

10:08远端认证health通过，PID1984520、session-8529.scope及clock
remote-process-monotonic:b7a04b62f38c4ae6adabb4da5c0fd8dc未变；两端1000/full。
远端可用132651040768B。无服务重启、重新打包或共享k3s操作，仍保留无连续
远端CPU监测的限制。已有只读发布缓存直接复用。

复用scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh，文件名
含W1但实际固定W2。FAASLORA_D1000_RUN_ID=d1029_20261103，RUN_TAG为
d1029_3b_reachable_lora_w2_full，CONFIG为本配置。显式unset三个详细TC timing、
D1000_PREFIX_COUNT、D1000_NATIVE_PROFILE及TC_DIAGNOSTIC_PREFIX_COUNT。
descriptor library与SHA沿D1027；不设prefix、不重新构建库或升级后端。

raw独占父目录：/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1029_20261103。
实际run子目录d1029_3b_reachable_lora_w2_full由原runner mkdir独占创建。
外层tmux；service72/80GiB、swap2、CPU4–23/28–47；aux3/4GiB、swap0、
CPU2/3/26/27及独立watchdog。启动内存≥102GiB、磁盘≥150GiB并覆盖32GiB
预测增长；运行底线16GiB内存/100GiB磁盘。一次一个重型任务，测量中不维护、
备份、改源码/参数或管理远端。测前代码/配置备份后启动同一完整键。

## 终态动作与未完成范围

先核对实际release，再复用D703分析器恢复完整SLO、阶段、画像绑定、原生
adapter/token与双端传输。必须显式指定本配置和D1028 manifest。核对每次
实际worker的66/24及capture receipt，保留编译冷/热状态、KV差异和全部失败。
达标后立即推进同版本W0/W1和完整外部比较，不追加小规模调参。未达标只
保留反例、根据新证据决定下一候选，不自动重跑。

D810外部baseline18点及适配局限继续复用，不重启HydraServe/Loquetier适配。
W2 vLLM为100%联合、16110.752GPU-s、P95TTFT441.635ms，ServerlessLLM
为98.125%、16084.201、480.724ms；它们超预算但延迟更好，不隐藏。
全部未完范围保留：两模型同版本W0/W1/W2与G1/G2、独立数值正确性、A1–A5、
S1–S13、CE/Pareto、B1/B2/C1/C2/C3及两轮审稿文档、后期图表和最终V2集成。
PrimeLoRA达标前不绘图；本候选不代表论文完整实验已完成。

## 已启动：同一完整键持续运行，不重复启动

10:10:43外层tmux `tc-d1029-full`启动。测前检查点
dd9c271c24790d604037dca9e9f8af19a7d5d51f已推送并ls-remote核验：
origin/retry14_continuous_queue_v2及faaslora_origin的
backup/d1029-reachable-lora-w2-entry-20261103。后者仅回退备份，不声称历史
分歧V2已集成，无force push。仅3份指定配置/文档暂存，diff、shell语法、
4项来源SHA、暂存凭据模式检查通过；沿用D1028已完成30项测试和原生画像。

启动前本机可用磁盘161285812224B，MemAvailable110048704KiB，GPU compute
为空，无其他重型实验/维护。实际service为
primelora-tc-svc-a56f81ef3c1d4d3892106e9a293f4b3e.scope，InvocationID
5ecb3409e0194b9b8a5e857befd009d8；aux为
primelora-tc-aux-cb4f4f2b7af6412eadac3bdc2f5c096d.scope，watchdog PID3728703。
主runner PID3728701，初始原生EngineCore PID3730298；已核对实际进程仍在
同一受限资源域。replay_ready的count/source_count=4000，view SHA为
5558de43ede0e2cf49c7478dcfb94640e7a8516406f53766fabfdf524bf91174；无prefix。
原生worker环境中三项TC详细timing及prefix/诊断变量均未启用。

初始worker日志/tmp/faaslora_worker_k01e1lja/worker.log已显示实际66/24捕获。
此次初始torch.compile为31.08秒，明确保存新AOT缓存；init engine为74.21秒，
其中包含编译。与D1027初始编译0.27秒的热态不同，不可把早段排队差全部归给
图裁减。KV为70592tokens。保留整段启动/准备资源与请求时延，不重启、不删
早段、不通过再跑热缓存替换本次。后续按全部worker逐项保存缓存/捕获证据。

10:14附近最新Live为arrived66/4000、done65、ok65、fail0，3个runtime，
1条处理中。只是执行进度，不是共同联合SLO或独立数值正确性结论。
watchdog样本182无warning/abort，high/max/OOM/kill/swap为0，服务峰值
17744351232B，主机可用96834506752B，磁盘160352641024B；低于下一重任务
150GiB启动线不等于触发本轮100GiB运行停止线，不中途维护或调门槛。
当前仍在同一完整回放，尚无终态、物理总GPU-s或最终SLO，不启动另一候选。
本后置交接随下个正常检查点提交，测量期间不额外推送。

### 10:22继续同一回放：首500请求已完成

本目标回合为verified wait。完整重读目标与PLAN，按monitor-experiment技能
直接核对原service InvocationID、tmux、主runner和watchdog实际live，再作
6次间隔约一分钟的轻量观察。没有重启、重跑、新候选、参数/源码修改、
GPU任务、扫描压缩、推送或远端操作。EXECUTION_STATUS仍为受保护用户修改。

10:22:29最新Live：arrived517/4000、done516、ok516、fail0，两个runtime，
1条处理中；完整计划分母仍为4000。首500请求完成不代表联合SLO或完整
正确性合格，日志旧5000ms显示不用于最终判定。service InvocationID仍为
5ecb3409e0194b9b8a5e857befd009d8，runner PID3728701和watchdog3728703在运行。
watchdog样本695无warning/abort，high/max/OOM/kill/swap为0，服务峰值
16.525714874GiB，主机available92.351348877GiB，磁盘146.965610504GiB。
磁盘高于100GiB运行底线；本轮结束释放后再核查150GiB新任务启动线，不在
测量中维护。仍未取得最终SLO/GPU积分，下一动作继续原运行至完整终态，
随后释放/核验/分析；此前所有未完实验、正确性及论文交付范围保持不变。

### 10:30同一回放已超过四分之一

本目标回合仍为verified wait：完整重读目标/PLAN与交接，先确认同一真实
service、runner/watchdog，再作6次约一分钟间隔的轻量观察；不启动新任务。
10:30:01 Live为arrived1016、done1015、ok1015、fail0，完整分母4000，两个
runtime；1条处理中。InvocationID仍为5ecb3409e0194b9b8a5e857befd009d8，
runner3728701/watchdog3728703仍live，原tmux存在。尚无最终结果，不根据
中间5000ms显示判断共同联合SLO，不计算/推造完整GPU积分。
样本1140无warning/abort，high/max/OOM/kill/swap为0，服务峰值
18.736724854GiB，主机available90.980960846GiB，磁盘145.129653931GiB。
无重启、配置/源码修改、重跑、维护、推送或远端管理；两个用户dirty文件
SHA未变。继续原运行至终态，再进行完整闭合与对照；所有后续范围不删减。

### 10:38持续运行与一次自然扩容的机制证据

继续同一监测任务，重读目标、monitor-experiment技能和当前交接；本上下文
已完整读取的PLAN SHA仍为d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
本回合核对实际service/进程后作8次约一分钟间隔的观察；无重启或新实验。

一次自然扩容的EngineCore PID3778337实际日志路径为
/tmp/faaslora_worker_5d7jdcmd/worker.log；10:30:11直接加载AOT，编译0.29秒，
内存估计图66/2、实际服务图66/24，10:30:35 init engine为27.57秒。
这是原生日志初始化子阶段，不是完整activation或TTFT；重复tqdm完成行
不重复计时。候选在自然扩容中真实触发，但不据此宣称总SLO/资源已改善。
运行完成后仍需把全部worker绑定allocation并保存日志与SHA，不只选该例。
观测runtime数从2到3再到2，保留自然闭环状态，不人工强制扩缩容。

10:37:57 Live为arrived1538、done1537、ok1537、fail0，完整计划4000，
两个runtime、1条处理中。原InvocationID不变，runner3728701与watchdog
3728703实际live；样本1610无warning/abort，high/max/OOM/kill/swap均0。
服务峰值21.909164429GiB，主机available91.931694031GiB，磁盘142.501834869GiB。
没有生产编辑、分析重任务、压缩、推送或远端管理。仍需继续完整终态及
release后核验，不能用局部运行替代G1/G2、正确性或全目标完成。

## 11:26完整终态：保留取舍结果，未达到共同联合SLO

持续轻量监测同一运行至4000/4000成功、失败0。11:17:59–11:18:00最终两个
实例退出，11:19前service、watchdog、replay均退出0；service资源域消失，
原生GPU context实际释放，独立查询无compute进程。未重复运行同键，未修改
控制/后端、未在测量中维护或操作远端。以下为同一完整4000请求的离线结果。

| 指标 | D1027 | D1029 |
|---|---:|---:|
| 联合TTFT/TPOT达标 | 3784/4000 (94.60%) | 3774/4000 (94.35%) |
| TTFT达标 | 3843 | 3867 |
| TPOT达标 | 3920 | 3901 |
| 物理生命周期GPU-s | 8316.260703600012 | 8167.260452414863 |
| GPU-s/offered-request | 2.079065175900003 | 2.041815113103716 |
| 平均TTFT/ms | 623.7720854831859 | 763.3946093007689 |
| P95 TTFT/ms | 1970.868099015 | 1890.5496881343424 |
| 平均TPOT/ms | 15.610764309 | 15.710652143074844 |
| P95 TPOT/ms | 27.179631824 | 27.332940477539193 |
| 平均E2E/ms | 2342.806521599 | 2493.134081077995 |
| P95 E2E/ms | 4835.258110892 | 4941.7237606830895 |

D1029联合少10条（−0.25百分点），距3800要求差26条；GPU-s减少149.000251185149
（1.7917%），P95 TTFT降低4.0753%，但平均TTFT增加139.622523817583ms。
与D1022相比联合多4条（+0.10百分点），GPU-s增加104.844736687373。
所有比较都是已知固定负载上的单轮描述，不是显著性、纯单因素归因或独立留出。
U虽低于12083.063870预算，联合SLO不满足，不能作为G1/G2达标点或宣称最优。
独立数值正确性仍未闭合，n_correct=null、ranking_eligible=false；上表资源
分母明确是offered而不是伪造correct。旧结果、原始失败及本候选均保留。

### 阶段、初态与机制证据

- TTFT-only违约127、TPOT-only93、两者6。TTFT违约的最大互斥阶段为
  arrival→admission 61条、admission→acquired 72条；不是全部归给GPU推理。
- 同请求对照D1027，138条从联合不达标变达标、148条反向变化；TTFT净多24、
  TPOT净少19。原生load调用1027次；dispatch tier为GPU2964/HOST15/NVMe524/
  Remote497，4000条字段完整。没有把状态命中当作性能归因结论。
- 八个预定500请求phase的联合数依次468/465/480/476/474/472/475/464。
  首500相对D1027少27条，后3500多17条（3306对3289）。这只解释观测分布，
  不移除首phase、不重定义分母，也不虚构“无冷启动”的反事实SLO。
- 首次观测admission前到达27条，全部TTFT违约。D1029初始compile31.08s、
  engine-init74.21s；D1027初始compile0.27s、init32.76s。含编译/捕获的子阶段
  不能与总启动重复相加；这一初态差异禁止纯图裁减因果归因和事后扣时。
- 全部4个真实分配worker已逐PID绑定日志：3730298/3733573/3733669/3778337。
  实际服务图均66/24，内存估计FULL为2；后3个worker的compile为0.27/0.30/
  0.29s，engine-init为27.59/28.81/27.57s。不是只选择最快worker报告。
  32个LoRA槽未减；D1028画像测得KV容量变化仍是本候选条件的一部分。

### 完整性、远端与资源收尾

复用D703分析器，4000条native generation/adapter binding/完整到达和时间线
通过；TTFT/TPOT/E2E重算误差均0。103个service类、53个preparation类严格
匹配本配置与D1028画像；4000条D747 native descriptor库SHA一致。
503次真实传输、1167457984线上字节，双端ID/bytes/SHA全部匹配，无请求期打包。
远端服务未重启；无连续远端CPU/干扰监测的限制继续保留，不能称已排除一切
远端干扰。远端准备分析覆盖497个dispatch Remote、492个可关联请求、491个
不同请求传输；与全服务503次传输的集合口径不同，不强行补齐或删掉其他传输。

3次natural scale-up、2次scale-down，全部4个物理allocation释放。4048个
资源采样，service峰值30440525824B，最低主机可用94329647104B，最低磁盘
144457195520B；high/max/OOM/kill/swap、foreign和escape计数均0。
工作HOST/NVMe缓存为空，147项历史保护SHA验证通过，两个用户dirty文件不变。
收尾分析为CPU2/3/26/27、3/4GiB、swap0、CUDA不可见；没有新增GPU回放。

首次remote-preparation离线调用因绝对/相对stage路径字符串不同而拒绝，
实际CSV SHA仍为275c714ef55f72102a7a18fb2046ffcc9d5c57818cd818dfea7a3c4094f76f5d。
未产生该分析输出、未改变原数据或校验代码；改用原seal记录的相对路径后通过。
这是离线调用错误，不是GPU实验失败，也没有借此重跑性能或覆盖分析产物。

### 产物与下一主线

curated：paper_results/ieee_tc/p2_backend/d1029_3b_reachable_lora_w2/；
completion已封存full4000/stages/paired_d1027/profile/descriptor/delivery/closure，
另存native_startup、paired_d1022及remote_preparation，不改completion manifest。
raw result SHA：8ca69c1d86b6843e87340f04cce3b05c2b68bf1891f0a071826b63c0492c6e7e。
launch SHA：d4e10358fd47bb353d0dc8b2785511f99e7a934cc116147ae418ee62372c908b。
full4000 SHA：895793a7f78a00615953ca41dcedc30fe210806ac63965c17e4010b561039fba。
4份原生启动日志已归档到raw/native_startup_logs.tar.gz，逐成员cmp通过；归档
SHA bd77c7b3af3a10563254227e94efedd80e6c27664ccac320655927a447604244，原日志未删。

本候选完整验证已结束，但没有达到性能目标；不自动重跑暖缓存、不额外加短
Full或绘图。下一步先安全压缩/清理已闭合可恢复证据，恢复新重任务150GiB
启动余量（11:26 df为160382476288B，略低于150GiB）；并沿3B主线核对本次
启动初态和稳态违约，结合已否决历史及官方代码确定一个新的可证伪方向。
当前不能把启动阶段删除、改阈值、改负载或因已知差26条而筛同配置重复。
保留全部后续模型/场景、独立数值正确性、基线比较、A1–A5、S1–S13、CE及
论文设计/审稿文档和最终V2交付范围；不恢复已闭合的HydraServe基础适配。

### 11:32备份与测量后必要存储维护已完成

结果检查点0cdc99fc8a4e153fe4cc4b172bc240949ef257ec已推送并ls-remote核验：
origin/retry14_continuous_queue_v2，以及faaslora_origin的
backup/d1029-reachable-lora-w2-outcome-20261103。21份指定文档/curated文件，
diff、19项整体bundle SHA、10项completion SHA、collector语法与暂存凭据模式
检查通过，用户修改未暂存。FaaSLoRA backup是可回退备份，不冒称分歧V2最终
集成；未force push。本次没有生产机制改动，因此未重复D1028的CPU/GPU资格。

完整比较结束后，复用compress_closed_main_outcomes.py，仅压缩D1022、D1027
两个已closed/released的main_outcome.json；运行结果JSON、当前D1029全部raw、
权重、负载、配置、环境和其他项目不动。压缩前与已有profile-binding seal
核对原SHA，逐文件检查活动引用；两次独立解压SHA与原SHA一致后，才移除
未压缩副本。保留同目录main_outcome.json.gz和compression_receipt.json，
需要原路径时可从归档完整恢复；旧来源引用和旧seal不追改。

回执：paper_results/ieee_tc/storage/d1029_closed_d1022_d1027_outcomes_20261103.json。
实际回收813350912B，available由160377487360B至161190809600B（约150.12GiB）。
这是文件系统余量快照，不保证下次启动仍满足门槛；下次重任务前重新读回。
维护scope为primelora-tc-aux-d1029a1200004f07989c33a15f848012.scope，CPU固定、
3/4GiB、swap0、低I/O优先级；peak144572416B，high/max/OOM/kill/swap均0，
退出0。147项旧保护再次通过。没有新GPU任务或全盘重复扫描。

只读日志还确认：D1028画像初始AOT键为9bc8f69a03d6ac74e3c559d001593a595e94f891440416e2de32251d24ec1834，
D1029 Full为68ad0d2bf035a619bf29b8cb2414f19d9e233505bd689c702446302309f13ff5；
两者分别实际编译30.86/31.08秒。这里只登记待解释的实际差异，不把哈希不同
直接判为实现错误，也没有据此实施预热/缓存替换或另一次回放。后续先对照
历史与官方vLLM配置/缓存身份代码确认其必要性，再结合稳态失约选择唯一候选。
完整目标保持active，当前无运行中的实验；下一步不是重新启动D1029。
