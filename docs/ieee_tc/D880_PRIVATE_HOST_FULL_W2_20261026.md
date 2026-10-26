# D880：匹配初始化后的私有CPU构造完整W2

## 完整性能结果（2026-10-26 13:37，13:41完成收尾）

4000/4000终态，原生计量与输入/绑定检查通过；本机GPU/scope和远端两个
交付服务/monitor已释放，launch各退出码0。complete.py已通过D879实际profile
绑定、147保护项及共同SLO/生命周期核算。独立数值正确性仍未知。

| 指标 | D870上一完整版本 | D880本次 |
|---|---:|---:|
| 联合达标请求 | 3773/4000 | 3761/4000 |
| 联合达成率 | 94.325% | 94.025% |
| 生命周期GPU-s | 11931.565094 | 11308.104268 |
| 平均TTFT(ms) | 705.991452 | 720.868700 |
| P95 TTFT(ms) | 2006.444557 | 2050.450553 |
| 平均TPOT(ms) | 15.579477 | 15.547622 |
| 平均E2E(ms) | 2464.077087 | 2482.941439 |

共同W2预算仍12083.063870 GPU-s；D880在预算内但联合差39条达到95%。
不因资源下降忽略SLO退化，不称达标或最优，不计算运行间CI。
full4000.json SHA `f887f158ac501251133d256d39ab87c119f3210eb13a00621d9300e745f995b5`。
delivery/owner/settlement/private-construction与配对阶段比较均已完成。
505次真实传输共1172100920B，ID/字节/SHA匹配、请求中不打包；服务峰值
33156608000B、high/max/OOM/swap0。147保护项不变。

实际私有构造1367个唯一joined完成回执、其中discarded0。请求快照按owner
取最后端点合计started1363/settled1362/failed0/discarded0，最后有1个in-flight；
这是请求期间快照而非shutdown总数。准备journal为completed1367、deferred1727、
file_held2，后两条无receipt原样保留；不得把缺失回执填成成功或失败。
物理GPU最终释放有独立完整证据。实际触发不等于因果性能收益或数值正确性。

相对D870：135条joint fail→pass、147条pass→fail，净少12条；GPU-s少
623.460826（5.2253%），P95 TTFT多44.005996ms（2.1932%）。两个新增GPU
首分配从业务后10.8/12.9秒延至323.4/325.1秒，解释625.220878GPU-s分配
边界差，最终释放则晚1.760052GPU-s；没有中间释放间隙。它不是已证明的
构造直接加速或收尾节省，扩容时机变化的原因还需控制记录解释。
平均arrival→admission多9.253425ms、admission→acquired多6.631615ms，
native prefill少0.457118ms；这些是嵌套分解中的对应项，不混加重叠子阶段。

**候选决定：不晋升为达标Full/默认配置。**保留opt-in代码、全部结果与失败；
单轮不作显著性推断，不补同键重复找胜出。下一步只复用既有日志回答扩容/
等待变化的有界问题，再决定撤销或新候选；不因构造确实执行而继续堆叠优化。
独立数值正确性仍null、G1/G2未完成，所有后续计划范围保留。
备份以raw/backup_completion.json为准，最终V2仍未同步。
以下保留原登记与运行期间记录，不追改历史。

实际启动补记：D879已完成，D880严格绑定检查通过；提交
`badf4bc4012be589d761ece700278bb4b254a80e`已核验备份，12:28观察16/4000
终态且仍推进。此为运行中状态，不是性能验收；同一tmux继续，禁止重启重复。

12:33:52轻量续查：同一运行已259/4000终态；watchdog sample410，
service memory15418335232B、high/max/OOM/swap均0，无warning/abort。
12:33磁盘可用173978484736B。仅观察与记录，不修改运行配置，不将前段
live CE或5000ms计数用于共同SLO排名；尚未生成完整结果或执行收尾分析。

12:42仍为同一存活tmux，777/4000；已观察三个活动副本。
watchdog sample891，service19526647808B、high/max/OOM/swap均0。
收尾准备仅在raw目录复用D870程序：close_delivery.py SHA
`5b6d5e90747755060e78ac7ebf43131fd9f1bce739bda027a4db947dbd8fee20`；
native_and_lifecycle.py SHA
`66508ca54b08e5dea245c0af7881aa61d6e22ab04a11c89cb342dd597494f509`。
后者在原统计上补实际private-construction证据，不改主指标：按native owner
报告请求期间累计端点、按token去重joined回执；shutdown计数未知则明确未知。
语法检查通过，尚未执行；不得把待分析程序存在当作证据已闭合。

13:00:16同一tmux/pane PID2939759 live，2039/4000终态；watchdog
sample1973，service25941368832B，high/max/OOM/swap0，无warning/abort。
12:59磁盘可用165728940032B。仅持续观察，无代码或远端调整；完整测量、
物理释放及收尾统计仍待结束，不把前半段外推为完整结果。

13:12:57同一tmux仍live，3027/4000；watchdog sample2723，
service31364001792B，high/max/OOM/swap0，无warning/abort。
此前2743条terminal记录流式核查均success且native_contract_matched，
error_types为空；仅过程监控，不是完整正确性/性能验收。
13:11:58磁盘可用161003499520B，略低于150GiB新任务线、仍高于100GiB
当前运行停止线。继续原测量，不并行清理；结束后先核对实际workspace释放
与空间，再执行后续任务，不能沿用旧启动余量。收尾程序尚未执行。

本键只在D879完整实测、严格导出且资源释放后启动。当前是条件性登记，不
以文件存在声称已经启动或完成；以raw/start_receipt、launch和completion为准。

同一D873/D874候选、D875原生资格，继承D876全部Full设置；只将服务/准备
profile和admission长度绑定更新为D879严格匹配身份，并使用独立输出/cache
目录。长度均值来自原完成长度证据，不使用新请求未来生成长度。三实例上限、
共同4000请求、W2 rotation100、prompt/targets、真实remote交付、内存/CPU、
共同SLO和Resident预算全部不改，不引入新参数搜索或短Full。

## 必要核验

复用D726已有运行时解析/严格loader检查：Full runtime和dedicated child的
模型身份必须等于D879 contract；服务及准备profile的SHA和context必须完全
匹配。此为修复D876曾漏掉的绑定检查，不重新测SLO/Resident或整套资格。
配置恢复旧profile和输出路径后必须与D876逐字段一致。D878 collector修复
只影响初始化测量，不作为生产优化收益归因。

## 问题与判定

检验把CPU私有对象构造从core同步事务分离，是否改善完整churn负载的联合
SLO与尾TTFT，及是否增加GPU生命周期成本。不能由65.8ms局部重叠或单次
准备速度推断完整效果。最近完整参照D870为3773/4000=94.325%，
U=11931.565094108693 GPU-s；共同W2预算12083.063870235463 GPU-s。
仍至少3800条联合达标，不改阈值、不补同键重复直到获胜。

结果保留原生token/adapter引用、CPU构造started/settled/failed/discarded、
真实获取次数和字节、完整allocation/release、取消与资源事件；原计算脚本
复用。独立数值正确性未证明时N_correct保持未知，不将绑定成功写作全部
数值证明，也不由一条W2点宣称双模型跨场景G1/G2全部完成。无CI、暂不画图。

raw：`results/ieee_tc/p2_backend_qualification/d880_20261026`；curated：
`paper_results/ieee_tc/p2_backend/d880_private_host_full_w2`。
旧147投稿项、用户manifest和D876–D879失败/测量全部保留。启动前明确文件
备份可回退，最终V2同步仍待完成。两模型同版本矩阵、其余baseline/dLoRA、
A1–A5/S1–S13、两轮审稿图文、数值证据和最终交付继续开放。
