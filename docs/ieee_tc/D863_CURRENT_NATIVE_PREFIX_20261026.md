# D863：当前3B短输出停顿的有界原生诊断

2026-10-26，已完成一次诊断并封存分析。沿用已批准主线P3，非新性能候选。

## 完成结果与下一主线

唯一实际GPU诊断为`d863_20261026_attempt2/3b_native_prefix1000`。
1000/1000请求成功，原生token、绑定和计时检查通过；独立数值正确性仍未知，
`n_correct=null`、`ranking_eligible=false`。全部三个native进程关闭完整，
1000/1000首末token窗口覆盖完整；没有扩大64MiB/200000span观察上限。
服务、回放、监控均正常退出，实际GPU释放，远端本次三服务全部inactive。

冻结D648阈值下：TTFT981、TPOT969、联合952/1000。该95.2%只描述预登记
前缀，不能替代D858完整4000的3746/4000=93.65%，不能宣告G1/G2达标。
本次TTFT均值623.977872ms、P951557.236173ms，TPOT均值16.384826ms，
实际GPU时间3289.653067秒；不与无观察器Full作性能胜负比较或线性外推。

31个TPOT违约均不超过64token，其中30个不超过32token。首末token窗口
分解如下（每组算术均值，ms；不是跨运行重复或反事实可消除时间）：

| 组 | 请求数 | 解码窗口 | 控制调用 | input非控制 | engine step | 其余 |
|---|---:|---:|---:|---:|---:|---:|
| ≤32token，TPOT通过 | 282 | 207.166 | 32.771 | 1.251 | 168.047 | 5.097 |
| ≤32token，TPOT违约 | 30 | 428.468 | 214.971 | 27.923 | 176.207 | 9.367 |
| 全部TPOT违约 | 31 | 476.960 | 238.358 | 28.490 | 200.640 | 9.472 |

违约组的控制交叠进一步分解：按需加载/取得引用66.909ms（控制交叠的28.07%）、
source snapshot56.088ms（23.53%）、HOST准备50.765ms（21.30%）、pending
GPU reference39.825ms（16.71%）；这四类合计89.61%。routing snapshot只有
6.001ms（2.52%），因此不能仅凭全进程snapshot耗时就再次尝试已否定的路由
投影。31条窗口中的scheduler子阶段仅1.729ms；多token分配step交叠28.251ms，
单token分配step171.288ms。分配形状不是GPU kernel阶段，不据此直接改token
预算或断言某阶段全部可删除。

下一只沿这四类实际控制入口核对共同工作、既有修改历史和官方后端执行路径，
寻找一个有语义依据的候选；不重做D851–D854 whole-owner线程、D844/D856
ctypes-stat、D860投影或已完成HOST融合。尚未选定/实施新的生产优化。
诊断未证明“所有剩余违约都来自控制”，也不把TPOT-only优化等同联合SLO达标。

146次真实远端交付均published、双端transfer ID/归档SHA/字节一致，合计
338845661线上字节，请求内打包0。初始分析按当前允许deferred的语义保留
pending；随后确认本次实际全部published，复用D723逐项证明，在独立
`delivery_closure.json`追加关闭，未改原pending记录或忽略取消。

服务峰值20921794560B，high/max/OOM/swap均0；离线分析峰值约1.04GiB，
无内存事件。147历史保护项和用户manifest零变化，工作缓存为空。
产物在`paper_results/ieee_tc/p2_backend/d863_3b_native_prefix1000/completion/`，
原生逐step大表留原始目录，由SHA引用；curated包约0.73MiB，不入库大日志。

分析复用D723的计量/窗口分析、D722的迭代分解和原始交付核验；未改观察器、
阈值、生产策略或论文公式。新增四个raw编排脚本`analyze.py`、
`collect_and_analyze.py`、`audit_iterations.py`、`finalize.py`均保留snapshot，
并用现有整数区间交叠函数将client分类相加核对至原client总量。
未增加GPU回放，未生成图。以下保留原登记与首次入口失败的完整说明。

## 首次启动入口错误与原样前缀适配

16项定向检查通过、e8e36dcf已远端备份后，第一次启动在preflight创建服务
资源域之前失败：W2入口原先只支持4000条完整源，拒绝任何diagnostic prefix。
没有模型加载、请求测量或launch.json；不是性能失败。原launcher.log、备份
和首次回执保留，tmux创建成功不等于服务运行成功。核对GPU空闲、该辅助域
已inactive且无launch证据后，以原InvocationID停止两个远端服务及监控；
stop_prelaunch_failure回执显示全部inactive/MainPID0。

修正为数据投影语义，而非绕过映射验证：始终对原4000条源与完整冻结W2 map
逐条核验，再从所得完整W2 entries取原序前缀。未执行尾部的错误映射也必须
拒绝；不重新生成rank/arrival/prompt/adapter，不修改原始文件。完整4000
路径输出不变，formal/full身份仍拒绝短前缀。服务和外置publisher使用同一
view SHA及完整source/map SHA。preflight允许这种明确非formal Unix前缀，
保留资源控制、完整source_count4000和严格prefix身份核对。

新增测试覆盖精确前缀、未执行尾部损坏、正式身份拒绝、service/publisher一致，
并复用现有完整W2/W0/W1/coverage测试。新执行使用d863_20261026_attempt2，
不覆盖原launch失败目录；这是协议错误修复后的同一诊断，不是同键择优。
涉及workload_generator和preflight的诊断投影支持；生产调度、数值执行、
core_timing及九公式仍未改。下表原登记路径之外新增attempt2配置和管理快照。

## 问题和已有证据

D862已经复用D858/D861完整4000请求，解释了首次TTFT跨在线扩容线与
GPU占用的闭环差异；D860已撤回，生产逐字节恢复D857。当前开发参照D858
联合3746/4000=93.65%，距共同95%仍差54条。D861全部93个TPOT违约均为
输出不超过64token；其TPOT-only组的同副本prefill包络重叠反而更少。
这些证据不支持直接降低token预算，也不能把剩余违约全部归于远端网络。

当前尚缺：在实际首末token窗口内，原生控制调用、engine step和其余时间
各占多少。D722复用D684原生1000请求，有312160个span、89915个step；但
早于D725图范围与D747/D751 HOST融合。D723是旧版100请求组件诊断；D843
是当前controller栈，不含所需原生逐迭代分解。三者不能替代当前测量。

本次不重复D844/D856 ctypes-stat、D851–D854 whole-owner线程、D860文件
投影、D712/D713阈值或已经完成的HOST融合方向。没有新的性能参数候选。

## 一次诊断与证伪边界

- 使用D858模型/运行配置，仅改output/NVMe/HOST三个独占路径。
- 原4000请求源、原W2 adapter map、原prompt/target/arrival均不生成新数据；
  执行既定前1000条，保留完整源身份和diagnostic_prefix_v1身份。
- max3、max-seqs8、token预算4096、graph范围、IEEE九公式、profile、
  共同SLO、真实远端已发布gzip和isal解码全部不变。
- 只启用已有host_spans_v1。不开python_frames、v2_components、HOST inventory
  子计时；不修改观察器，不添加同步、CUDA事件或读回。
- 每原生进程维持64MiB/200000span上限，记录观察开销/截断/时钟/进程身份。
  若截断，报告覆盖范围，不扩上限或重跑同键来补齐有利样本。
- 以现有分析器交叠原生首末token窗口，分控制client、单/多token分配step、
  input非client和未覆盖时间。嵌套时间不重复相加，异常和无覆盖窗口保留。
- 若控制调用占主要部分，才沿实际utility边界查重复工作；若step占主要部分，
  先查已记录分配形状及本地源码；若未覆盖部分主导，保持归因未知。三者均
  不预先决定改批处理，不把未覆盖时间直接命名为输出交付或GPU计算。

只执行一次；1000条不代表4000条的后期工作集，也不是M1/M2。观察开销留在
时延中，不与无观察器Full作性能优胜比较，不构造去掉组件后的反事实TTFT。
独立数值LoRA证明仍未知，n_correct=null、ranking_eligible=false。

## 官方实现依据

[vLLM 0.30 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
的busy-loop先处理输入再执行step；step包含输出入队，异步batch还可能回收
前一batch，故主机span不是纯GPU kernel计时。它支持观察边界，不证明本机
瓶颈。[官方profiling说明](https://docs.vllm.ai/en/v0.30.0/contributing/profiling/)
明确观察会干扰性能；本次采用已有有界轻量span而非完整Torch trace，并保持
诊断/正式测量身份分开。

## 变更、资源和收尾

| 文件 | 改动 |
|---|---|
| configs/ieee_tc/20261026_d863_3b_native_prefix1000.yaml | D858三路径克隆 |
| scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh | 登记prefix、原W2与decoder入口 |
| scripts/ieee_tc/profile_d328_controller_python.sh | 登记原生span观察tag |
| tests/test_ieee_tc_d456_diagnostic_contract.py | 三项D863接口检查 |
| raw d863_20261026管理脚本与prelaunch快照 | 复用既有启动/备份/远端释放过程 |
| 生产算法、core_timing、原生后端 | 零修改 |

CPU资格/备份3/4GiB、swap0、CPU2/3/26/27；服务72/80GiB与既定外置回放和
watchdog，磁盘150/100GiB及新增峰值检查不变。不操作共享k3s，不并行维护。
147历史项和用户manifest保护；只stage明确文件，无凭据/原始大日志入库。
先检查备份再启动，结束后实际释放GPU、按本次InvocationID停止远端服务，
再分析，不开启重复诊断或画图。

完整同版本两模型W0/W1/W2、baseline/dLoRA、A1–A5、S1–S13、数值证明、
审稿图文和最终V2同步均保留。下一步由该诊断的实际覆盖和分解决定唯一候选。
