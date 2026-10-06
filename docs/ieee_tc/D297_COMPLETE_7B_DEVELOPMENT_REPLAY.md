# D297：退出依赖优化后的完整 7B W0 开发回放

2026-10-07。延续D296的同一候选，不添加算法修改、不调整参数、不重新抽样。
本项补足100请求前缀不能提供的持续负载/热点轮换证据，不是正式留出或G1/G2
结果。执行前完整读取源计划与冻结指标V1，达标前不画图。

## 依据与固定配置

- D296的100请求完成、实际GPU退出和CPU验证已封存于提交300c789；总GPU-s
  727.912111，收尾36.264719。P95用户TTFT10.771356秒，并未改善尾延迟。
- 本次保留其全部model、backend、resource_coordination、preloading配置，
  仅恢复4000请求、Full场景名和新的输出/缓存目录。既有W0 trace和500逻辑
  adapter清单直接复用，无新权重、负载或工件池。
- vLLM0.30/FP16/FLASH_ATTN、TP1、4slots/4seqs、1024context、759/256上限、
  fixed_length_greedy_v1、prefix off、真实预发布远端均不变。
- 原open-loop到达、共同部署通知60秒、请求保护1800秒、自然闭环、完整
  allocation/release计量不变；保留开发期5000ms在线控制，不能当正式共同SLO。
- 之前D219完整回放与当前候选在源码/主动准备资格等方面不同，只是历史参照，
  不能把本轮与它的差异归因于单一D296改动；100请求不能线性外推完整GPU-s。

## 启动前机制语义复核

当前原生主动准备由`_coordination_enabled`选择`capacity_only`，并通过
`_run_ieee_owned_preparation_plan`→`ieee_prepare_host`执行E(t)。旧汇总
`effective_capacity_admission_enabled=false`是旧coordinator/可选请求容量
见证的开关，不等于当前原生主动准备关闭。D226的旧路径解释不能直接套用。
本次不为消除旧标签而改变普通请求的加载优先语义。

D296原始`main_outcome.json`实测：4次activation均policy=full；167个residency
epoch；171个file plan；101个GPU plan全部capacity_only=false；144次GPU
attempt中115完成、29延后。109份receipt明确proactive_admission_evaluated=true，
13份为false（包括复用已在GPU的工件），22份无该标志，不补算成主动决策。
这些是实际触发的证据，不是完整数值正确性或性能收益的证明。

## 验收与安全边界

1. 原始4000 offered集合不删请求；失败、缺失、取消和真实token数全部保存。
2. 核验trace/subset、prompt/target、native token、dispatch tier/slot/owner绑定。
3. 正确性未知仍保持未知；不以HTTP成功、数量相等或近并列解释赋值N_correct。
4. 按V1重算平均/Type-1 P95/P99 TTFT、TPOT、各阶段、物理GPU-s和互斥窗口；
   记录actual mechanism events，不用旧汇总零计数替代原生路径证据。
5. 同一72/80GiB服务、3/4GiB外置域、swap和CPU限制、150/100GiB磁盘门；
   新增磁盘峰值沿用完整回放32GiB保守预算。已封存D219 run约828MiB，缓存
   预算和日志上界另保留，不能只用短回放磁盘量放宽门槛。
6. 必须先通过直连认证health及远端配置核查，再启动本机回放。沿用现有
   paired remote monitor，3B artifact unit只是空闲监控对象，无3B推理。
   测量期间不重启、改配置、打包、清理、压缩或另起压力测试。
7. 只运行此重任务；不启动暖参考、Resident或其他分析重任务并发争用。
8. 完成后释放→验证→无图数据/解释→备份。单次完整开发回放无CI、不宣称最优。

启动：`bash scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh d297`，
通过既有受限runner执行；取消diagnostic prefix，不绕过guard。目录已存在即拒绝。
原始目录：`results/ieee_tc/p2_backend_qualification/d297_20261007/7b_full_w0/`。
所有历史结果、用户manifest和论文源文件不动。

启动前核查：结构化比较确认只改变请求数、场景名、输出/缓存路径；运行代码
SHA仍为`cc821e089b952b66bc10be13e3b8ab11700c9d77dc542c20ca7edb582b1113f8`。
新配置SHA`ecf75d4d6a27ac9c54e3347dfa1089aafa8c159a84676039de86c2216fd012be`。
启动器bash语法通过；系统Python的open-loop与资源保护94项测试通过。
本轮没有改serving代码，D296的634项CPU、34项native环境专项和22项分析器
验证继续作为同一实现证据。尚未开始GPU回放，启动后另记真实状态。

下一步依据本轮完整证据继续7B并完成共同warm/Resident及旧Prime新口径对照；
数值资格与G1/G2未成立前不进入3B、外部基线、消融或正式绘图。

## 实际启动与运行中状态（2026-10-07 01:20+08）

已在提交`a40df0d307c31fa38f11f0b77b72dd16bba10907`推送并核对远端后，
于01:16启动唯一tmux任务`primelora-d297-7b`。上述“尚未开始”描述只对应
预登记检查时刻，不是当前状态。不得因尚无最终结果文件而重复启动。

- 直连认证health结束`814229.57178896`，部署通知`814247.332460961`，
  回放原点`814307.332460961`（均本机monotonic）。此次health先于本机启动。
- 回执确认W0/count4000/source_count4000/rate_scale1，无prefix；trace SHA
  `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`，
  view SHA`a5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d`。
- 服务scope`primelora-tc-svc-72524b343c5c45a699736635a24184bd.scope`，
  invocation`3d9973309a41446baa8df1e8ac763a91`；外置watchdog PID1851665，
  独立aux scope`primelora-tc-aux-7f9be371ba2644b5a885484a29b73f98.scope`。
- 远端7B PID727107/invocation`2ceae044ea1b4290b738ef29f16f179d`；
  3B PID727105/invocation`cee328bfa4654233813c435c85789b96`（无3B推理）；
  `primelora-artifact-monitor-d297.service` PID727110/invocation
  `bec592d608814aa7a9a4512c7d364bff`。按身份停止monitor后，由原有退出逻辑
  停止本次配对服务；不能停止后来另一个invocation。
- 远端监控位置`/home/lab14/primelora_remote/tc/d297_20261007/remote_monitor.log`；
  7B clock`remote-process-monotonic:b7998243a505498db48f39cde67db697`，
  后续仅收取与该clock及本次transfer ID匹配的交付证据。
- 预发布交付方式`prepublished_gzip_v1`和`artifact_timing_v2`不变；NIC1000Mbps/full。
  既有D78/D80一次性交付缓存直接复用，用户本轮再次批准不意味着重新生成。
- 启动前空闲磁盘161,199,501,312bytes（约150.129GiB）、MemAvailable
  110,615,760KiB（约105.49GiB），满足原门槛。01:20监控sample220：
  服务peak18.240692GiB、主机available90.266891GiB、high/max/OOM/swap均0，
  无安全告警；磁盘149.509441GiB，只禁止下一重任务启动，不触发100GiB停止门槛。
- 此刻79请求到达、74正常完成、0报告失败；仍是部分运行，不能称4000完成、
  数值正确、联合SLO达标或优于历史。最终审计和真实释放后另补完整结果。

本轮只作轻量监控和持久记录，不并发运行分析、哈希扫描、压缩、清理或另一个
GPU任务。真实读取/传输/竞争/本地准备继续计入服务时间，不减去实际获取时延。
