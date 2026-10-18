# D695：同副本 GPU 准备资格的完整3B W0评估

2026-10-18。D693是唯一生产候选：按native owner派发最终GPU准备动作，
同owner最多一个，其他owner及HOST/NVMe仍并行，全局并发3不变。假设、
历史日志/代码和官方vLLM依据见D693文档。本轮不新增优化或参数搜索。

D694已完成100请求真实远端原生资格：100/100生成与绑定、4owner的162次
最终GPU动作重叠0，HOST交接19次；共同联合SLO97/100。该短资格不能替代
完整性能证据，故直接扩到4000，不再追加1000前缀或局部剖析。

## 冻结合同及单轮身份

- 主仓库父checkpoint为`0a8a458a050f42762d6ff3e246e382b2343b3deb`，baseline
  仓库保持`e16af33aaf41f02a62ecdce59f6f3227813244c3`，不改对手。
- 配置逐字等于D682全量配置，仅改D695输出/NVMe/HOST目录。与D694相比
  只有请求数100→4000和路径变化，保持同一生产候选，不按完整结果调阈值。
- vLLM0.30.0、FP16/TP1、8GPU LoRA/32CPU LoRA、rank16、8seq/4096batch、
  chunked prefill、无prefix cache、context1024、输出cap256、prompt759。
- 500个既有逻辑adapter、原seed42 W0 trace、D645 service/preparation profile、
  initial observed-arrivals、confirmed routing、handoff、hierarchy/admission原样。
  IEEE九公式/核心语义和现有HOST租约交接不变。
- 源trace SHA `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`；
  全量view SHA `0d998f48a006d638dda9b8d714f667fe3a87f4d2c88044ce19a816b308494a37`。
  不新增权重/trace，不复制完整工件池，不使用未来请求信息。
- 真实HTTP使用D78/D80通知前发布gzip，运行中不打包、维护或扫描远端。
  真实传输、必要读取、物化和请求竞争计入延迟，无模拟等待。
- V1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`，
  单轮V2 SHA `193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d`；
  D648共同SLO和D657资源参考不变。既有已知trace上的开发观察，不伪装独立留出。

## 预先规定的分析

1. 全4000原生合同、458224tokens，无fallback；dispatch与E2E分解完整。
2. 已保留movement日志验证每owner最终GPU动作不重叠、全部订阅/物理引用释放；
   保留GPU准备成功/延后、源确认/epoch/租约和所有旧机制触发证据。
3. 共同联合SLO至少3800/4000为时延可行条件；边际TTFT/TPOT、平均/P95/P99、
   E2E、阶段分解及8个500请求组完整报告。不给单项TPOT退化另设否决阈值。
4. 物理GPU-s来自真实allocation/release，含准备、业务、drain、释放，互斥
   分项之和等于总量。不用idle折扣或旧InfraGPU终端显示替代物理积分。
5. 与父候选D682（3694/92.35%，U15896.253912）及D672（3647/91.175%，
   U14931.314880）等完整历史点并列；D688退化91.125%及已撤回身份继续保留。
   不只挑慢对照，不重复同键求赢家，不计算运行间CI或显著性。
6. 独立数值正确性仍未知时保持n_correct=null，不因原生生成与绑定自动升级
   正式排名。G1不要求75%预算，G2另需W1/W2及共同预算，不由本W0宣告完成。
7. 完整结果决定候选去留；若未达标，先以本轮证据判断具体瓶颈，不堆叠补丁、
   放宽评价尺子或继续重复短前缀。所有失败和退化永久保留。

## 安全、存储和进度

复用D293 runner、D694资格/管理、D682完整流程。服务72/80GiB、swap2GiB；
辅助3/4GiB、swap0、CPU分离；一次一个重型，watchdog与真实整树释放保持。
推理机150GiB启动/100GiB安全停止、102GiB内存启动门槛不变，实时核验。
共享k3s不操作；HydraServe已有隔离适配/实际结果复用，不重启基础适配。

本轮存储仅无损压缩已闭合D684诊断main_outcome，原SHA
`d2235bf4c1c17f7ee2960f126cee898805f0538caa69d2683cfa992738dfcc39`与两次
解压一致，回收259760128B。新映射见
`paper_results/ieee_tc/storage/20261018_d695_closed_outcomes.json`；原审计与
结果身份不改。147个保护文件及用户manifest不变；已回收空间不重复计数。

当前启动前准备，尚无本轮性能结果。完成后按实际资源释放→远端停止→离线
审计→封存/备份推进。原始目录`results/ieee_tc/p2_backend_qualification/d695_20261018/`，
汇总`paper_results/ieee_tc/p2_backend/d695_3b_full_w0/`，不覆盖旧结果、不画图。
其余模型/场景/G2、3B基线/dLoRA、独立数值证据、A1–A5、S1–S13、成本/CE、
审稿文档与最终交付完整保留，整体目标ACTIVE。

### 运行进度（2026-10-18 20:10，本机时间）

9个明确路径已推送`9cc0baa65a9f977f9619b5867b14baec60ee5cdf`，远端一致。
存储维护、装配、备份与启动辅助域均已实际递归空关闭，事件/swap0。
唯一`tc-d695-3b-w0`已启动；当前实际service unit为
`primelora-tc-svc-86f8c15b493c42fa917022a61ac5ab9c.scope`，InvocationID
`100ee05a6fbf4e6f8c84298f3ccbe54b`。启动时真实链路1000Mb/s/full。
最近核验actual service仍active，18/4000请求终态，已出现3个自然扩容实例；
watchdog样本117无警告/中止，service peak20295401472B、事件/swap0。
当前只是运行中观测，不是结果、通过或候选接受。

收尾管理脚本已准备；离线审计复用D694（扩为4000和8组），新增保留日志的
同owner动作区间检查；`run_analysis.py`复用D682明确要求的vLLM0.30.0分析
环境，以免再误用缺ijson环境。均尚未执行，须等待实际服务与GPU释放。
运行期间不压缩、清理、构建、推送或改远端，状态更新在下一正常checkpoint备份。

### 持续监测（2026-10-18 20:28，本机时间）

同一service/InvocationID与tmux身份持续存活，终态进度1072/4000；这是运行中
观察，不是完整性能/正确性通过。watchdog样本1184：主机可用93145358336B、
服务峰值20885311488B，high/max/OOM/oom_kill/swap均0，无warning/abort。
可用磁盘159486517248B，低于150GiB新任务线但高于100GiB运行停止线；不在
性能回放中压缩或清理。没有重启、第二次运行、配置改变、远端管理或推送。
继续等待同一4000回放和真实释放；收尾后才分析共同SLO与物理GPU时间。

## 完整结果、候选裁决与收尾 — 2026-10-18 21:18

本轮4,000/4,000请求终态；原生合同审计通过，458,224个原生token，GPU-owner
动作4,343次且同owner重叠0，HOST lease transfer 1,422次，远端132次传输、
306,360,162线上字节。服务、launcher、3B/7B artifact服务和监控均在资源
释放后按实际InvocationID关闭；远端传输与监控文件SHA已从远端复核。收尾及
离线分析scope均无high/max/OOM/oom_kill/swap事件。

冻结口径下的完整观测为：

| 指标 | D695 当前候选 | D682 父候选 | 差异（当前−父） |
|---|---:|---:|---:|
| 联合达成 | 3618/4000 (90.450%) | 3694/4000 (92.350%) | −76 请求 (−1.900 个百分点) |
| 平均 TTFT | 844.735 ms | 804.280 ms | +40.456 ms |
| P95 TTFT | 2177.395 ms | 1992.432 ms | +184.963 ms |
| 平均 TPOT | 17.450 ms | 16.778 ms | +0.672 ms |
| 平均 E2E | 2823.288 ms | 2723.519 ms | +99.768 ms |
| 生命周期 GPU | 15904.504 GPU-s | 15896.254 GPU-s | +8.250 GPU-s |

D695 因联合 SLO 更低、TTFT/E2E 更高且 GPU-s 未减少，不能作为 PrimeLoRA
性能改进接受；D682 也仍未达到95%共同SLO，故本结果不构成G1完成。`n_correct`
仍按协议保持未知，不因HTTP成功或token数量正确自动升级数值正确性，也不
进入正式排名。所有D695原始日志、失败/退化证据和分析回执永久保留。

阶段证据显示回归主要集中在 admission→engine dispatch（236.451 ms，D682
为228.645 ms）、runtime slot wait（251.838 ms，D682为245.914 ms）和 native
decode（1785.235 ms，D682为1747.424 ms）；因此“同owner串行化消除快照竞争”
虽通过CPU/资格测试，却没有在完整工作负载上转化为资源或SLO收益，不能继续
叠加未经验证的修补。

基于该预先规定的完整回放裁决，已将生产 `OwnedMovementQueue._pump` 恢复到
D693父版本（文件SHA `825ead7f23686b8f501dd8d833cfa27111cc55fbbfda1191bed6a25b1f48654f`）。
D693候选、D695运行及其测试/文档证据仍保留；恢复不是删除失败结果。回退后
相关CPU回归测试187项全部通过（native vLLM 0.30.0环境；仅有既有依赖警告）。

下一步不是重复D695同键，而是对D682/D677/D695的阶段证据做离线归因，登记
新的、可证伪的PrimeLoRA瓶颈假设；在没有新机制假设前不启动下一轮GPU回放。
7B、3B W1/W2/G2、外部baseline、消融、S1–S13、成本/CE及独立数值正确性仍
保持ACTIVE，且本轮未生成论文图表。
