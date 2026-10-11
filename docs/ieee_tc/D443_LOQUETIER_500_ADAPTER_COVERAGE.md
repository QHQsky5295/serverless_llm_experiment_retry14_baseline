# D443：Loquetier 完整 500-ID 功能覆盖（预登记）

日期：2026-10-12。D442 原100请求已完成；本次不是其重复。

## 问题与范围

D442 覆盖29个请求adapter，已验证原生计量、静态目标、完整池准备和实际释放；
但不能据此证明全部500个adapter都能被请求并进入原生forward。因此按完整计划
补齐500-ID功能覆盖，不因前缀联合SLO为0而改变基线配置或开展性能搜索。

复用 D433 已有 `adapter_coverage_v1` 输入索引视图：源seed42前500请求的
arrival/prompt/token保持原样，按既有规则逐一绑定冻结池500个逻辑ID。
视图SHA为 `286f7547a8129fd6e75885379428b84c9a5c23a7eb02bf1d7546f49bcf2cee5d`；
目标总计68,925原生输出token。此视图只作功能验证，不作W0主比较、预算参考
或独立训练模型多样性证明。500逻辑ID的重复/零权重限制仍须披露。

## 不变条件

- D441 native config SHA `d9f3b81d3d90340739ec66b45d36c1c71c706640dd2cd790a19d3ea4a80a617a`。
- D437静态四分片、每份125-ID，FP16/batch2/context1024、顺序启动，原生SMLM。
- 公共notice后完整500-ID远程准备，notice+60秒到达；不移到ready后。
- D78/D80已发布只读gzip复用；不重新打包、发布、生成权重或完整负载。
- 原生Python3.9/公共Python3.12、内核、依赖、runtime和analyzer均不改变。
- service72/80GiB、swap2GiB、CPU4–23,28–47；aux3/4GiB、swap0、CPU2,3,26,27。
- 磁盘150/100GiB、预估新增20GiB、现有启动门槛/watchdog/tmux不降低。

唯一raw目录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d443_20261012`。
私有可重建缓存：`/tmp/tc-d443-7bl`。沿用D442启动/远端/封存脚本，
只修改campaign与既有500-ID视图参数。无新的实验框架。

## 验收与后续

500/500终态、全部500-ID实际原生绑定、68,925原生token、可靠计时/内容来源、
四分片实际服务和物理GPU释放须通过。已有D394共同阈值只作本视图诊断；
SLO不达标不掩盖、不调优此基线。无静默fallback，失败保留，不盲目重跑。
原D385数值参考是原100请求，不能套用于此adapter重映射；原始数值未知标记
保留，不把结构绑定等同完整语义证明。通过后按原1000→单次4000W0推进。

启动前53项MAIN证据测试和18项BASE汇总/HTTP测试均通过；D442封存578文件
SHA、147旧保护条目、用户修改及两原生checkout补丁均未变化。执行前双仓库
备份与fresh安全/远端健康检查仍必须完成。此预登记不代表GPU覆盖已通过。

## 实际结果（03:09:04正常结束）

本次同配置运行500/500请求原生计量/来源/绑定通过，全部500-ID实际服务，
68925原生token，0失败，计时重算误差0ms。四分片实际GPU释放，远端三个
自有服务停止。联合SLO0/500，TPOT达标387/500；不是W0主性能结论。
U3247.515872713411GPU-s，6.4950317454268225/绑定有效请求。
TTFT均值218590.293ms/P95375885.992ms；TPOT均值62.948ms/P9568.556ms。
500远程获取/420740410线上B/12985984450物化B，无请求打包。
服务峰值42149494784B，high/max/OOM/swap均0。原始数值未知标记保留，
未对重映射500使用原100参考，不宣称独立完整数值证明或基线最优。

详见`paper_results/ieee_tc/loquetier/d443_500_coverage_summary.json`；MAIN
`docs/ieee_tc/D443_LOQUETIER_500_ADAPTER_COVERAGE.md`记录分解、限制和完整来源。
执行MAIN702672d48712b64df62284174b1de417f22dba4c、
BASE9e0ad55d7f051a031ee4099d59d1a4718e64a59f，不以本结果文档提交替换执行身份。
不调优/重复本覆盖，下一步同配置原1000兼容，再单次4000W0。
