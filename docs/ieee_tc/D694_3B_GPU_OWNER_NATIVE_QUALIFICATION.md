# D694：同副本 GPU 准备资格的真实推理检查

2026-10-18。推进D693唯一生产候选，不增加新优化：同一native owner的最终
GPU准备动作依次派发，其他owner及HOST/NVMe仍并行，全局并发仍为3。
完整假设、历史失败及官方源码依据见D693文档。本轮不是完整性能评估。

## 合同与资格

复用D681的100请求W0前缀、既有3B工件、真实远端已发布gzip、生成合同、
vLLM0.30.0、profile及所有非目标参数；只改运行身份/冷缓存路径。D693
712项回归及3项原生方法检查以源码SHA绑定复用，不重复运行或累加。
九公式、原生epoch、物理容量、admission、共同SLO与GPU计量均不变。
父版本为4cda40b9cf9792cb7a15e0e8fbc53d9b1ecd3ad5。

配置：`configs/ieee_tc/20261018_d694_3b_full_smoke100.yaml`。
复用既有D293入口；删除D694分支及两个3B guard项后必须逐字等于父版本，
不改历史分支、不打开额外profiling。工作负载、权重和完整数据均不重新生成。

放行顺序为闭合日志存储维护、参数/来源核验、明确路径备份、实时资源检查、
远端健康、100请求原生推理、释放资源、统一审计。全部100请求须原生token
数匹配，无fallback，dispatch及准备状态可追溯，实际worker正确限制并释放。
独立数值正确性不能由HTTP成功或token数量代替，仍为UNKNOWN，n_correct=null。

## 安全与边界

复用共同72/80GiB服务包络与3/4GiB辅助包络，不与构建、压缩或另一推理并行。
保留150GiB磁盘和102GiB启动内存门槛，147个历史保护文件及用户manifest不变。
D681/D682临时NVMe缓存已为空，未删除唯一数据。本轮仅将已闭合D681的
main_outcome无损gzip归档，双重解压SHA一致，释放25,640,960B；原SHA及新
归档路径见`paper_results/ieee_tc/storage/20261018_d694_closed_outcomes.json`。
不可把此前回收空间再次记为本轮回收。

共享k3s不操作，既有HydraServe隔离适配结果复用。远端只管理本项目拥有的
服务；不得在正式请求中临时打包，也不在测量中调整远端配置。

## 结果与下一步

本轮100请求原生资格已通过。原始记录位于
`results/ieee_tc/p2_backend_qualification/d694_20261018/`，小型证据位于
`paper_results/ieee_tc/p2_backend/d694_3b_full_smoke100/`。
100请求仅判定集成资格，不排名、不外推4000请求SLO或GPU-s；下一登记
一次完整3B W0新执行键，不追加1000请求前缀、不重跑同键选择胜者。
当前3B仍未达到共同95%联合SLO，其余W1/W2/G2、基线、消融、敏感性和交付
完整保留；本轮不画图，也不修改论文。

### 本轮结果

| 检查/观测 | 结果 |
|---|---:|
| 请求及原生生成合同 | 100/100，17,369 tokens |
| 原生owner / GPU准备尝试 | 4 / 162 |
| 同owner最终GPU动作重叠 | 0 |
| 最终movement状态 | 268 completed，无活动订阅 |
| GPU准备获取 / 延后 | 153 / 9 |
| HOST→GPU租约转移核验 | 19/19 |
| dispatch tier完整性 | 100/100：GPU49、HOST19、NVMe3、Remote29 |
| 冻结共同联合SLO | 97/100；TTFT97，TPOT100 |
| 平均 / P95 TTFT | 654.713 / 1603.218 ms |
| 平均 / P95 TPOT | 13.942 / 17.139 ms |
| 平均E2E | 3116.397 ms |
| 物理生命周期GPU-s | 793.585321 |
| 真实远端传输 | 29次，67,350,156 wire bytes，0次请求打包 |
| activation类别 | initial1、natural_scaleout3 |
| 服务内存峰值 | 20,739,088,384 B |
| high/max/OOM/OOMkill/服务swap | 全0 |
| E2E分解最大误差 | 0 ms |

重叠检查来自已保留的`coordination_after_shutdown.ieee_movements`中完整
started_at/finished_at区间，不新增在线探针、不将ready时间代替动作时间。
确认执行互斥，并不证明全部快照过期已消除，更不是完整性能收益。

启动时远端实际链路1000Mbit/s/full，部署后无额外远端调整。launch与watchdog
均exit0；全部本机服务和辅助进程实际释放，本项目远端两服务及监控已停止。
日志中的`engine_dead`目录名来自启用KEEP_DEDICATED_WORKER_LOGS的收尾保留
分支，不单凭目录名推断请求失败；原始日志保留，launch/cleanup及生成审计通过。
旧InfraGPU/折扣成本显示不作为本轮物理积分：采用真实allocation/release的
793.585321 GPU-s；各互斥时间窗求和一致。

首次离线分析误用旧环境，因缺少ijson在导入处失败；收集已成功。保留失败
analysis1.log和回执，仅切换既有vLLM0.30.0环境对同一原始结果离线审计，
不重跑GPU。最终分析通过，实际峰值139,182,080B，事件/swap0；各scope均
递归空关闭。独立数值资格仍未知，n_correct=null，不进入G1/G2排名。

启动前9路径已推送45f4a3d07044e5bafc39d563d899fb31291308c5；本节与完成证据
随后以独立checkpoint发布，不覆盖启动前manifest或D681历史数字。
