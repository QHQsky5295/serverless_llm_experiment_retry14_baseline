# D442：Loquetier 原生证据校验与首次 100 请求资格

日期：2026-10-12。沿用 D437–D441 的固定配置，不做基线性能搜索。

## 原因与修改边界

D441 已接入实际原生 worker，但已有离线汇总只认识 vLLM 与 S-LoRA。
为避免将外来字段缺省为“正确”，增加显式 `loquetier` 分支：核验原生
request/adapter ID、绝对工件路径、整数分片内索引、真实 forward 次数和终态。
不增加假的 vLLM 字段，不把数值正确性未知改写成通过。
MAIN 的现有审计入口增加 `loquetier_static_shards_v1`：按 D437 静态归属而非
RR 验证目标；核验来源、ready、完整 500-ID 准备、实际 GPU 进程与释放。
共用物理并集/准备计量原语，明确区分 Loquetier 的单进程和 S-LoRA 的 API
父进程/原生子进程，不制造不存在的父子关系。

已读取完整计划、执行状态、V1/V2、最新基线保真要求、D432/435/438–441
日志和代码；重新联网核查 [原论文](https://arxiv.org/abs/2511.00101)。保留
官方虚拟模块、ModelGeneration、MixedLoraModel/SMLM；实际源码仍为
`aae33baeeb19777129c1ccbff99a898d4a0e2c63` 加三个历史兼容补丁。
本次仅修正分析身份，没有修改原生计算、分片、batch、精度或后端。

## 预登记实验

- 唯一原始目录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d442_20261012`。
- 私有可重建下载目录：`/tmp/tc-d442-7bl`；不复制到保留证据中。
- 复用 D432 的启动、远端管理、冻结与证据保留脚本，仅改 campaign/原生身份。
- 原 seed42 的前 100 请求，29 个请求 adapter、17,369 目标 token；不重抽样。
  仍准备全部 500-ID，四份各 125；不能缩池迎合请求集合。
- 保留 D441 native config：FP16、batch2、context1024、静态目标、顺序启动。
  原生 Python3.9，公共服务 Python3.12；原生内核与依赖不再变更。
- ingress8510/router8511/workers8523,8533,8543,8553。
- 真实远端 D78/D80 已发布只读 gzip；共同 notice 后进行本次完整池传输，
  notice+60 秒开始业务，不平移到 ready，不在请求中打包。
- 复用既有 gated launcher/tmux/watchdog；service72/80GiB、swap2GiB、
  CPU4–23,28–47；aux3/4GiB、swap0、CPU2,3,26,27。新增峰值估计20GiB；
  推理机150/100GiB与102GiB空闲启动检查不降低。
- 先原生计量/绑定/真实远端/实际释放，再用已有 D385 独立参考作有界数值
  对照。保留差异，不要求跨后端逐 token 相同，不宣布完整数值证明。
- D394 共同 SLO 只作该前缀诊断；不是完整 W0、G1/G2 排名或最优工作点。
- 实际失败保留；不盲目重试、不改 batch/缩池优化基线、不重复 D438 容量。

执行前完成 CPU 回归、旧结果保护、源码/秘密检查与双仓库备份。
本次 CPU 结果与运行状态见 MAIN `EXECUTION_STATUS.md` 及后续独占产物；
此文的预登记内容不构成“GPU 实验已通过”。通过后仍须500-ID覆盖、原1000
兼容和一次4000 W0；剩余完整矩阵不取消。
