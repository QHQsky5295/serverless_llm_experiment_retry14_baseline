# D588：完整W1之后的存储整理

2026-10-16。D587 Loquetier W1已经完成、审计并推送到
`8b072ac01022194efaf7a064ea4714f39f46d95a`。本轮未运行新性能实验、未改变服务
代码或评价规则；先恢复下一次同配置W2的磁盘余量。

## 实际回收

| 操作 | 验证与保留 | 回收字节 |
|---|---|---:|
| 移除D587私有下载对象 | 500对象/5000文件，当前副本与保留原件全量SHA一致；保留metadata及原始测量 | 13004480512 |
| D551已关闭main_outcome无损gzip | 两次独立解压SHA一致，保留逐文件映射及旧身份 | 382058496 |
| 合计 | 不重复计入D586历史回收 | 13386539008 |

下载删除唯一目标为`/tmp/tc-d587-7bl/remote_artifacts/objects`，载荷12985984450字节。
重新检查GPU/服务真实释放、实际资源域关闭、结果已备份、当前文件大小和inode、
原件SHA、软链接/嵌套挂载、可读进程引用以及后续配置/脚本引用。receipts/journals/
run_contract的保留副本也逐项核验。未删除唯一模型、工件、trace或原始成功/失败记录。
进程检查范围为可读同UID的FD/cwd/root与fuser；三项无读取权限，明确保留可见性
限制，不宣称扫描了整机所有用户的文件描述符。

压缩对象为已关闭并发布的D551 PrimeLoRA W1：
`results/ieee_tc/p2_backend_qualification/d551_20261015/7b_initial_control_full_w1/launch.launch/physical_deployment/main_outcome.json`。
原始及两次解压SHA均为
`4c4f9da6a874019bbad5e43edda4a738b12c02ebfe7fe48194c757f21a414298`，
与已发布result2来源一致。当前以相同路径加`.gz`保留，不修改其观测值或复用身份。
映射位于`paper_results/ieee_tc/storage/20261016_d588_closed_d551_outcome.json`。

整理后可用161569964032字节，约150.47 GiB。下一次完整Loquetier预计新增峰值
20 GiB，仍按max(150 GiB,100 GiB+1.5×新增峰值)重新准入；当前快照不保证未来
空间或内存。不得降低推理机150/100 GiB门槛，不改远端服务或其他项目。

## 资源与证据

- 维护限定3/4 GiB、swap0、辅助CPU组，不与推理并行。
- 下载核验/回收峰值3221749760字节，memory.high事件4547；max/OOM/kill/swap0。
  这是离线维护的回收压力，不能隐去，也不是D587性能运行的资源事件。
- 压缩峰值500035584字节，high/max/OOM/kill/swap0；实际维护域均已关闭。
- 147个受保护投稿文件和三个既有用户修改文件SHA不变。
- 复用既有逐文件核验、关闭、压缩、归档和发布脚本，仅绑定本次明确目标。
  资源技能区分实际限制与可用快照，备份技能检查来源、压缩回读及敏感信息。

下一步按相同Loquetier冻结配置完成W2，不因低SLO新增基线调优。然后补齐必要
7B比较并集中优化PrimeLoRA。3B、dLoRA限定比较、G1/G2、A1–A5、S1–S13、
成本/CE、两轮审稿文档和最终图表继续保留。无新性能或最优性结论。
