# D406：ServerlessLLM 7B 全500逻辑工件执行检查通过

2026-10-11。本次仅补齐 D405 确认缺失的功能覆盖；不作为 W0 性能比较、
独立数值正确性证明或最优配置。沿用冻结指标 V1＋单轮 V2，不计算 CI。

执行 main `6eed716bed1ed994ab1d35e7a00c6ecf0469e4d7`，baseline
`7a37e1f9b3220b26f9f0166e36e2821e84aa1d4b`。原始数据：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/serverless_qualification/d406_20261011`。

## 实验问题、合同与结果

原 trace 前500条的 request ID、prompt、到达、输出目标不变；使用明确的
adapter_coverage_v1 索引，将静态集合的500个逻辑 ID 各执行一次。既有
工件、权重及完整 trace 文件均未重新生成。派生行保存原 row SHA/adapter，
与 W0 有不同 view SHA/profile。并发使用公开 target4，其他设置与 D404
相同（min1/max4/keep10、max_num_seqs4/max_loras4、原生快速加载后端）。

| 指标 | 本次观测 |
|---|---:|
| 计划／到达／提交／终态／原生合同通过 | 各500 |
| 失败／静默 fallback | 0／未发现 |
| 实际执行的不同逻辑 adapter | 500 |
| 原生输出 token | 68,925，逐请求等于目标 |
| 实际提供服务的后端／ready 回执 | 4／4 |
| 原生计时、E2E／TPOT 重算最大误差 | 0 ms |
| 真实远端交付／请求期打包 | 500／0 |
| 线上传输字节 | 420,740,410 B |
| 验真后的本地物化字节 | 12,985,984,450 B |
| 本轮 cache-hit 请求 | 0（每个 ID 只访问一次） |
| 实际物理 GPU 占用 | 2,717.3239195765927 GPU-s |
| 平均／Type-1 P95 TTFT | 5,452.221／43,617.366 ms |
| 平均路由队列／service TTFT | 4,672.127／725.348 ms |
| 平均 E2E | 14,336.943 ms |

资源互斥分项：到达前187.756990、到达窗口2442.469864、drain63.985562、
清理23.111504 GPU-s；原始精度之和与物理并集一致，所有 lease 已关闭。
服务 peak64,123,109,376 B，swap0，high/max/OOM/OOM-kill 均0；主机最低
可用内存59,220,758,528 B，运行期最低磁盘149,497,323,520 B。低于150GiB
只禁止下一轮重型启动，不是本轮100GiB停止门槛，未更改安全阈值。

实际 scheduler 回读为已冻结修复 SHA、正确 CPU/cgroup，循环存活、无异常、
分配队列为空。原生轮转、扩缩容、placement、加载和迁移算法未替换。
500份请求／工件 receipt／远端日志逐一关联，所有5000个物化文件按静态
内容清单核验；没有用健康检查、文件可下载或 HTTP200 单独代替原生执行。

## 能够与不能够得出的结论

能够说明：在本机原生 ServerlessLLM 路径下，既有500个逻辑工件都能完成
指定请求合同；之前“实际仅触达132个”的覆盖缺口已补齐，不需要重跑500检查。

不能把 D406 与 D404 的延迟下降全部归因于 target4：请求群体、adapter映射
和输出分布不同。不能与 PrimeLoRA D371 的4000请求直接排名，不能声称公共
工作点最优，也不从500请求推算4000资源或共同预算。

`n_correct=null`、`independent_lora_numerical_correctness=false` 等原始未知
字段保持原样。内容SHA、原生绑定与 token 数检查不自动成为独立 PEFT 数值
正确性证明；500逻辑 ID 也不等于500独立训练模型。相关既有数值资格应先审计
复用，缺失部分补齐后才允许进入完整系统正确性主张。

## 分析、保护与回收

复用 `scripts/ieee_tc/audit_d402_serverlessllm_path.py`，增加显式
`--population adapter_coverage_v1`，重建派生输入、全集合及原生输出身份。
新增6个CPU拒绝/接受测试通过（0.036秒，4GiB/无swap）；原D404默认人口
离线复核也通过，全部数值与封存结果完全一致，没有新增性能重复。

本轮 audit：`paper_results/ieee_tc/baseline_audit/d406_7b_coverage/audit.json`，
SHA `24da017f3612e2e15c8533b03d30ffa3ad612c8b52e6dfa3a7bd2c445cb419d7`。
同目录 evidence_manifest 保留完整小文件来源，不把大原始日志提交 Git。

实际 GPU 已释放，overlay 恢复原始字节；三个本轮远端服务均已停止，端口
不再监听。远端传输/监控日志复制后，两端 SHA 一致。旧147项保护验证前后
完全相同（唯一历史差异仍是用户自己的 replay 修改），没有覆盖投稿结果。

仅回收 `/tmp/tc-d406-7br/remote_artifacts/objects`：已完成字节核验，确认
服务退出、无已观察到的打开引用、无软链或嵌套挂载、未来配置/脚本未引用，
保留原始工件池及所有 receipt/journal。`lsof` 对其他系统/容器挂载存在权限
警告，未声称做了完整特权级全机扫描；补查同UID的FD/cwd/root与fuser。
非强制删除后回收观察值13,003,907,072 B，可用空间162,353,836,032 B。
回执在 raw 的 cache_cleanup_before/after.json。以后重验缓存字节需要重新
物化；不因此抹掉已经保存的完整SHA证据，不为重验而自动重新下载整池。

下一主线：审计可复用的数值资格，原1000请求兼容／公开配置验证，冻结后
完整4000 W0，与已封存 PrimeLoRA 7B在同一指标口径下比较；不重跑 D404、
D406 或旧轮询对照，不提前进入3B或图表工作。
