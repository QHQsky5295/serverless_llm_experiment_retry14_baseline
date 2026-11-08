# D1131：7B W1 RoutingHandoff 收尾修复后的唯一执行

2026-11-08，启动前登记；当前性能结果尚不存在。

## 问题、重做原因与边界

继续 PLAN §8 A5 的累积链：ElasticOnly → RoutingOnly → RoutingHandoff →
CapacityOnly → Full。检验在相同服务分桶路由、按需缓存及扩缩容规则下，加入
激活绑定的 handoff 是否减少首次服务准备等待，以及其资源—质量代价。
收益取决于准备及时性及实际使用，不能用计划计数替代服务收益。

D1130 完成全部 4000 请求，但 CPU helper 收尾调用当前 Python 不提供的
pidfd 包装接口，导致 launch.pass=false、没有成功导出。该运行的原始失败、
完整请求观测、物理 GPU 计量和恢复结果全部保留，仍为 R3 诊断，不进入成功
排名。此次重新执行由确定的生命周期错误触发，不由结果排序触发，不从两个
运行中选择更好的数值。D1130 详情及测试见
[恢复记录](D1130_7B_ROUTING_HANDOFF_RECOVERY_20261108.md)。

已修复代码在 `7ce2f815d697218733ee799c8017f27335a089de`，并已推送 V2。
CPU helper 与物理 GPU 计量共享既有 Linux pidfd_open 绑定，补充同 ABI 的
pidfd_send_signal，使用 poll 等待真实退出，保留 UID、birth、session、
cgroup 所有权检查。没有按 PID 兜底终止、伪造释放或改变推理机制。

## 固定的实现及输入

新配置 `configs/ieee_tc/20261108_d1131_7b_routing_handoff_w1.yaml` SHA：
`04e36f3e720e3eb53e262da3a77cc42336819f3273bf20d03df9cfb67be11fbb`。
与 D1130 YAML 解析后比较，仅三个私有输出/缓存路径改变；归一化后完全相等。

- Llama-2-7B、vLLM 0.30.0、FP16、TP=1、min1/max4，14 slots/seqs，
  1024 context/batch、0.92 utilization、prefix=false，D362 profiles。
- 原 D1100 predictive autoscaler、5 秒窗口、27.43356142744625 ms service bin；
  九个 IEEE 行间公式、预算、物理容量检查及监测语义不变。
- `routing_handoff_v1` / `tc_ieee_routing_handoff`，service_bins，
  demand_lru + handoff_only。只允许带 activation_id 的 handoff 准备；
  稳态 hierarchical residency 和软 admission 关闭。Full 及其他消融未改。
- 复用原 source42 的 4000 请求、500 逻辑 adapter、W1 到达映射、60 秒通知，
  fixed_length_greedy_v1 及原生 token 合同；不生成 trace 或工件。
- 真实远端已有只读预发布 gzip，线上不打包、不使用本地 remote fallback；
  远端获取次数及传输字节是机制结果，仍计入服务端到端时间。

D1130 既有 60 项机制/配置 CPU 检查复用。收尾修复的四项真实部署 Python
测试已通过；92 项 CPU 回归中 91 项原组通过，旧高 FD 夹具在测试域调整
LimitNOFILE 后单独通过。恢复读取器检查已通过。没有重复 GPU 资格实验，
没有将测试域限制修改带入正式服务。当前只新增路径相等性检查，已通过。

## 唯一启动及安全条件

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1131_20261108/7b_routing_handoff_w1 \
  d1131_7b_routing_handoff_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1131_20261108/7b_nvme_routing_handoff_w1 \
  /dev/shm/tc-d1131-7b-routing-handoff-w1 \
  configs/ieee_tc/20261108_d1131_7b_routing_handoff_w1.yaml tc_ieee_routing_handoff
```

tmux `tc-d1131-routing-handoff-7b-w1`；只启动一次，不复用失败键。runner 管理
本实验专属远端服务的启动、监控和关闭；不预先手动启动、不修改共享 k3s。
启动前确认新路径不存在、GPU compute 为空、MemAvailable≥102 GiB、磁盘
≥150 GiB、专属远端 inactive/MainPID0。保持服务72/80 GiB、swap2 GiB，
辅助3/4 GiB、swap0，原 watchdog 16 GiB/100 GiB 停止线及32 GiB 增长预估。
运行时不做测试、重分析、压缩、推送或代码修改；持续监测同一个执行句柄。

2026-11-08 17:29 +08 预检：根分区可用163866685440 B，MemAvailable
118945280 KiB，GPU compute 为空，远端 inactive/MainPID0，双方 NIC 1000 Mbps。
正式启动前再读即时资源，不把此快照当未来保证。

## 分析、贡献与后续

正常终态及真实释放后保存同 clock 的远端 journal，复用现有 D1113 collector、
D299 review 和 D455 stage analyzer，显式指定 tc_ieee_routing_handoff。
核查4000输入/生成合同、dispatch tier、生命周期积分、handoff计划与实际
完成/使用、first-service，并确认稳态主动准备和软 admission 未误开启。

直接复用 D1129 RoutingOnly、D1119 ElasticOnly、D1128 CapacityOnly、D1115 Full；
不重跑或挑选对照。自然闭环激活数与准备选集可以不同，其同计划时序因果
问题仍留给受控 A2/A4。边际收益依赖累积顺序，不写成普适独立贡献。

口径沿 V1 + 单轮 V2、D394 SLO、D594 W1预算。展示全部联合达成率、GPU-s/
offered、P95 TTFT；95%不是系统纳入门槛。未知 n_correct 保持 null，不把
原生 token 数量一致当数值正确性证明。首个正常完整结果封存，单轮不报CI；
G1条件资源优势、G2共同预算质量、完整资源—质量关系分开判断。若资源变少
而质量更差，保留取舍，不改成全面最优，不削弱消融或按排序重跑。

随后沿主线完成当前7B受控 A2/A4、S2/S3、M2/S5及其他敏感性；3B缺项、
CE/Pareto、审稿文档、IEEE图表及V2交付均保留。共享 Kubernetes 适配已经
隔离完成，不因本次重启部署。完整 PLAN SHA：
`b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f`。
