# D1130：累积handoff消融的完整请求观测与收尾失败

## 结果及复用边界

Llama-2-7B / W1 / RoutingHandoff唯一运行完成4000请求，但CPU准备进程收尾
失败，原始launch保留`pass=false`。没有成功结果导出，不把恢复文件冒充原始
成功结果。通过保存的`completed_scenario_windows`恢复请求观测，并用原D299
原生绑定、时间线、冻结SLO和物理区间积分逻辑核查；未知数值正确性仍未知。

| 项目 | 恢复的观测 |
|---|---:|
| 计划/完成/原生数量及绑定匹配 | 4000/4000/4000 |
| 联合TTFT/TPOT达成 | 3929/4000，98.225% |
| TTFT边际达成 | 3967/4000 |
| TPOT边际达成 | 3959/4000 |
| 完整物理GPU占用 | 7372.557966 GPU-s |
| GPU-s/offered-request | 1.843139492 |
| 平均/P95 TTFT | 785.018156 / 1880.049470 ms |
| 平均/P95 TPOT | 34.362467 / 56.408508 ms |
| 平均E2E | 4571.519570 ms |
| 真实分配及释放 | 19/19 |
| 远端传输/线上字节 | 132 / 131480060 |

TTFT/TPOT/E2E原生重算误差均0。2282资源样本的high/max/OOM/kill/swap均0。
完整物理区间已保留，不能把CPU收尾失败误写成资源计量缺失；同样不能把
请求完成误写成整套服务正常收尾。该点在主记录中保留R3执行失败身份，
恢复观测只用于诊断，不进入成功执行或最终排名。不按指标优劣决定是否重做。

源outcome SHA：`068788766da16b8821ae1ddbd2635c65eb7548fd9d572e68ff6af266a3387e6e`。
数据与来源：`paper_results/ieee_tc/ablation/d1130_7b_routing_handoff_w1/recovery1/`。
原生n_correct=null，不把GPU-s/offered改写成已证明的per-correct成本。
没有运行间CI，没有独立留出主张，也没有从计划数推导handoff物理收益。

## 根因与修正

实际Conda Python未提供`os.pidfd_open`和`signal.pidfd_send_signal`。旧D468/D469
的CPU辅助进程路径直接调用它们；已有物理GPU生命周期路径则早已使用Linux
UAPI绑定。此次unused CPU helper仍存活时，旧接口缺口在最后收尾被触发。

复用同一个已验证pidfd_open实现，移入不导入模型的轻量模块；CPU辅助进程和
物理GPU计量共享它。补充相同Linux ABI的pidfd_send_signal，保留birth/UID/
会话/资源域校验。采用已由D541验证的poll退出事件，避免select的FD编号限制。
没有按PID盲杀、吞掉未知所有权、伪造release或延长SLO/超时；不改IEEE九公式、
路由、控制器、预算和任何消融非目标逻辑。

依据：[Linux6.8系统调用表](https://github.com/torvalds/linux/blob/v6.8/arch/x86/entry/syscalls/syscall_64.tbl)、
[pidfd_send_signal规范](https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html)，
并核对本机`asm/unistd_64.h`。这属于实际兼容性缺陷，不需从同类系统论文
另造算法解释；前述D27/D541和D468/D469历史没有被绕过。

## 验证、资源与后续

- 实际部署Python的四项真实进程测试通过：缺Python包装接口、FD≥2048、
  外来owner拒绝、错误FD保留内核错误。无GPU加载。
- 92项CPU生命周期/工厂/物理计量回归中91项直接通过；旧高FD测试在默认
  soft FD限制1024下失败，单独使用测试域LimitNOFILE8192后通过。不改变服务
  资源配置、不重复已通过的大组。新增恢复读取器身份/完整人口检查通过。
- 两次测试夹具/启动限制失败及一次SHA抄写错误保留在recovery_notes中；
  它们不算额外性能运行，也不隐去失败。
- 仅删除本轮已退出服务的两个私有可重建缓存。逐文件与保留原池核验SHA，
  记录inode、挂载/链接/进程引用、配置引用后删除；保留唯一工件和全部原始
  测量。根分区实测回收3305160704B，可用163876327424B；HOST释放不计入该值。
- 当前没有GPU实验，远端专属服务inactive/MainPID0，共享Kubernetes未改。

下一动作是备份本次失败证据和兼容性修正，然后仅对受影响RoutingHandoff
登记一个修复后的独立执行键；不是重跑Full/基线，也不是为寻找更好排序重复。
已完成的7B十一主消融和RoutingOnly封存不动。修复后正常完成该累积节点，
再做A2/A4同计划时序、S2位置/信息、S3争用、预算/规模及剩余敏感性。3B缺项、
CE/Pareto、审稿文档、IEEE图表和V2完整交付仍保留，不关闭总目标。
