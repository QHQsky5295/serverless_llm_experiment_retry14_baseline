# D1081：保持完整状态的正需求候选表示

当前主线：3B A1 Full 的净收益。不是重做 baseline、共享 k3s、正确性资格或
启动窗口扫描。以 D1046/D1047/D1045 为已保留 Full；D1077–D1079 的240秒
NoRouting/路由诊断不升级为主结果。保持60秒主协议及 D648 联合阈值。

## 假设、已有证据与不同于历史的部分

D1064–D1067 显示每轮 owned planning CPU 约210 ms，不能全部归因为背包。
D1066 只消除整个窗口没有需求的轮次；D1068 只处理所有正需求均已 GPU-ready
的轮次。其余轮次仍为静态集合中 h=0 的选项生成双份类字典、候选记录、JSON
及校验遍历，尽管 `generate_ieee_epoch` 从来不把这些记录加入求解器。

新假设：在同一个已验证完整源视图上，直接保存正需求选项，可减少规划表示
开销，而不改变任何候选收益、选择、替换对象或实时资源检查。先用已有 CPU
夹具比较密集/稀疏表示，不为没有可测收益的候选启动 GPU 回放。

这不是 D1067 已否决的 JSON decoder 替换，也不是 D193/D751 图投影、D737
延迟解码、D998 延后观察或 D163/D741 消费者重复解码。文件/native 全量状态、
物理 footprint、零需求但仍占容量的替换对象全部保留。没有 TTL、跨轮缓存、
新的调参量、少取观测或提前取消计划。

## 数学与接口边界

在固定需求窗口中，h=0 => F=h max(0,d_source-d_target)=0。现有代码只对 h>0
构造 PreparationCandidate，因此删除的是未进入求解器的审计表示，而非可选
集合。所有输入选项仍进行类型、重复目标和源类一致性检查；未知正需求成本
继续报错，不填零。完整 source_view 仍供所有 native/file 替换损失计算。

新表示显式标记 positive_demand_v1，记录省略数并绑定 plan SHA。旧密集计划
继续按原验证路径读取；这不是运行失败时的兜底。内部生成选项不作为用户可调
超参数。CPU 资格后才决定是否接入实际 owned worker。历史 hash 不追改。

## 原始资料参照（本轮联网核查）

- [vLLM 0.30.0 Scheduler](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/scheduler.py)
  在当前 running/waiting 集合处理实际工作，并对 num_new_tokens=0 跳过执行。
  借鉴“只物化实际工作”的实现原则，不移植它的调度目标或声称其验证本候选。
- [vLLM SchedulerOutput](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/output.py)
  将执行消息和长期状态分开。本候选保留完整源状态，仅精简无效选项记录。
- [dLoRA 原论文入口](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)
  强调请求与 adapter 协同；不能由这一相关性推定本系统所有控制开销都有收益。

## 有界验证

复用 existing PlanningCPU/OwnedPreparationPlanning 夹具，覆盖 handoff/residency、
替换保护、零需求、混合需求、未知正需求成本、篡改和旧计划兼容。只测试选项
表示的 CPU 组件；扩展到500个逻辑候选的内存夹具不是新工件/负载，也不是
真实服务结果。完整执行目标比较忽略新 plan SHA 的合法传递，不能忽略任何
内容、预算、选中目标或受害对象差异。组件效果不换算成端到端改善。

若等价且收益成立，启用同一个 worker 路径，按原 D1047 服务配置登记一次
3B W1 完整回放；不增加短性能回放，不延长准备窗口，不修改路由、公式、
SLO 或模型配置。无收益则停止该方向。保留所有尝试与失败，不按结果筛选。

仍待 Full 净收益、M2有限预算、7B A1/累积、A2–A5、S1–S13、成本/Pareto、
审稿文档、同版本图表和实际 V2 整合；总体目标保持 active。图表继续延期。

## CPU 组件观测与接入决定

三块交错、每种表示每块五调用全部保存，不做 CI。单位 ms，表内为组件 CPU
均值，不是服务 TTFT；输入源图始终是既有四适配器夹具。

| 模式 | 逻辑候选 ID | 密集表示 | 正需求表示 |
|---|---:|---:|---:|
| handoff | 4 | 0.9014 | 0.5058 |
| handoff | 100 | 3.8875 | 0.6468 |
| handoff | 500 | 36.4939 | 1.2097 |
| residency | 4 | 0.7407 | 0.6869 |
| residency | 100 | 4.1136 | 0.8371 |
| residency | 500 | 19.4184 | 1.4820 |

结果在 `paper_results/ieee_tc/p2_backend/d1081_positive_demand_options/`。
源图、选择和执行目标等价测试通过；接入 owned worker，并在现有 CPU 回执中
记录选项作用域、省略零需求数和物化正需求数。不是把全部210ms CPU都消除，
也没有证据可以把表内节省直接换算成 GPU-s。历史 Full/A1 不覆盖。

资格记录：首次四项新测试通过。接入后的21项针对性测试通过（9项未选），
覆盖旧计划兼容、源图隔离、消费者投影、实际 worker 入口与验证失败传播。
最终测试域峰值620351488 B，high/max/OOM/swap均0。测试域限定3/4GiB、
无GPU；没有修改推理服务72/80GiB。19条现有依赖/pytest配置告警保留。

失败/中止也保留：组件第一次在测量前因未显式 taskset 而拒绝 CPU affinity；
第二次完成内存夹具计算后因资源回执仅接受 UUID scope 名而失败，未保存计时，
不把该次称为成功数据。第三次使用标准 UUID 域、显式 taskset，完整原始值
成功保存。没有按快慢丢弃计时，第二次缺失明确披露。较宽的30项测试尝试在
spawned worker 导入期间触及3GiB memory.high（所见75459事件、无OOM），
停止精确所属测试域；该批未完整通过，不降低约束。变更不涉及 IPC/取消机制，
因此只补与变更直接相关的21项，不重启无关的大型后端资格。全部测试子进程
已确认退出。组件夹具中的 fetch 是临时测试文件，不是174远端的新下载。

## 唯一完整回放登记（尚非完成回执）

- RUN_ID `d1081_20261106`，RUN_TAG `d1081_3b_positive_demand_options_w1`。
- 原样使用 `20261104_d1047_3b_event_busy_time_w1.yaml`，SHA
  `263bdcd6ff255108ddaffdd9ce46898d7f12be15948d7fe3d7ac27d5ea8828d2`。
- 复用 D1000 runner，W1、fresh_d747_v1、Full service-bin路由、原 admission、
  D1031 profiles、真实HTTP已发布gzip。显式 replay delay=60s，无prefix截断。
- 4000请求，原seed42 trace/subset；`formal_run=false`/exploratory保持原身份。
  不伪造独立留出身份，不因零权重池重启正确性支线；n_correct未知如实保留。
- 先记录实际准备时间、所有违约与物理GPU并集。D1047 3836/4000、
  U6182.715971、P95 1114.091001ms；已有简化消融优势仍需解释。
  W1预算6754.783565160236。不得使用scenario infrastructure估计代替物理U。
- 当前实现包含D1066/D1068，故对旧D1047的变化是开发版本总体比较，不把全部
  差异因果归于本次表示。先完成这一键，完整结果决定接受/撤销，不重复求胜。
- 144项旧投稿hash已核对零变化；用户dirty两文件hash不变。约153GiB可用，
  按现有150/100GiB和102GiB空服务内存门槛启动；不在测量中清理/压缩。
- 174只启动既有3B服务，未重建交付缓存、未动7B/k3s，health确认
  `prepublished_gzip_v1`和`artifact_timing_v2`。真实请求传输仍计入时延。

完整源计划 SHA 为 d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
冻结指标 V1 与单轮 V2 未修改。检查点使用 github-sync 白名单备份，不能强推
已分歧的 FaaSLoRA V2；单独备份不是最终整合。
