# D1052：CapacityOnly 3B W1 单因素消融

2026-11-04。当前为 A1 自然闭环消融，不是重新调优 Full。承接 D1049
实现审阅及 D1050/D1051 handoff 对照；完整目标保持 active。源计划
SHA 为 d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
冻结指标 V1 与单轮 V2 继续生效，不改共同 SLO、预算或评价方向。

## 问题、假设和唯一差异

问题：在相同路由、handoff、层级驻留、后端及扩缩容策略下，论文软 admission
E(t) 是否减少主动加载对服务的干扰，并改善完整工作负载的资源—SLO 关系？
如果机制有效，应出现软策略的 admit/defer 差别及服务结果变化；单有推迟
动作不能证明收益，必须同时计入后续请求承担的加载和持卡成本。无触发则
不归因；CapacityOnly 更好或呈 trade-off 时也保留结果，不筛选运行。

从 D1047 Full 配置只增加 resource_coordination.coordination_enabled=false。
另改输出/NVMe/HOST 独占路径。其余配置逐字段一致。保留原始可执行 scenario
tc_ieee_full，分析和图表必须标为 CapacityOnly，不能误记为另一轮 Full。

D1049 已发现激活入口未透传 capacity_only，原默认 False 会让禁用软准入
只覆盖稳态准备。此次生产代码仅增加一行：

```python
capacity_only=not self._coordination_enabled,
```

该行位于 _add_ieee_dedicated_instance 的文件准备计划调用，覆盖 initial 和
natural_scaleout。Full 的有效参数仍为 False，保持既有语义；不修改九公式、
软容量计算、native victim/replacement、引用、槽位、KV 和物理容量检查。
请求驱动加载、handoff 目标、ready 后驻留及真实远端交付继续启用。

历史核查定位至激活路径引入提交 4af6a9cc5，并检查所有 _coordination_enabled
使用位置，确认 IEEE 入口不会因该开关切换为 legacy routing/control。
这是一处消融接线修正，不是基于 NoHandoff 较好而修改 Full 策略。

官方 vLLM 0.30.0 scheduler 的 active-LoRA 限制及 allocate_slots 路径说明
执行可行性仍依赖原生槽位/KV 分配，不能把“关闭软准入”实现为关闭物理保护。
参考：[vLLM 0.30.0 scheduler 源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py)。
这一源码对应关系不证明本项目 E(t) 带来性能收益；收益由消融观测决定。

## 已完成检查及其范围

- ActivationPreparation 与 AdmissionEquationContract 共 21 项 CPU 测试通过；
  新测试覆盖 initial/natural × Full/CapacityOnly 四种组合、相同计划与预算收尾。
  本组保存工具观测摘要，不冒称保留了完整 stdout 或峰值内存记录。
- 三项原生 owner CPU 测试通过：Full 压力延后不驱逐、CapacityOnly 使用相同
  victim 且保留引用、所有槽位 pinned 时不能绕过物理拒绝。保存完整工具输出。
- 复用 D1047 共同配置、运行入口、W1 source/view SHA、SLO/profile 资格。
  逐字校验生产文件相对父版本恰好多上述一行，而非跳过全部来源检查。
- 147 个旧投稿保护文件和两个用户 dirty 文件保持原 SHA；不覆盖历史结果。

prelaunch 第一次检查使用了不符合既有 UUID 约定的 scope 名称，在创建输出前
被拒绝；未启动 GPU。改用既有合法命名后继续，失败回执保留。没有修改安全
条件来绕过检查。所有 CPU 检查限定 3/4 GiB、swap=0、CPU 2,3,26,27、CUDA 隐藏。

预检查产物：paper_results/ieee_tc/ablation/d1052_3b_capacity_only_w1/prelaunch_v2/。
首版 prelaunch/ 保留；提交检查发现 YAML 末尾多余空行，规范格式后新建 v2
来源回执，配置含义不变，未重复 CPU 测试或运行 GPU。首次提交检查在 commit
之前停止，未向远端推送不合格版本。
原始工作目录：/home/qhq/serverless_llm_experiment/results/ieee_tc/
p2_backend_qualification/d1052_20261104/。

## 唯一完整回放与收尾

复用 scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh：

```text
FAASLORA_D1000_RUN_ID=d1052_20261104
FAASLORA_D1000_RUN_TAG=d1052_3b_capacity_only_w1
FAASLORA_D1000_CONFIG=configs/ieee_tc/20261104_d1052_3b_capacity_only_w1.yaml
FAASLORA_D1000_WORKLOAD=W1
FAASLORA_D1000_HOST_DESCRIPTOR=fresh_d747_v1
```

不设 prefix 或详细 profiling。使用既有 4,000 请求、500 逻辑工件池、真实
远端已发布 gzip、冷本轮本地缓存、共同 60 秒准备和 W1 30 秒 phase 间隔。
同 vLLM 0.30.0、FP16/TP1、32 slots、c8、max_batched_tokens=4096、
max_model_len=1024、GPU utilization=.72、1–3 实例及 D1031 profile。
不重建输入/工件，不新增 Full 回放或 GPU smoke。

启动仍检查磁盘至少 150 GiB、内存余量、GPU 空闲，沿用服务 72/80 GiB、
swap 2 GiB、独立回放/watchdog、磁盘 100 GiB 和主机内存 16 GiB 停止规则。
一次一个重型运行，测量期间不压缩、安装、推送或操作共享集群/远端配置。

完成后先释放资源，再检查全部请求、原生 token、输入映射、输出 digest、
dispatch tier、原生 admission/movement、生命周期 GPU 占用及资源事件。
重点提取 initial/natural/稳态准备是否都使用 capacity_only=True；demand
路径与物理拒绝单独统计，不把“软策略未执行”当作“无安全检查”。
与 D1047 Full 比较联合 SLO、GPU-s、P95 TTFT/TPOT 和完整阶段分解，不将
不同自然闭环计划声称为同候选受控 S3。S3 仍需按原计划单独完成。

## 证据边界和后续范围

本轮仍为已知 source42 开发负载的一次描述性消融，无运行间 CI。保留
n_correct=null、independent_numerical_proof=false、formal_ranking_eligible=false。
3B 工件全零更新与独立数值可区分性限制继续披露。此前 Full/NoHandoff 的
206 个输出 digest 差异及其有限参考解释详见 D1051；不默认新差异均正确，
也不通过反复回放获取相同输出。输入/绑定/长度检查不冒充数值正确性证明。

HydraServe 已使用独立 rootless Podman 适配；共享 k3s 不属于本项目，
不得修改其 kubeconfig、服务或资源。3B W0/W1/W2 已有 D807–D809 回放，
不因权限问题的迟到回复重新部署或回放；pp1 未覆盖 pipeline/consolidation
的限制保留。

尚未完成的 A1 十一变体、两个 7B 累积节点、A2–A5 confirmed-state/受控
handoff/first-service、S1–S13、CE/价格/Pareto、两轮审稿设计文档、IEEE 图表
与最终 V2 集成都不删减。EXECUTION_STATUS.md 为用户预存 dirty 文件，
不覆盖；本文件承接 D1051，不能让其旧 D998 页首覆盖当前阶段。

本文件预登记时尚未启动 D1052 GPU；启动/完成事实另附回执，不预写结果。
