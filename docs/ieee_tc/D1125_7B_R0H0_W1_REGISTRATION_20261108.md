# D1125：当前 7B R0H0 W1，补齐路由与 handoff 四格

2026-11-08，性能运行前登记，尚未启动。D1124 已完整分析并备份到
`a0849029db4afc7a2276360e81ac6a1f4d376abc`，远端 V2 HEAD 已核对一致。
继续 PLAN §8 A1/A2；不重新运行 Full、已完成基线或 3B，不重开共享 k3s。

## 科学问题与已指定来源

| 四格 | readiness 排序 | activating handoff | 来源 |
|---|---|---|---|
| Y11 | 开 | 开 | Full，D1115 |
| Y10 | 开 | 关 | NoHandoff，D1124 |
| Y01 | 关 | 开 | NoRouting，D1123 |
| Y00 | 关 | 关 | R0H0，本轮唯一新增 |

检验两项机制的条件收益和交互，区别于“Full 优于某个单因素消融就证明协同”。
若有有益交互，应由四格的真实资源/服务结果及机制回执共同支持；不预设正负。
低优指标 Γ=Y10+Y01−Y00−Y11，高优指标反向；同时展示两种 handoff 条件下的
routing 差、两种 routing 条件下的 handoff 差，不能只报一个正 Γ。
P95 交互不是可加和的逐请求阶段分解，也不是固定驻留/激活轨迹的反事实。
单轮、已知 source42 开发回放，无 CI、显著性或独立留出主张。

Full 相对 NoRouting 的 GPU-s 更少，但联合 SLO/P95 TTFT 更差；相对 NoHandoff
的 GPU-s/P95 TTFT/联合 SLO 较好，而平均 E2E/TPOT 较差。全部反向结果保留。
不更换 Full、不重跑前三格、不改变协议或消融非目标逻辑来获得预期排序。

## 唯一配置变化

配置 `configs/ieee_tc/20261108_d1125_7b_r0h0_w1.yaml`。
SHA256：`8213f9e60b61335d38ac85410ab819c8430a1cdae68df9051ea2bbf701e99bac`。
相对 D1100/当前 Full，仅有两个目标开关和三个独占 output/NVMe/HOST 路径：

```yaml
resource_coordination:
  ieee_selection_strategy: load_only
  ieee_handoff_policy: no_handoff
```

保留 `routing_policy=ieee_confirmed` owner、同一可行集、live Q、稳定 tie-break、
原子 reservation、来源取得/释放、请求驱动准备、ready 后稳态 hierarchy、软
admission 和物理安全。实际排序和投影均为现有 load_only；不是 legacy
least_connections。初始和自然激活均不创建 handoff 计划或准备入口；不清每
phase 的缓存，不关闭请求准备和 ready 后的驻留。

Llama-2-7B、vLLM0.30.0、FP16/TP1、14 slots/seqs、1024 context/batch tokens、
.92 utilization、prefix cache false、D362 profiles、service-bin
27.43356142744625ms、5秒需求窗口、min1/max4 及完整 D1100 控制规律保持。
自然闭环下队列、驻留、后续计划和激活可以改变，这是结果，不要求逐事件相同。
不修改 IEEE 九公式或生产源码；新增配置和两个局部测试。

## 历史和实现依据

复用 D1056 已运行的 R0H0 组合语义及测试 fixture，不复用其 3B 配置/profile/
性能。D1056 的资源交互为负，P95 正交互也只是两个退化差之间的差值；不能
把这一历史包装为已证实的正协同。D1123 的 load_only 和 D1124 的 no_handoff
分别已在当前 7B 完整触发，当前只补组合。

已核对生产代码历史 fdeea80f/00f2a873：零收益/零需求驻留简化在四格共享。
源码 `_add_ieee_dedicated_instance` 仅在 policy 非 no_handoff 时建计划；
ready 稳态路径独立保留。沿用 D1123/D1124 的官方 vLLM LoRA 加载边界说明，
本次不是新性能优化，不升级后端或引入新机制。

## 公共合同与最小检查

复用原 4,000 请求、500 逻辑 adapter 工件、W1 8×500 视图、相位间30秒/
相位内间隔减半、60秒部署通知和 fixed_length_greedy_v1；不生成新权重/负载。
真实远端预发布 gzip、无模拟等待；请求实际读取/传输/竞争/物化均计时。
保持 exploratory/formal_run=false，不把协议成功数补成独立数值正确数。

冻结 V1/单轮 V2、D394 SLO、D594 W1 预算 6756.571050141065 GPU-s 不变。
四格 max4 不冒充 cap3 的 G2 点，不拼接不同配置的成本和延迟。
数值身份限制、同后端输出 digest 差异及未知原因原样保留；n_correct 未证实
不补造。先报告原生合同、完整物理 U 和 SLO 观测，再说明排名适用范围。

十项局部 CPU 测试：两开关/非目标配置等价、组合 runner owner/hierarchy/
admission、六项 load_only、profile owner 和来源生命周期。D1124 三项
NoHandoff 激活/初始/ready-residency 测试复用不变源码的已完成证据，不增加
blanket GPU smoke。结果写入本轮 prelaunch；不存在独立外部评审则明确
`[pending external review]`，不把执行者检查称为独立审阅。

## 唯一启动与资源

按 run-experiment 使用 tmux `tc-d1125-r0h0-7b-w1`，下方是登记命令，并非
启动回执。运行前先备份配置/测试/文档，然后刷新空闲 GPU、102GiB 可用内存、
150GiB 可用磁盘；服务72/80GiB/swap2，辅助3/4GiB/swap0，CPU隔离及独立
watchdog、16GiB内存/100GiB磁盘停止线不变。没有新额外安全矩阵。

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1125_20261108/7b_r0h0_w1 \
  d1125_7b_r0h0_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1125_20261108/7b_nvme_r0h0_w1 \
  /dev/shm/tc-d1125-7b-r0h0-w1 \
  configs/ieee_tc/20261108_d1125_7b_r0h0_w1.yaml tc_ieee_full
```

仅执行键及三路径未用时启动一次。复用既有远端专属服务与 D80 交付缓存，
不重复发布、不改变共享集群或他人进程。测量期间不测试、维护、压缩、安装、
扫描哈希、推送或改变生产/远端配置；只做低扰动监测，不按阶段胜负停跑。

## 收尾、解释与下一项

必须实际终态、GPU释放和专属远端服务退出后，复用 D1113/D299/D455 及
D1123/D1124 分析；D1056 四格公式只复用方法，绑定当前四个 7B 源结果。
核验4,000请求的输入/adapter/prompt/target/原生计数与dispatch tier，实际
load_only选择、所有activation的no_handoff/空plan/disabled、零handoff动作、
保留steady-residency/admission、真实传输及全部GPU lease。

Full 的物理 handoff 证据直接复用 D1124 已完成审计：native GPU/HOST 各115
次，engine-ready前0/0，first-dispatch前47/55；只有3个首请求adapter匹配。
共享文件的单个remote job是demand先订阅、handoff后合并，不称独立预取。
不重新审计同一物理证据或从计划数量推导隐藏启动开销。

first-service 绑定每epoch最早实际dispatch，initial/natural/controlled分别
报告，未用/失败不删，不为凑样本补跑。四格保留完整GPU-s、SLO、TTFT/TPOT/
E2E、互斥资源窗口、资源事件、机制触发、输入来源和负结果。
完成解释/状态表/保护核对/封存备份后，下一项当前7B DelayedHandoff，然后
NoHOST、CapacityOnly和累积节点；受控A2–A5、S1–S13、CE/价格/Pareto、
审稿设计文档、IEEE图表和V2交付继续保留。整个目标未完成，不切模型绕主线。

PLAN SHA256：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。
不 stage 用户 manifest、既有 dirty EXECUTION_STATUS 全文或未关联文件。
