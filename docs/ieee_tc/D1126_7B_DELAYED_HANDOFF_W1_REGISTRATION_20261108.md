# D1126：当前 7B DelayedHandoff W1

2026-11-08，性能运行前登记。继续 PLAN §8 A1/A2 的既定顺序；不重新运行
Full、已完成基线或 3B，不改共享 Kubernetes。D1125 已封存并推送
`4ed05eb836c24d710659d190afcbcbd69eae0676`，远端 V2 HEAD 已匹配。

## 科学问题与控制

问题：同一准备规划规则下，把 handoff 的实际执行推迟到 engine-ready，
会如何影响生命周期 GPU 占用、联合 SLO、尾延迟和自然扩容首服务？如果
启动与准备的重叠有效，应看到实际物理准备时序及相应服务结果的差异，
不能用计划条数或登记时间代替真正加载。结果可以无差异或相反，全部保留。

唯一参照是当前同配置 Full D1115，原始结果 SHA256
`99574e407543545937b93d0277c7c70a4d3c28eb8107290d79db65aa1777e577`。
不使用 cap3 工作点的成本或旧 3B 性能，不为了配对重复 Full。
这是自然闭环 A1：每次激活仍在触发时冻结本轮计划、顺序、目标和预算，
Delayed 仅等引擎初始化后执行，不在等待后重新挑选计划。然而跨运行的
队列和需求反馈可改变后续计划，不能称为 A2/A4 的跨运行同 plan-hash 对照。
受控同计划实验仍在后续范围中。

## 配置和实现

`configs/ieee_tc/20261108_d1126_7b_delayed_handoff_w1.yaml`，SHA256
`feb55626baf8b0abecb0df8c948cf95adb7aee8865e98e5a59d87396bfafdb4d`。
相对 D1100/Full 只有 `resource_coordination.ieee_handoff_policy: delayed`
和独占 output/NVMe/HOST 三路径。保留 confirmed routing、service bins、可行集、
原子 reservation、ready 后 hierarchy、软 admission、物理安全、autoscaler。
仍为 Llama-2-7B/vLLM0.30.0/FP16/TP1、14 slots/seqs、1024 context/batch tokens、
0.92 utilization、prefix-cache false、D362 profiles、27.43356142744625ms bin、
min1/max4、5秒需求窗口和同一 D1100 控制规律。九公式和生产代码不变。

已读 D1054 DelayedHandoff 历史、现有 activation/prepare 实现及
fdeea80f/00f2a873 相关修改；复用测试与时序审计方法，不复用 3B 参数/结果。
现有 delayed 分支等待 activation-ready future；冻结 epoch 后不重新采样
需求或预算，执行时仍保留真实安全检查。不是新的算法优化，不新增猜测性机制。

Full 的物理证据复用 D1124 已完成审计：native GPU/HOST 加载各115次，
engine-ready 前均0；first-dispatch 前分别47/55；3个首请求 adapter 匹配。
共享文件的一个 remote job 是 demand 先创建、handoff 后合并，不称独立预取。
D1054 也提醒：queue started_at 不一定是实际传输/加载开始；后订阅者不能
继承先创建任务的起始功劳。按真实 native span、owner/clock、attached_at
与 engine-ready/first-dispatch 边界核验。相同 adapter 命中不证明连续驻留。

## 公共协议、测试与解释边界

复用既有4000请求、500逻辑 adapter、source42 和 W1 8×500视图（相位内
间隔减半、相位间30秒），共同60秒部署通知、fixed_length_greedy_v1、真实
远端预发布 gzip，无模拟等待；不重新生成权重/负载，不清每 phase 缓存。
请求触发的读取、传输、竞争及本地物化仍计时。单轮已知开发回放，
formal_run=false，无 CI、显著性、独立留出或全局最优主张。

完整读取冻结 V1/单轮 V2，沿用 D394 SLO 与 D594 W1 预算
6756.571050141065 GPU-s。报告完整原生合同、U、联合 SLO、TTFT/TPOT/E2E，
正确性未独立证实的字段仍未知；同后端 digest 差异保留审计，不能用数量
正确补造数值正确。本轮 max4 不冒充 G2 cap3 点。

四项针对性 CPU 测试已经通过：一开关/三路径之外完全一致；runner 保留
routing/hierarchy/admission；Delayed 冻结 epoch 且 ready 前不复制；Full
文件准备可先于 ready 且 adoption 无预算空窗。测试主进程0、4/4通过；
run-u393 的管理 wrapper 收尾 timeout/exit1，journal 有 inotify ENOSPC，
实际 MainPID0、cgroup 已移除。记录异常而不重复测试，不改全局限制。
详情见本轮 prelaunch_tests.md。外部指定评审模型不可用，
`[pending external review]`；本地检查不冒称独立评审。

## 唯一启动与安全

先保存配置、测试和登记备份，再刷新资源；tmux 名
`tc-d1126-delayed-handoff-7b-w1`，只启动一次：

```bash
bash scripts/ieee_tc/run_d1100_7b_busy_service_rate_w2.sh W1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1126_20261108/7b_delayed_handoff_w1 \
  d1126_7b_delayed_handoff_w1 \
  /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1126_20261108/7b_nvme_delayed_handoff_w1 \
  /dev/shm/tc-d1126-7b-delayed-handoff-w1 \
  configs/ieee_tc/20261108_d1126_7b_delayed_handoff_w1.yaml tc_ieee_full
```

启动前GPU空闲、可用内存至少102GiB、磁盘至少150GiB且符合增长公式；
服务72/80GiB、swap2GiB，辅助3/4GiB、swap0，既有CPU隔离和独立watchdog。
运行停止线仍为16GiB内存/100GiB磁盘，不把启动门槛当运行中止线。
三路径须未用；复用远端专属服务和已有缓存，不改共享服务/他人进程。
性能测量期间不测试、安装、清理、压缩、哈希扫描、推送或改变服务配置。

## 收尾与后续

实际终态及GPU/远端释放后，复用 D1113 collector、D299 review、D455 stages，
比较同一 D1115。保存同clock远端journal及parent链接供既有分析器使用。
核对4000请求输入、adapter、prompt、target、原生计数、dispatch tier与全部
GPU leases；检查 delayed activation policy、计划冻结/执行等待的物理回执，
同时检查 routing、steady residency 和 admission 保留。first-service 按最早
实际dispatch绑定，initial/natural/controlled分开，缺样本不追加同键运行。
复用 D1054 时序分析方法，仅新增当前 Delayed 的审计，不重复 Full 物理分析。

完整指标表、未知/失败/退化、来源和保护核验封存并备份后，下一项当前7B
NoHOST、CapacityOnly、RoutingOnly/RoutingHandoff；之后受控A2–A5、S1–S13、
CE/价格/Pareto、两轮审稿文档、IEEE图表及V2交付。整个目标仍active。
PLAN SHA256 `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。
不暂存用户 manifest、既有 dirty EXECUTION_STATUS 或无关文件。
