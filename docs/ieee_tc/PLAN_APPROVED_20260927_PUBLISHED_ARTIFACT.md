# PrimeLoRA → IEEE TC：完整实验、自适应优化与逐实验图表交付计划

## 执行修订：2026-09-27（已发布工件边界；用户最新澄清）

- 正式研究背景为：远端已有发布完成、可直接读取的 LoRA 工件；不在每次下载请求中临时打包、压缩、转格式或扫描整池。此类人为附加的制品制作不进入正式服务测量路径，也不作为 PrimeLoRA 的性能贡献。
- 优先复用远端既有不可变文件和已冻结内容清单，直接传输；不重新生成权重、负载或第二套完整工件池。所有系统采用同一冻结交付协议，保留各自合法的请求驱动获取、缓存和层级驻留机制。
- 若确需发布前离线准备，须对整个既有静态集合采用相同规则、在共同部署通知之前完成并单独记录；不得利用未来请求/热点、只准备 Prime 的命中集合，或将推理本机的预加载与资源占用移出生命周期计量。
- 真实远端到推理机的数据传输、本次请求引发的共享带宽竞争、必要对象读取/响应和本地加载属于实际服务路径；共同环境不意味着每系统获取次数、并发或累计耗时相同。分别测量必要读取/发送/接收，不把整个获取时间冒称纯链路时间。
- 主 TTFT/E2E 使用修正后真实请求时间线，不通过从旧结果减去累计打包时间构造。D75–D77 动态打包路径结果保留为内容/取消功能检查和交付方式诊断，不进入新主比较或冻结准备时间 profile。内容 SHA 证据可复用；传输方式改变后仅重新验证受影响的路径。
- 当前远端实际协商100 Mbps：网卡支持并通告1 Gbps，但对端仅通告10/100。未确认交换机端口/布线根因前不擅自改链路；不将0.25/0.5/1G限额标成实测吞吐。不删除真实传输延迟来回避硬件限制；使用已有分层/状态机制实验与如实标注的敏感性界定结论。
- 其余核心公式、Prime优先主线、暂停baseline、推理机磁盘/内存安全限制和历史保护规则不变。下文关于动态打包计量的记录仅适用于历史诊断；以本节的正式研究边界为准。

## 执行修订：2026-09-27（用户最新授权，优先于下文历史排程）

- 本计划已获执行授权；下文“本轮不执行/计划模式”保留为最初批准时的历史背景，不再表示禁止执行。
- 当前 D74 Serverless 3B 修复版完成后，按“释放资源→校验→分析→图表→提交备份”收尾，暂存基线工作。已经准备但未运行的 3B 原版保留为待执行，不为凑齐对照继续跑。下一主线先完整实现并验证 PrimeLoRA 的 IEEE Full；随后恢复原基线顺序。此前通过的矩阵、未完成项和失败证据均保留。
- 仅制品传输节点退出通用 150 GiB 磁盘门槛，使用本节下述并发临时打包峰值、日志及安全余量规则；推理节点原磁盘、内存、防 OOM 门槛完全不变，不删除唯一工件。
- 保留 PrimeLoRA 在 NVMe/HOST/GPU 合理驻留、减少远程获取次数及并发竞争的合法优势。不强制各系统具有相同 miss 次数、传输次数、字节数或远程耗时，不关闭缓存/准备机制来“消除环境差别”。
- 远端实验无关任务和管理操作是外部干扰；动态打包是制品交付实现开销；真实请求引发的读取、传输、miss 和竞争是场景内开销。分别测量，不能混称网络时间。共同面临的开销不假定影响均等。
- 正式测量期间不重启远端服务、不调整配置、不进行额外下载/打包压力测试、哈希扫描、清理或压缩。共同服务配置、资源边界和监控方式在验证后冻结；记录外部干扰，不能根据结果胜负决定哪些运行受污染。
- 记录远端等待/打包/发送及本地接收/解包的关联 span、字节和次数。保留完整用户 E2E，同时给出阶段分解；跨主机未校准时间不直接相减，重叠阶段不重复相加，不能简单以 E2E 减去累计打包时间伪造“无外力”结果。优先检查能否共同复用已有不可变交付制品，将非必要动态打包移出请求关键路径；改变交付方式须重新验证和冻结执行合同，不改权重、不重建负载或整池复制。

## 一、最终目标、本轮修订与执行规范

本版完整替代上一版，可从头执行到尾。保留此前通过的模型、工件、负载、指标、主比较、消融、敏感性、历史复用和数据保护要求。本轮不执行代码修改、远程服务操作或实验。

### 1.1 系统优化目标

以两个目标评价 PrimeLoRA，不再以 CE 单指标决定系统优劣：

1. **G1：相同工作负载全部正确完成，并满足共同 TTFT/TPOT 服务要求时，减少生命周期 GPU-s/request。**
2. **G2：满足共同资源预算和服务要求时，降低扩容与 adapter churn 场景的 P95 TTFT，并检验联合 SLO 达成率是否非劣。**

CE、美元成本、平均延迟、吞吐作为补充，不能替代正确完成、共同 SLO 和物理资源占用。

持续优化以 PrimeLoRA 在 G1/G2 上领先为目标，但区分：

- 实现正确；
- 实验完成；
- 观测值领先；
- 统计证据支持领先。

不能通过筛选正式运行、削弱 baseline、改变共同协议或隐藏退化，把未达到的目标写成已经达到。

### 1.2 本轮客观裁决

| 内容 | 决定 |
|---|---|
| ChatGPT 指出的内存启动门槛不自洽 | 采纳，改为“剩余可能增长量＋安全保留量＋余量”的检查 |
| `memory.high` 影响性能、不能只当告警 | 采纳，纳入所有实验及复用合同 |
| CPU、Ray、swap、spill、真实 worker 限制需要明确 | 采纳，补入统一资源协议 |
| 截断运行不能表现为低成本成功 | 采纳，保留完整计划分母、观察截断和失败分类 |
| 为资源护栏增加整套新性能矩阵 | 不采纳；增加安全自测和资格检查即可 |
| Serverless 四分钟等待已经证明 GPU 吞吐不足 | 不采纳该既有归因；发现更具体的控制路径瓶颈，优先核查 |
| Serverless 最小实现修复 | 按本轮确认允许；保留原有调度、扩缩容和加载策略，公开补丁 |
| Serverless 图表名称 | 所有新图表统一显示 **Serverless**，不带 `-new` |
| Prime 尽量采用可观测量自适应 | 采纳；保留必要物理边界、协议常量和按模型配置 |
| 每个实验及时画图或制表 | 采纳，成为进入下一实验的交付步骤 |
| 图形突出 Prime 优势 | 用合适的问题、指标和视觉重点体现，不用删点、裁轴或混用工作点制造优势 |

### 1.3 不变的工程边界

- IEEE 版本论文的核心思想、九个行间公式及其语义保持一致。
- 允许优化异步流水线、缓存表示、数据结构、队列、后端集成、批处理、CPU/NUMA 使用、资源生命周期和控制实现。
- 不为适配旧实现而反向修改论文核心公式。
- 每轮优化必须结合历史日志、代码历史，以及联网查到的相关原始论文、公式、实现文档和源码。
- 不使用未来请求、未来热点或未来实际生成长度。
- 不新增 LoRA 权重，不重新生成完整执行负载，不复制整池工件。
- 正式工作使用既定 V2 worktree，不在当前 IDE 的 dirty main 中启动正式实验。
- 不覆盖旧投稿结果，不修改论文 LaTeX/PDF，不新增 13B 或 SGLang 实验。

### 1.4 当前比较系统

| 图表名称 | 内部身份与角色 | 状态 |
|---|---|---|
| PrimeLoRA | 被测系统 | 先进行 IEEE 对齐和后端优化 |
| vLLM | 与 Prime 同版本的 serverful LoRA 后端 | 排除后端升级独享收益 |
| S-LoRA | 既有 serverful Multi-LoRA 基线 | 复用环境，重新审计生成与计量 |
| Serverless | 内部仍保留 `serverlessllm_new`、官方 commit、补丁和环境身份 | baseline 阶段优先处理 |
| HydraServe | 新增 serverless 论文系统 | 官方专用后端，完整资格待验证 |
| Loquetier | `Loquetier + static-sharded deployment` | 图注说明静态分片适配，不冒称原生集群功能 |
| ElasticLocality | 强内部对照 | 普通 locality-aware routing，不是外部论文基线 |
| dLoRA | 有限最近邻对照 | 单独进行 3B W1/W2 比较 |

主矩阵为前七项，包括 PrimeLoRA。dLoRA 单独报告有限覆盖。

Chameleon、ELORA、Toppings 保留相关工作与官方制品审计，不自动扩入性能矩阵。HydraServe、Loquetier 未通过资格前不得标记为已复现成功。

### 1.5 执行主线

1. 建立安全护栏、资产保护清单和远程自动管理能力。
2. 并行进行只读审计：Table 1/Full 来源、Serverless 历史运行、各 baseline 复现条件。
3. 完成 Prime 的 IEEE 公式合同、计量正确性和新版 vLLM 接入。
4. baseline 重型实验优先级：**Serverless → vLLM → S-LoRA → dLoRA 3B → Loquetier → HydraServe**。vLLM 的共同后端资格随 Prime 接入提前完成。
5. 已合格系统先做开发期比较，及时定位 Prime 瓶颈。
6. 公共协议冻结后完成 M1/M2。
7. Prime 未达到目标时进入有证据的开发优化循环，冻结新版本后完整评估。
8. 完成核心消融及必要机制证据。
9. 最后执行敏感性和补充分析。
10. 汇总两轮审稿证据，交付图表、文档和 V2 分支。

任何阶段均执行：

**运行结束 → 资源释放 → 数据校验 → 诊断图/表 → 结论与下一步 → 下一实验。**

---

## 二、安全资源包络、进程管理与磁盘规范

### 2.1 当前事实与适用范围

本轮只读检查得到：

- 根文件系统约 90% 已用，可用约 339 GiB。
- 物理内存约 125 GiB，当前 `MemAvailable` 约 103 GiB。
- swap 已使用约 8.3 GiB。
- 24 个物理核、48 个逻辑 CPU，双 NUMA 节点。
- 已安装 Times New Roman。
- 当前 scope 脚本没有完整硬内存限制，不能直接视为防 OOM 保证。

这些是检查时状态，不是执行时资源保证。每次启动均重新检查。

### 2.2 共同服务资源包络

默认配置：

| 项目 | 限制 |
|---|---|
| 整套被测服务 | `MemoryHigh=72 GiB`、`MemoryMax=80 GiB` |
| 服务资源域 swap | `memory.swap.max=2 GiB` |
| 回放器与外置监控 | 独立资源域，合计最多 4 GiB |
| 服务 CPU | 当前拓扑的逻辑 CPU `4–23,28–47`，共 20 个物理核、40 个逻辑 CPU |
| 回放与监控 CPU | `2,3,26,27` |
| 不分配给实验的 CPU | `0,1,24,25`，为系统及交互保留 |
| 重型任务并发 | 一次一个；不与安装、构建、压缩或另一模型服务并行 |
| 构建并行 | 默认最多两个编译任务，独立受限，不与性能运行并发 |

CPU 集合在实施时核对拓扑后冻结；若机器拓扑不同，按相同物理核/SMT 分组规则生成并记录，不逐系统任意改变。

- 服务内部允许在共同 CPU 集合内进行原生线程和 NUMA 优化。
- 默认遵循实际 first-touch，局部绑定根据 GPU 拓扑和验证证据冻结。
- 不通过 Ray 的 `num_cpus` 或 Pod 的 CPU request 冒充操作系统隔离。
- 记录实际 affinity、cpuset、quota、父级限制和 CPU 节流。
- 被测 router、autoscaler、planner、后端、Ray 控制组件、workers、HOST 缓存和解包辅助进程均计入服务预算，不能移到外置监控组规避限制。
- 远端 artifact 服务另行记录资源，不混入推理机 80 GiB，也不忽略其开销。

`memory.high` 会导致回收和节流，`memory.max` 才是硬限制。两者都属于性能条件；出现 high 事件但正确完成的运行仍然有效。[Linux cgroup v2 文档](https://docs.kernel.org/admin-guide/cgroup-v2.html)

### 2.3 修正内存启动门槛

启动检查采用：

\[
A_0\ge
\Delta M_{\rm svc}^{remain}
+\Delta M_{\rm aux}^{remain}
+M_{\rm stop}+M_{\rm margin}.
\]

其中：

- \(A_0\)：检查时主机可用内存；
- \(\Delta M^{remain}\)：检查后相应资源域尚可能增加的占用；
- \(M_{\rm stop}=16\) GiB；
- 本机额外余量 \(M_{\rm margin}=2\) GiB。

若服务和外置工具均未启动：

\[
A_0\ge80+4+16+2=102\ {\rm GiB}.
\]

若部分进程已启动并计入当前占用，只计算其剩余增长，不重复扣除。

同时检查祖先 cgroup 的实际剩余容量。主机空闲充足但父级限制不足，同样不能放行。

此公式是保守准入检查，不是独占预留保证。资源不足时等待或清理本实验可重建资源；不能只对失败 baseline 临时降低包络，也不把 swap 当作物理余量替代。

### 2.4 真实限制必须先于进程启动

- 在启动模型、控制器和 workers 前建立资源域。
- 不把已经加载模型的进程事后迁入组，当作完整限制和计费。
- systemd user manager 可见不等于控制器已经正确委派。
- 保存 PID、UID、容器/Pod 身份、cgroup 路径、有效限制和父级限制。
- 扩容产生的新进程也必须被覆盖。
- Docker/Kubernetes 实际 worker 必须受限，不能只限制启动 CLI。
- 优先使用共同父资源域；若只能逐 Pod 设置限制，合计上限必须满足共同预算，并公开分配方式及与共享池的差别。
- 无法核验实际 workers 时，停止高风险资格与正式实验，不静默回退为无限制运行。

实施依据本机 systemd/cgroup 版本，而非照搬最新在线文档中的可选功能。[systemd 委派说明](https://systemd.io/CGROUP_DELEGATION/)

### 2.5 Ray、共享内存、swap 与重试

- 使用本轮专属 Ray 集群和地址，不连接遗留的未受限集群。
- 对象存储以**推理机本次服务合计 8 GiB**为资格起点，不是每个 worker 各 8 GiB。
- 实际 raylet 数、逐节点容量和合计值必须记录。
- 允许在 80 GiB 总包络内根据验证证据调整内部内存分配，但冻结后不能逐测试点改变。
- `/dev/shm`、tmpfs、HOST tier、CPU KV、模型重复物化、解包和对象存储均纳入审计。
- Ray spill 使用本轮可追溯目录，记录溢写、恢复、磁盘峰值与 I/O。
- 对象存储属于服务总量，不与顶层 cgroup 再重复相加。
- 分别记录 `memory.swap.current/max`；整机旧 swap 占用不能代表实验 swap。
- 核查所用 Ray 版本的内存监控、worker 终止、actor 重启和重试行为；保留原生合理恢复。
- 所有 attempt 关联原始 request ID，重试不重置到达时间、不删除资源开销。

Ray 对象存储和内存保护有独立行为，必须与 Linux OOM 分开追踪。[内存管理](https://docs.ray.io/en/latest/ray-core/scheduling/memory-management.html)、[OOM 防护](https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html)

### 2.6 外置监控与安全退出

每秒记录：

- 主机 `MemAvailable`、swap 和内存 PSI；
- 顶层服务 cgroup 的 current、peak、high/max/oom/oom_kill 增量；
- cgroup PSI、swap、CPU 节流；
- GPU 分配、显存、进程身份；
- Ray object store、重启、spill 和关键服务状态。

磁盘、inode、quota 每 30 秒检查。子组数据用于分解，不与父组重复求和。

默认保护规则：

1. `MemAvailable <24 GiB`：告警，不启动下一任务。
2. `MemAvailable <16 GiB`：立即安全中止本轮。
3. `MemAvailable <24 GiB`，且主机 memory PSI `full avg10 ≥10%` 连续 10 次一秒采样：安全中止。
4. GPU 或主机发生需要管理员处理的异常：停止 campaign，不自动重启整机或全局 GPU reset。

阈值属于本机保护协议，不是性能调参变量；实施前通过小型自测冻结。

外置 watchdog 独立于被测服务组。紧急收尾最多等待 10 秒，再清理本实验进程；普通收尾默认最多 60 秒。只处理所有权明确的进程，核对 PID 与启动身份。

GPU 未释放则记录持续占用和截断状态，停止后续实验，不伪造 release 时间。

### 2.7 中止分类与分母

区分：

- `system_resource_failure`：服务预算内的分配失败、框架内存终止或硬限制失败；
- `environment_interference`：有证据的外部作业干扰；
- `protocol_or_launcher_error`：准入、限制或监控配置错误；
- `safety_abort_unattributed`：原因尚未确认。

仅凭主机可用内存下降，不能宣布 baseline 超过 80 GiB。

每次运行保存：

\[
N_{\rm plan},N_{\rm arrived},N_{\rm submitted},
N_{\rm terminal},N_{\rm correct},N_{\rm good}.
\]

截断运行：

- \(N_{\rm plan}=4000\) 不变；
- 尚未到达的请求不伪造 timeout；
- 只报告观察到的 \(U_{\rm obs}\) 和条件时延；
- 不进入完整 M1/M2 达标排名；
- 系统导致的中止进入失败汇总；
- 环境或 launcher 错误修复后重跑同一 block，保留旧 attempt。

不得通过“补足五个成功运行”洗掉系统失败。已确认会威胁主机的相同配置可停止后续重复，标记 `not_run_safety`，不伪造五次测量。

### 2.8 磁盘门槛与清理

对推理/构建节点的每个相关文件系统检查（制品专用节点使用后述独立规则）：

\[
D_{\rm free}\ge
\max\{150\ {\rm GiB},
100\ {\rm GiB}+1.5\Delta D_{\rm peak}\}.
\]

\(\Delta D_{\rm peak}\) 是本任务尚未发生的新增峰值，包括转换、临时压缩、解包、构建、容器层、spill 和日志。

- 低于 150 GiB：不启动新的重型任务。
- 低于 100 GiB：安全停止本实验的新增写入，保留必要日志。
- 结果、临时文件、容器、spill、远端打包目录分属不同文件系统时分别检查 inode、quota 和容量。
- 复用既有存储审计和清理脚本，不按目录名执行大范围删除。

允许自动清理：

- 已结束实验的可重建工作缓存；
- 无进程使用的临时解包和构建中间件；
- 有可靠保留副本、SHA 一致、引用核查通过的重复件；
- 不再被复现配置引用的过时缓存。

保护唯一模型、LoRA、trace、原始测量、失败证据、源码历史、用户修改、仍被引用的环境及其他项目数据。

同 SHA 不是充分删除条件，还需检查软链接、挂载点、打开文件和引用。日志无损压缩后验证解压 SHA；旧投稿零变化目录不原地压缩。

清理回执记录路径、原因、引用检查、保留位置和释放空间。批量哈希、压缩和清理放在性能实验之间。

### 2.9 制品专用节点的独立磁盘规则（2026-09-27 批准）

对临时归档、日志实际所在的每个文件系统分别计算：

\[
D_{\rm free,f}\ge R_f+\lceil1.5(P_f+L_f)\rceil,
\qquad P_f=\sum_j c_j s_j^{remaining}.
\]

- `c_j` 是该文件系统上所有实验客户端/模型合计、可验证的最大并发打包数；不能把每 worker 的上限当成全服务上限。未知时不填零，也不为了过门槛而只限制某一系统的传输并发。
- `s_j^{remaining}` 是对应工件单次临时归档尚可能增加的分配字节上界；基于现有工件/归档实测和格式上界，不能只用零权重压缩后的平均大小推断最大值。文件已占用部分不重复扣除，压缩变化要有余量。
- `L_f` 覆盖本轮日志、失败收尾和监控的新增上界；`R_f>0` 是单独记录依据、运行前冻结的系统/其他用户安全保留量。保留原有 1.5 倍新增量余量，不继承推理机 150/100 GiB 固定底线。
- 同文件系统的各服务峰值求和；不同文件系统、inode 和实际 quota 分别检查。只有容量检查通过不等于服务、正确性或性能已合格。
- 准入只判断是否能安全运行，不改变冻结的缓存、路由、压缩、并发或限速配置。运行中出现保护事件保留原始数据并按既有失败/干扰规则分类，不在运行中降低门槛。
- 推理机仍采用 2.8 的 150/100 GiB，内存条款完全不变。远端规则的依据、实测值和回执纳入 REMOTE_ACCESS 与运行 manifest。

---

## 三、远端自动连接、服务管理与资产复用

### 3.1 远端配置

| 项目 | 固定配置 |
|---|---|
| SSH | `lab14@10.199.227.174:8122` |
| artifact HTTP | `192.168.4.174` |
| 推理机实验网 | `192.168.4.178` |
| 3B/7B HTTP 端口 | `18080` / `18081` |
| 服务脚本 | `/home/lab14/primelora_remote/remote_artifact_node/server.py` |
| 3B 工件 | `/home/lab14/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope` |
| 7B 工件 | `/home/lab14/primelora_remote_artifacts/llama2_7b_a500_v2_publicmix` |

只运行 artifact 服务，不安装推理后端，不启动 13B，不重新上传或解压整池。

### 3.2 自动认证与持久管理

你已授权首次认证和专用公钥部署。实施时：

1. 核验主机身份，不关闭 host-key 检查。
2. 使用已提供的凭据完成首次受控认证。
3. 创建不覆盖已有文件的专用 SSH 密钥，私钥放在仓库外，权限 `0600`。
4. 增量添加远端授权，限制不必要的转发能力。
5. 建立 `primelora-artifact-174` 别名，验证 `BatchMode=yes`。
6. 验证新会话中不依赖手动密码输入即可管理两个服务。
7. 持久记忆保存主机、端口、别名、密钥位置和服务管理规则，**不保存明文密码、私钥或 HTTP token**。

正常情况下，后续由 Codex 自行连接、启动、停止、重启和验收，无需你手动执行远端命令。

指纹变化、权限撤销或网络不可达时明确报告，不绕过安全检查。本轮仍在计划模式，尚未部署密钥，不能声称目前已具备免交互访问能力。

### 3.3 服务管理接口与验收

复用原服务脚本。3B 启动参数：

```bash
python3 /home/lab14/primelora_remote/remote_artifact_node/server.py \
  --root /home/lab14/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope \
  --host 192.168.4.174 --port 18080 \
  --token-env PRIME_REMOTE_TOKEN
```

7B 使用对应工件目录及端口 18081。

管理入口提供 `status/start/stop/restart/health`：

- 优先使用远端可用的受管理用户服务；否则使用具有进程身份校验、锁和日志的受控后台管理。
- token 通过受保护环境传入，不写进命令参数。
- 启动前检查目录确实存在，避免脚本创建空目录掩盖错误。
- 健康且配置一致的既有服务直接复用。
- 不终止未知端口监听者。
- stop/restart 核对 PID、UID、命令及进程启动时间。

验收必须覆盖：

- 两服务的启动、停止、重启和重新连接；
- 认证 health；
- 完整 manifest 与冻结 500-adapter 清单一致；
- 两模型实际下载、解包和内容 SHA；
- 全 adapter 功能覆盖；
- 失败、取消后的临时文件清理；
- 远端 CPU、RAM、磁盘、网络监控。

单纯返回 PID 或查看三个 adapter 不算验收完成。

### 3.4 真实远程主协议

- cold/first-touch 从同一真实远端 materialize，不静默回退本地 frozen pool。
- materialize 后允许各系统原生 NVMe/HOST/GPU 缓存。
- 每轮初始化规定缓存；不在每个 burst、rotation 或 scale-out 清空全部缓存。
- backbone 本地磁盘、HOST 和 GPU 状态分别记录。
- fetch 关联 request/transfer ID、服务端记录、线上字节、解包字节和内容 SHA。
- 正式路径直接获取已发布工件，不执行按请求动态打包/压缩；必要读取、传输和本地落盘/校验分别计量，依据内容 SHA 判断权重一致性。
- 减少 fetch 带来的传输、请求竞争和本地加载减少属于待测收益；历史动态打包减少不作为新正式实验的贡献。
- 原生 HTTP 主实验不注入模拟等待；核对实际生效值，不能只看 wrapper 默认参数。
- 不清远端整机 page cache，不用 SSH 隧道替代正式数据链路。

本地模拟历史结果保留为历史或诊断证据，不冒充本轮真实远程主结果。

### 3.5 数据与历史结果复用

保留既有 7B/3B 工件池和 seed42 源 trace；只保存参数化变换和索引，不复制完整数据。

| 等级 | 处理 |
|---|---|
| R0 | 同执行合同、字段完整，直接复用 |
| R1 | 执行有效，仅统计或图表错误，离线重算 |
| R2 | 合同不同，复用工件、环境、代码适配和诊断，不混入新主结果 |
| R3 | 执行错误或关键事实无法恢复，修复后重跑受影响运行 |

合同包括代码、后端、输入、生成、初态、在线超时、CPU、内存 high/max、swap、Ray/spill/retry、监控、真实远程路径和区组。

旧无限制运行不能仅凭一次低内存峰值自动视为 R0。曾用于选配置的数据保留开发身份，不能因重新命名而变成独立测试。

能从日志恢复的指标先恢复。只重跑受影响执行键，不因一个汇总错误重跑全部实验。

---

## 四、公共负载、生成、指标与统计协议

### 4.1 共同准备窗口与请求发生器

主实验固定：

```text
init_mode=deployment_notice_v1
arrival_mode=trace_open_loop
control_mode=natural_closed_loop
```

- \(t=-60s\) 发出共同部署通知，\(t=0\) 开始业务。
- 准备资源计费，不按各系统 ready 时间平移业务。
- 未 ready 正常排队，已 ready 不人为卸载。
- 只使用静态清单、合法历史和冻结 profile。
- 全冷启动放在启动诊断、A4、S1，不新增完整冷启动主矩阵。

请求按原计划时间到达，不等前一请求完成。记录计划到达、任务创建、提交、连接池与 semaphore 等待。

先用轻量响应端点验证回放注入能力。客户端瓶颈是测量问题，服务端背压是系统行为；提交滞后不从用户时延中扣除。

相同 req/s 但重新生成 Poisson 到达，不等于同一 trace。并发上限也可能改变实际到达，必须审计。[vLLM benchmark 文档](https://docs.vllm.ai/en/latest/cli/bench/serve/)

### 4.2 三个主场景

| 场景 | 设置 |
|---|---|
| W0 | 原 4,000 请求、500 adapters、Zipf 1.0、rotation 500 |
| W1 | 同序列分为 8×500，phase 内时间压缩一半，phase 间隔 30 秒 |
| W2 | W0 arrival/prompt/token 不变，rotation 100 |

W1：

\[
b_1=0,\qquad
a'_r=b_k+\frac{a_r-a_{k,\mathrm{first}}}{2},
\qquad
b_{k+1}=a'_{k,\mathrm{last}}+30s.
\]

不按 Full cooldown 改间隔，不等待实际 drain，不强求八次自然扩缩容，不额外添加末尾空闲期。

### 4.3 固定生成契约

主协议 `fixed_length_greedy_v1`：

\[
target_r=\min(source\_expected_r,256).
\]

- 相同 canonical prompt、tokenizer、特殊 token、prompt hash。
- prompt 内容最多 759 tokens，最终上下文必须合法。
- `temperature=0`、`top_p=1`、无 stop，忽略 EOS 至目标长度。
- Prime/vLLM 使用原生 `token_ids`。
- S-LoRA 使用 native SSE 整数 `token.id`。
- 其他系统使用经核验的原生计数。
- 不用 expected tokens 或文本重分词替代主计数。
- 全部 `actual_tokens==target_tokens`，无 base-model fallback。
- 跨后端不强求逐字相同，但必须验证正确 adapter；同后端消融输出变化需审计。

非法目标在全局输入审计中统一处理，不能逐系统选择性删除。

记录逻辑 adapter 数、独立权重 SHA、来源和填充关系；500 个逻辑 ID 不自动等于 500 个独立训练模型。

### 4.4 请求指标

定义计划到达 \(a_r\)、提交 \(e_r\)、后端 dispatch \(d_r\)、首末 token \(f_r,l_r\)、完成通知 \(c_r\)、原生输出数量 \(n_r\)。

\[
TTFT_r=f_r-a_r,
\qquad
TPOT_r=\frac{l_r-f_r}{n_r-1}\quad(n_r\ge2).
\]

\[
E2E_r=c_r-a_r
=(e_r-a_r)+(d_r-e_r)+(f_r-d_r)
+(l_r-f_r)+(c_r-l_r).
\]

若 Dispatch Wait 为 \(d_r-a_r\)，不能再重复加入提交滞后。

跨主机未校准时间戳不直接相减。1 ms 恒等式与 TPOT 重算门槛只用于可靠共同时间域；其他阶段采用本地 span 或明确同步误差。

联合 SLO：

\[
I_r=
\begin{cases}
0,&z_r=0,\\
\mathbf1[TTFT_r\le\theta_r^T],&z_r=1,n_r=1,\\
\mathbf1[TTFT_r\le\theta_r^T]\mathbf1[TPOT_r\le\theta_r^P],
&z_r=1,n_r\ge2,
\end{cases}
\]

\[
A_{\rm joint}=\frac{\sum_r I_r}{N}.
\]

失败先分支处理；单 token 的 TPOT 为 N/A。重试不增加 offered 分母、不重置原到达时间。

### 4.5 物理 GPU 生命周期

\[
U=\sum_d\int a_d(t)\,dt,
\qquad
R=\frac{U}{N_{\rm correct}}.
\]

\(a_d(t)\) 表示服务持有该物理 GPU，而不是 GPU utilization。

- 同卡多进程计一次，TP 多卡分别计。
- startup、preparation、inference 重叠时不重复累计。
- GPU 常驻守护进程仍持卡则继续计量。
- CPU-only 准备不虚构 GPU 时间。

可加和分项采用互斥时间窗：

1. 业务到达前；
2. 共同到达窗口；
3. 到达结束至全部请求终态；
4. 请求终态至资源释放。

令终态边界不早于到达窗口结束，空区间计零，要求：

\[
\sum_h U_h=U.
\]

活动区间另用于重叠与关键路径分析，不相加充当账单。

单元测试：GPU 分配 10 秒，其中 6 秒准备与启动重叠，总量必须是 10 GPU-s。

若 \(p\) 为美元/GPU-hour：

\[
C_{\rm GPU/req}=\frac{pU}{3600N_{\rm correct}}.
\]

零正确完成时 GPU-s/correct-request 和 cost/correct-request 为 N/A，不产生有限“优值”。

CPU、HOST、远端资源另报，不把 GPU-only 成本称为完整云账单。

### 4.6 SLO、goodput 与超时

共同 warm reference：

- 使用最终共同 vLLM 和相同资源包络；
- adapter 真正 GPU-ready；
- batch 8、固定 prefix-cache；
- 既有输入长度四分位分组，重复边界合并；
- 每组 256 请求、三轮；
- TPOT 样本至少两 token。

若 batch 8 不可行，在正式冻结前统一选择所有参考条件可行的最大 batch≤8，并公开，不逐系统选择参考。

\[
\theta_r^T=5T_{\mu,\kappa(r)}^{warm},
\qquad
\theta_r^P=2P_{\mu,\kappa(r)}^{warm}.
\]

倍率参考 HydraServe；分组、样本数、95% 联合达成是本项目协议，不称行业标准。[HydraServe 论文](https://www.usenix.org/system/files/nsdi26-lou.pdf)

\[
\mathrm{SLO\ goodput}=\frac{N_{\rm good}}{T_{\rm observation}},
\qquad
\mathrm{good\ requests/\$}=\frac{N_{\rm good}}{C}.
\]

观察窗口包含固定到达窗口及 drain；终态后清理计入成本，不人为扩大 goodput 分母。

没有应用 deadline 时：

- 资格/验证请求保护为共同 1,800 秒；
- 从所有实际参测系统的合法路径取得启动耗时 \(T_\mu^{init}\) 和正确验证 E2E 最大值 \(T_\mu^{val}\)；
- 缺记录不填零；
- 冻结：

\[
H_\mu=\max(600,2T_\mu^{init},2T_\mu^{val}),
\]

\[
\Delta_r^{timeout}
=\max\{H_\mu,\,
10[\theta_r^{T,0}+\max(n_r^{max}-1,0)\theta_r^{P,0}]\}.
\]

这是有限保护，不是理论上界。正式运行不根据胜负调整；SLO 违约本身不立即取消。

### 4.7 重复、配对与结论

- 源 trace 保留 `source_trace_seed=42`。
- 41 为开发身份，42 为 smoke，43–47 为首批正式运行块。
- 新运行编号不是新独立真实 trace。
- 有既有未用时间窗口时按索引划分；没有则明确结论是固定回放下的运行变异。
- 每轮从相同合法历史和 profile 重置，不继承上一正式轮的学习状态。
- 主比较默认五块，支持性消融三块；正式前冻结。
- 系统顺序平衡；只有真实共同输入和区组才进行配对。
- 先计算 run-level 指标与 Type-1 分位数，再计算 paired difference、均值与 95% t-CI。
- 多基线共同优越性采用预登记 Holm 校正，家族错误率 0.05。
- 4,000 请求不能当作 4,000 独立重复。
- SLO 零容差非劣是严格主张判据，不是运行有效性门槛。
- 并列、未分辨、失败和无触发全部保留，不无限追加正式重复寻找显著性。

---

## 五、P0—P3：IEEE 对齐、强后端与自适应优化

### P0：主表与 Full 来源统一

历史 7B 主表 TTFT `563.9507 ms` 与消融 Full `545.667 ms` 来自不同运行。审计全部指标的：

- source run-set；
- 模型、后端、参数、输入；
- 生成合同、时间定义；
- 成本模型、analyzer 和图表来源。

处理原则：

- 同一主结果展示必须引用同一 canonical run-set。
- 合法独立配对重测、受控 activation、profiling 可以单独存在，但明确 case 和条件。
- 不改旧数字强行一致，不挑有利 Full 子集。
- 先交付来源对照表；发现统计问题离线修复，执行问题才重跑。

### P1：九个公式的实现合同

| 项目 | 不可改变的语义 |
|---|---|
| Tier | 最快有效副本；GPU 必须 backend-executable |
| Service estimate | \(\widehat S=\widehat D+\widehat T+\widehat O\)，按论文观测类更新 |
| Routing | 同快照可行集、服务时间分桶、\(Q_i\) 和稳定 tie-break |
| Benefit | \(F_i^k=h_a\Delta d_i^k\)，收益密度与真实 footprint |
| Handoff budget | 层级剩余预算、单最终目标、物理执行时 reservation |
| Residency objective | GPU→HOST→NVMe 条件性单层选择 |
| Residency constraint | 保守取整 DP，超限时原字节约束 greedy |
| Admission | 实际 KV、复用容量、压力及论文 \(E_i(t)\) |
| Physical capacity | used＋reserved 不超物理预算 |

正式 Full 禁用论文路由式未定义的 handoff 优先保护前缀，保留普通可行性和原子 reservation。

测试覆盖：

- 空集、边界、零需求；
- DP 小规模穷举；
- 并发 admission、取消、重复 transfer；
- 共享副本、引用、驱逐、backend invalidation；
- KV 不足、替换回滚、池复用、workspace；
- release、GPU-ready 发布顺序和 dispatch 快照。

不另开旧 B4 的定义写作任务，但实现必须遵守 IEEE 已明确的状态和 EWMA 语义。

### P2：新版 vLLM

当前候选为 vLLM 0.30.0；在独立环境中验证，不覆盖旧环境。[官方发布](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)

验证：

- SM86、宿主驱动、CUDA wheel、FP16、7B/3B；
- mixed-rank、target modules、动态 LoRA；
- CUDA Graph、批处理、取消和异常；
- 原生 token 与实际 GPU slot；
- eviction/registry 同步；
- warm TTFT、TPOT、吞吐、显存和启动；
- 后端常驻进程的真实释放。

选择最新兼容且经验证性能良好的版本；版本新不等于必然更快，回退必须记录证据。独立 vLLM 基线使用同版本及合理优化机会，不强行替换其他论文基线的专用后端，不自动升级宿主驱动。

### P3：优先可观测量驱动的自适应优化

#### P3.1 参数分类

| 类别 | 处理 |
|---|---|
| 论文明确定义的公式与配置语义 | 保持不变 |
| 硬件/安全上限 | 显式配置，不伪装成可学习量 |
| 公开实验协议：60秒、30秒、95%、75%等 | 固定，不随正式结果改变 |
| 可观测运行量 | 优先在线更新 |
| 模型特征与初始化 profile | 允许按 7B/3B 分别配置 |
| 无观测依据的 magic number | 逐项审计，优先删除或以实测推导代替 |

自适应不等于“所有常量都取消”。不新增一批未经推导的系数替代原来的手工参数。

#### P3.2 自适应输入与实现方向

优先使用：

- 当前队列、admitted/pending 数、原生 iteration token 预算；
- 实际 KV block、reserved/free、adapter slot、workspace；
- 完成事件、传输字节与时间、真实 footprint；
- 本服务资源域的内存压力和占用；
- 当前及历史请求的 prompt、declared output limit、已生成长度；
- 完成加载更新的分层准备成本；
- 论文指定窗口中的需求分布和完成请求统计。

保持论文 EWMA、窗口和分桶语义。允许改进观测、初始化 profile、执行效率和内部表示；不偷偷用另一种预测器替代行间公式或关联定义。

例如：

- 用真实分配量计算 footprint，避免按模型名称硬编码 adapter 大小；
- 用后端 block 事件恢复可用 KV，而非固定“预留若干 GB”；
- 用完成加载实测更新成本，不按本次测试带宽标签查人工延迟表；
- 用资源预算与实际任务 footprint 约束传输执行，不按正式测试点更换并发数；
- 需求、就绪、失效事件及时更新状态，避免无条件轮询或跨请求重复全量扫描。

主机全局 `MemAvailable` 主要用于外置安全保护，避免把其他用户压力变成 Prime 独享的隐蔽控制信号。

#### P3.3 按模型配置的边界

允许分别冻结 7B/3B 的：

- profile、观测类边界；
- backend batch/token/slot 配置；
- footprint、KV block 与 workspace 参数；
- 论文允许配置的窗口、bin width、控制上限。

每模型的同一工作点跨 W0/W1/W2、带宽和 churn 敏感性固定。在线状态可以自适应变化，**控制规律不能按场景名、run block 或正式结果切换**。

Full 与单因素消融共享非目标自适应逻辑，避免消融同时换控制策略。

#### P3.4 每轮优化必须完成的证据链

1. 主线阶段和未达标指标。
2. 历史实验、修改历史和失败尝试。
3. 联网核查的原始论文、公式、源码位置、版本和适用条件。
4. 一个可证伪的瓶颈假设。
5. 与 IEEE 合同的对应关系。
6. 最小正确性测试和验证比较。
7. 完整验证回放与副作用。
8. 图表、结论、接受/撤销/等待新证据。
9. 返回主线 tracker。

参考方向包括 dLoRA 的请求—adapter 协同、ELORA 的 LoRA/KV 管理和 vLLM 的批处理/内存实现；只借鉴与实际瓶颈相关的内容，不因论文名称相似直接移植。[dLoRA](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)、[ELORA](https://arxiv.org/abs/2505.03756)、[vLLM 优化文档](https://docs.vllm.ai/en/latest/configuration/optimization/)

一个假设默认最多两轮最小验证；没有新证据则归档、转向下一瓶颈。这是投入控制，不是“两次即可证明假设错误”的统计规则。

只维护一条候选优化主线，不堆叠未验证补丁。正式结果反馈开发后，新版本完整评测并保留旧版本身份，不能称为全新未见 workload。

---

## 六、baseline：Serverless 优先审计和公平复现

### 6.1 Serverless：本轮已经确认的证据

历史清洁结果：

| 模型 | 平均 Dispatch/Admission Wait | 平均 service TTFT | 完成吞吐 |
|---|---:|---:|---:|
| 7B | 236.727 s | 408.59 ms | 约 0.96 requests/s |
| 3B | 237.313 s | 498.50 ms | 约 0.96 requests/s |

本轮核对发现：

- 官方历史 commit `9f50241baa5386e06a9321c51f19a9ef5f964c2b` 的 router，每取出一个请求都会把 `instance_options` 置空，先等待一秒，再读取 ready 实例。
- 这不是“只有没有 ready 实例时才等待”，也不是本地 wrapper 新增的等待。
- 3B 原始日志中，相邻请求进入后端的间隔中位数约 **1.002 秒**。
- 当前证据强烈支持串行轮询是重要瓶颈，但尚未通过修复对照量化它解释了多少总体延迟。
- 不能继续把历史结果直接写成“四张 GPU 推理吞吐不足”或“Serverless 架构必然需要四分钟启动”。

官方源码可以直接核查该路径。[历史 commit 源码](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/routers/roundrobin_router.py)

### 6.2 Serverless 审计顺序

先离线审计，再决定重跑：

1. 核对历史实际导入源码、官方 SHA、本地兼容补丁、环境和启动命令。
2. 核对 clean 7B 与旧 7B 的区别，不混用被替代结果。
3. 重建 arrival、enqueue、allocation、backend-start、首 token、完成时间线。
4. 分离客户端提交、router 队列、实例启动、adapter 获取、backend 调度和推理。
5. 检查实际 ready 数、并发、队列容量、actor 限制、CPU/内存压力和远端获取。
6. 核对原生 checkpoint loader、格式、scheduler/migration 是否真的启用，不把普通加载路径等同于论文的快速加载路径。
7. 对照原论文的硬件、模型、初态和 latency 定义，不直接用多租户排队时间否定其启动数据。[ServerlessLLM 原论文入口](https://www.usenix.org/conference/osdi24/presentation/fu)

原生部署曾使用 min=1、max=4、keep_alive=10，7B target=2、3B target=8；这些是历史配置，不直接成为新冻结最优配置。

### 6.3 最小修复与对照

按你本轮确认：

- 优先采用可核验的官方修复。
- 无适用官方修复时，允许最小补丁：**先检查 ready；仅在不可分配时等待**。
- 保留原 round-robin 次序、queue capacity、autoscaler、加载与迁移策略。
- 不移植 Prime 的 readiness ranking、planner 或 admission。
- 不把等待简单改成另一个更小 magic number。
- 保留取消、并发容量检查和失败恢复。
- 公开官方 SHA、补丁 SHA、修改理由、资格测试及原版证据。

先做无需 GPU 的 ready/empty/full/cancel 多请求测试，再做两模型各一对 1,000 请求的原版/修复版开发对照。该小对照用于归因，不以单次运行宣称显著性。

随后修复版按完整资格和 M1/M2 进入正式重复。诊断对照计入开发工作，不新增第二个 Serverless 主矩阵行。

### 6.4 Serverless 最终判定规则

| 发现 | 处理 |
|---|---|
| analyzer 或汇总错误 | R1 离线重算 |
| wrapper、输入、token、启动或加载路径错误 | 修复后重跑受影响键 |
| 官方实现的可修复轮询问题 | 最小修复、公开前后证据，主比较采用合格修复路径 |
| 原生公开配置不合理 | 验证阶段合理调整并冻结 |
| 合同正确，修复后仍有真实排队瓶颈 | 如实报告，并以阶段证据解释 |
| 原机制在本机无法忠实启用 | 标注范围或资格失败，不声称完整复现 |

历史 warm-min4/wait90：

- 保留为诊断，不替代新的共同 60 秒准备协议。
- 受外部显存占用污染的 7B 运行不进入清洁主比较。
- “只改善约 0.8%”不能证明 GPU 已饱和，可能是未消除串行分配限制。
- 在共同准备与完整计费下，公开的常驻配置可以作为清楚标注的工作点；不冒称它验证了 elasticity。
- 不通过隐藏 warm 配置获得免费准备。

### 6.5 名称规范

- 所有新图例、坐标、表格显示名称：**Serverless**。
- 数据内部保留 `serverlessllm_new`、版本、补丁、环境和工作点。
- 图注/方法文档说明 `Serverless` 对应 ServerlessLLM 的被测实现。
- 原版与修复版诊断用“修复前/修复后”表示，不使用 `-new`。
- 主表不同时摆放旧、新两个同系统行。

### 6.6 其余 baseline

**vLLM：**随 P2 建立同后端参照、warm reference 和 Resident 预算参考。

**S-LoRA：**优先审计 native SSE、prompt、adapter identity、固定输出和流式完成事件；能恢复的旧计量先重算。

**dLoRA：**

- 优先复用成功的 3B DP2/TP1 路径。
- `migration_type=3` 对应 PERIOD_MIG，但还需实际迁移事件证明。
- 不先重复四 worker 已知高内存失败。
- 7B 只有出现实质新条件时重新资格，不把既有失败泛化为所有部署都不可行。

**HydraServe：**

- 保留官方修改后的 vLLM 0.4.2、启动重叠、pipeline 与 consolidation。
- 允许构建、API、路径、模型注册和非原云环境适配。
- 先核查 Kubernetes/GPU 容器资源控制和 per-request adapter 身份，再启动大回放。
- 验证 pipeline/consolidation 前后的 LoRA 正确性和所有 worker 计费。
- 本地 backbone、真实远端 LoRA 的场景限制必须说明。
- 不触发的机制不伪造触发；不替换其核心策略。[官方实现](https://github.com/LLMServe/hydraserve)

无法忠实运行时依次核查 ServerlessLoRA、BlitzScale 官方路径；资格失败记录保留，不能用自制框架冒名替代。

**Loquetier：**

- 从已有小规模成功 gate 出发，先查重复物化、实际 footprint 和误加载训练状态。
- 原生实例采用 1/2/3/4 卡静态分片，按真实字节与验证需求平衡。
- 分片和允许复制在验证期冻结。
- 前端只做静态映射，不加入 Prime 策略。
- 保留原生模块、SMLM 和异步批处理。
- 128/256 adapter 成功不能替代 500-adapter 完整资格。[论文入口](https://arxiv.org/abs/2511.00101)

### 6.7 公共资格和 B6

每模型依次：

**100 请求 smoke → 500-adapter 功能覆盖 → 1,000 请求兼容 → 4,000 请求回放。**

完整替代要求两模型和完整池；单模型成功可以独立报告。

B6 固定工作点 `dLoRA_B6_resource_opt`：

- 仅在 3B W1/W2 验证；
- 全部正确、联合 SLO 达标后，最小化两场景等权 GPU-s/request；
- 不可行使用 M1 诊断规则；
- W1/W2 各五正式块，共十次；
- 对比 Prime/ElasticLocality 的 M1 工作点；
- 公开 dLoRA 选择域较窄，不能推断 W0；
- 可以离线检查其预算合规性，但不能称已经优化 G2 工作点。

---

## 七、M1—M3：主比较与持续优化闭环

### M1：共同 SLO 下最少 GPU 时间

验证可行配置要求所有目标场景和验证重复均：

\[
N_{\rm correct}=N,\qquad A_{\rm joint}\ge0.95.
\]

在可行集中最小化 W0/W1/W2 等权 GPU-s/request；平局以归一化 P95 TTFT、稳定配置 ID 决胜。

每系统、每模型冻结一个跨三场景的 `resource_opt`。

无可行配置时：

1. 有全部正确完成配置：最大化最差验证块 SLO，再最小化平均 GPU-s/offered-request。
2. 无全部正确配置但存在合法执行：先最大化最差完成比例，再最大化最差 SLO，再比较 GPU-s/offered-request 和固定配置 ID。
3. 无合法路径：`qualification_failure`。

前两类标 `diagnostic_nonfeasible`，不进入达标资源排名。

正式矩阵：

\[
7\ \mathrm{systems}\times2\ \mathrm{models}
\times3\ \mathrm{workloads}\times5\ \mathrm{blocks}
=210\ \mathrm{slots}.
\]

报告正确数、失败、联合 SLO、GPU-s、P95/P99 TTFT、TPOT、E2E、吞吐与资源压力。

### M2：共同预算和 SLO 下尾延迟

Resident-vLLM：

- 相同最终后端，四卡常驻；
- 普通原生 cache，完整池可访问；
- 无 Prime planner/handoff/admission；
- 相同准备窗口、CPU/内存包络和生命周期；
- 每模型、每场景三次完整正确验证。

\[
U_{\mu,w}^{ref}
=\frac13\sum_jU_{\mu,w,j}^{Resident},
\qquad
B_{\mu,w}=0.75U_{\mu,w}^{ref}.
\]

不从短回放线性外推，不用失败或不完整参考定标。

\[
\mathcal X_s=
\{x:N_{\rm correct}=N,\ U\le B_{\mu,w},
A_{\rm joint}\ge0.95,\ \forall w,j\}.
\]

在 W1/W2 中最小化参考归一化 P95 TTFT 的等权平均。峰值四卡、共同 CPU/HOST 包络同时成立。

\[
7\times2\times2\times5=140\ \mathrm{slots}.
\]

- 每模型冻结一个 `budget_latency_opt`。
- 相同执行键与 M1 复用。
- 报绝对 GPU-s、预算比例和平均 GPU 数。
- 共同预算是共同上限，不是强制实际消费完全相等。
- 无可行配置时仅报告合法诊断点或不可行，不填零 SLO、无穷 TTFT。
- 不拼接 M1 的成本与 M2 的延迟。

### M3：比较后优化

Prime 未达到 G1/G2 时：

1. 检查共同合同、正确性和指标。
2. 看完整阶段分解、负载与资源图。
3. 回开发期按 P3 优化。
4. baseline 获得合理公开配置和确认错误修复。
5. 冻结新版本，完整评估全部规定块。
6. 保留旧结果，不筛选赢家。
7. 不能改变 60 秒、30 秒、95%、75% 等协议追随正式胜负。
8. 无合法改进路径时报告实际边界，不把跑完写成最优。

---

## 八、A1—A5：核心消融、状态传播与 first-service

### A1：主 case 与十一配置

主 case：

`A1-W1-NATURAL-CLOSEDLOOP`

共享 Full `resource_opt`、共同初态、监控、外部资源包络和非目标自适应逻辑。

| 变体 | 唯一目标差异 |
|---|---|
| Full | IEEE 对齐全系统 |
| ElasticOnly | 同 autoscaler/backend，load-only，关闭主动 adapter 管理 |
| ElasticLocality | 二元 GPU locality、负载回退、原生 cache 与普通制品 LRU |
| FixedFull | Full 机制，四实例常驻 |
| LastKnown | 只改变 router 所见 tier 的新鲜程度 |
| NoRouting | 可行集/预留不变，改为 live load 排序 |
| NoHandoff | 禁止 activating replica 的 ready 前主动准备 |
| R0H0 | NoRouting＋NoHandoff，hierarchy/admission 保留 |
| DelayedHandoff | handoff 延后至 engine-ready 执行 |
| NoHOST | 关闭受管 HOST 复用，GPU/NVMe 预算不变 |
| CapacityOnly | 关闭主动 admission 软策略，保留物理安全 |

ElasticLocality 是强内部对照，不冒充单因素消融：

- 可接纳 GPU-hit 副本优先最轻载，超过 load-slack 回全局最轻载。
- 验证候选 `{0,1,2}`；最优持续处于上端点时，仅扩展一次到合法 `{4,8}`。
- 不使用 \(D+T+O\)、收益 planner、handoff 或有效容量。
- 与 Full 相同 autoscaler 和实例范围。

NoRouting 保留 admission/reservation、handoff、hierarchy、物理可行性，只替换目标排序。

### A2：handoff、延迟执行和四格

准备事件至少保存：

`trigger_reason, activation_id, plan_id, target_replica, adapter, tier, start_ts, ready_ts`。

NoHandoff 检查所有入口，包括 startup refresh、auto-warmup、HOST promotion、GPU forwarding。保留请求驱动加载、ready 后稳态规划和已有副本合法准备；共享 HOST/NVMe 自然复用，不人为清空。

Delayed 受控比较冻结相同 plan hash、目标、顺序和预算，只延后开始。记录旁路和请求抢先完成。自然闭环中反馈改变计划时，不声称目标完全相同。

四格：

- \(Y_{00}\)：R0H0；
- \(Y_{10}\)：NoHandoff；
- \(Y_{01}\)：NoRouting；
- \(Y_{11}\)：Full。

低优指标：

\[
\Gamma_j=Y_{10,j}+Y_{01,j}-Y_{00,j}-Y_{11,j}.
\]

逐匹配块计算，再求区间；高优指标反向。Full 最低不自动证明协同。

### A3：confirmed-state propagation

三类证据：

- **A/A：**同不可变快照、tie-break、随机状态；shadow 不重复更新状态。
- **Live：**实际执行并测服务结果。
- **Shadow：**相同实际状态下比较信息视图的决策，只执行一路，不生成反事实 TTFT。

LastKnown 只陈旧化 tier 及派生成本。可行性、queue、pending、GPU 利用率、EWMA、权威 registry、planner/admission 和安全校验保持 live。

主周期按最终实际同步 cadence 冻结；没有原生周期时明确这是替代设计。周期扫描放 S13。

报告状态年龄、false-ready、missed-ready、决策分歧和实际反馈。

### A4：activation 与 first-service

互斥分类：

- `initial`：共同部署通知或初始池触发，即使 \(t=0\) 后才 ready 也不改名；
- `natural_scaleout`：业务内在线控制自然触发；
- `controlled`：预登记诊断指令触发。

每个 epoch 的 first-service 绑定第一条实际 dispatch：

\[
r_v^*=\arg\min_{r\in\mathcal R_v}d_r.
\]

分别报告：

\[
T_v^{engine}=t_v^{ready}-t_v^{trigger},
\]

\[
TTFT_v^{first}=f_{r_v^*}-a_{r_v^*},
\]

\[
T_v^{activation\to token}=f_{r_v^*}-t_v^{trigger}.
\]

不把三个量混称启动延迟。失败的第一条 dispatch 不用另一请求或另一副本成功 token 补齐。

分组报告 epoch 总数、有 dispatch 数、有首 token 数、unused rate；没有 epoch 时为 N/A。

自然诊断保留 Full、ElasticOnly、CapacityOnly、NoRouting、NoHandoff、R0H0、Delayed：

- 先三个预指定块；
- 任一变体的 natural-scaleout 有 dispatch epoch 不足 20，则全体补至五块；
- initial/controlled 不用于凑数；
- 五块后停止，有效首 token 样本不足则描述性报告。

受控 A4 保留 8-phase 诊断。指令时序固定，真实 ready 时间照实记录；Full/Delayed 冻结同准备目标，无 handoff 对照不强加准备任务。

tier 字段应覆盖全部实际 dispatch，快照在 reserve 后、resolve 前冻结；不变量冲突必须为零。缺字段、失败和未达到样本门槛分别报告。

### A5：累积消融与运行数量

累积链：

`ElasticOnly → RoutingOnly → RoutingHandoff → CapacityOnly → Full`

- 新增两个 7B 中间节点，各五块。
- 7B 十一配置各五块。
- 3B 十一配置各三个预指定块。
- 同条件 Full/ElasticLocality 复用 M1。
- 不用五块 Full 与三块变体伪配对。

条件性新增：

\[
11\times5+11\times3-16+2\times5=82.
\]

82 不含受控 A4、额外 W2 LastKnown、独立 Full 或详细 profiling；合同不匹配时不能扣除对应复用。

NoHOST 单独记录 staging/page cache/CPU tensor。CapacityOnly 保持 victim selection、replacement 顺序和物理保护一致。

主四面板：

- GPU-s/request；
- 联合 SLO；
- P95 TTFT；
- P95 TPOT。

补充四面板保留 P95 TTFT、平均 E2E、Cost/request、CE。完整表含平均 TTFT、吞吐、字节、机制触发和资源事件。

相对变化：

\[
\text{lower-better}=\frac{reference-value}{reference},
\quad
\text{higher-better}=\frac{value-reference}{reference}.
\]

未触发不能归因；未分辨和 trade-off 如实呈现。

---

## 九、S1—S3：必要 motivation 与机制证据

| 实验 | 设置 | 判断 |
|---|---|---|
| S1 四层成本 | 两模型，各按 footprint/rank 分层选 24 个既有 adapter，四 tier、三轮交错 | 是否存在足够大的实际层级成本差 |
| S2 副本错配 | 请求、adapter 与总缓存资源固定，分别改变实际副本位置和信息视图 | 区分缓存有没有、缓存在哪里、router 是否知道 |
| S3 加载争用 | 稳定 decode 下准备/按需加载，三轮 | admission 是否减少实际干扰，而非仅推迟工作 |

S3：

- Full/CapacityOnly 收到相同准备候选、初态、目标和触发时间，由各自 admission 决定执行/延后。
- ElasticLocality 只接收同一请求，保持原生按需加载，不注入主动 promotion。
- 记录 candidate/admitted/deferred/completed 数量和字节。
- 覆盖后续请求承担的延期成本，不能只展示局部 TPOT 改善。

共同要求：

- NVMe 命中 page cache 不称物理盘读取。
- 不整机 drop caches。
- SSE chunk 间隔不冒称 kernel 逐 token 时间。
- KV 压力用实际 block、stall 和 preemption 判断。
- 层级不是主要瓶颈时收窄 motivation，不注入 sleep 制造结论。
- 记录 GPU topology、PCIe/NVLink/NCCL、网络、磁盘、CPU、H2D、prefill/decode。
- 没有跨卡流量可以说明所测机制不依赖该流量，不能外推 H100 性能。

---

## 十、S4—S13：敏感性和补充实验

这些实验在主比较、核心消融和必要机制证据之后执行，不抢占主线资源。

| 编号 | 设置 | 系统与重复 |
|---|---|---|
| S4 到达强度 | 0.5/1/2/4/8 倍；3B 取 1/4/8 | 7B 七系统；3B 同七系统锚点，三块 |
| S5 GPU 规模 | 1/2/4 展示点；主配置选择仍允许 3 卡 | 7B 七系统，三块 |
| S6 带宽 | 0.1/0.25/0.5/1 Gbit/s、原生不限额 | 7B Full、ElasticOnly、ElasticLocality、Serverless、HydraServe，各三块；3B 低带宽/1G/原生锚点 |
| S7 分布/churn | stationary、rotation100/500/2000、Zipf0.6/1.4、gradual500且50% overlap | 同上五系统，先一块；stationary/100/500 补至三块 |
| S8 到达代表性 | 既有公开数据的三个预登记 CoV 类别 | 7B Full、ElasticLocality、Serverless、HydraServe，三块 |
| S9 长度/KV | cap64/128/256、既有长请求尾部 | 两模型 Full、CapacityOnly、ElasticLocality，三块 |
| S10 控制开销 | 100/250/500 候选 | 两模型三轮，CPU/内存/P50/P95/fallback |
| S11 后端归因 | 旧/新 vLLM、同生成合同 | 两模型 warm 和 Full 配对，三块 |
| S12 自然停止 | 同 prompt、EOS/stop | 两模型 Prime、S-LoRA、Loquetier，三块 |
| S13 设计补充 | shadow周期0.1/0.25/0.5秒；预算0.5/0.75/1；HOST预算转NVMe | 能离线计算则复用，实际改变配置则三块 |

补充规则：

- S4 保留过载失败；另用固定 wall-clock 热点表区分 arrival 加速与按请求轮换的耦合。
- S5 完整 500 池不能运行则标资源不可行，不缩池掩盖。
- S7 报实际 unique、独立权重 SHA 数、effective count、entropy/Gini、first-touch、reuse-distance、Jaccard turnover 和 remote miss。
- S8 CoV 使用固定 60 秒到达计数的标准差/均值；按来源和时间选窗口，不按胜负挑选。缺类别不造数据，重建到达注明 trace-driven replay。
- S9 不造长 prompt，报告原始/执行长度与截断比例。
- S12 不要求 actual==cap；输出长度不同不能直接进行 E2E 系统因果归因。
- S13 的 HOST→NVMe 是另一种预算分配设计，不混入 NoHOST 主消融。
- 前置相同执行键复用。
- 旧 pool-size 100–500 图保留为历史，不充当新后端或真实远程证据。
- 单机四卡不写成多计算节点 scale-out 证据。

### 10.1 带宽接口与微测

应用层聚合速率：

| Gbit/s | MiB/s |
|---:|---:|
| 0.1 | 11.9209 |
| 0.25 | 29.8023 |
| 0.5 | 59.6046 |
| 1.0 | 119.2093 |

- 复用 fetcher，聚合本次服务所有 worker/进程，不各自拥有完整额度。
- 新接口明确为 `FAASLORA_STORAGE_BANDWIDTH_MIB_S`；旧 `_MBPS` 仅兼容。
- 使用线上实际字节计量；解包字节另报。
- 四并发、小/中/大既有 adapter、三重复，无需 GPU。
- 检查总 wall time 与总字节/额度；burst allowance 显式扣除和记录。
- 记录 configured/achieved throughput、bytes 和 injected wait。
- 1G 限额不代表实际达到 1G。
- 不把 250 MiB/s 当真实 1GbE 点。
- no-delay local-sim 仅作路径诊断，不称 100G 或严格性能上界。

### 10.2 离线成本、CE 与 SLO

从有效旧日志和新日志分别计算，不混合同：

- 互斥时间窗 GPU-s、重叠活动；
- GPU-s/request；
- cost/1M output tokens、cost/1M total tokens；
- SLO goodput、good requests/$；
- cost–latency、resource–SLO Pareto；
- 90/95/99% 达成要求及 TTFT/TPOT 倍率敏感性；
- 原 CE 及对数分解。

保留原 CE 的延迟定义和单位；按既有 E2E 协议：

\[
CE=\frac1{LC},
\qquad
\ln\frac{CE_P}{CE_B}
=\ln\frac{L_B}{L_P}+\ln\frac{C_B}{C_P}.
\]

广义 CE 使用共同参考归一化：

\[
CE_{\alpha,\beta}
=\left(\frac L{L_0}\right)^{-\alpha}
 \left(\frac C{C_0}\right)^{-\beta},
\quad
\alpha,\beta\in\{0.5,1,2\}.
\]

idle factor 为 `{0,0.238095,0.5,0.75,1}`；价格比、成本 break-even、CE break-even分别求解。

统一 GPU 单价整体缩放不改变 GPU 时间排名。idle 折扣不等于释放物理 GPU。

只改变离线评价无需重跑；若改变在线策略或选了新配置，则登记新执行键，不能换标签冒充重新优化。

---

## 十一、每个实验的图表与学术绘图规范

### 11.1 交付节奏

使用 `academic-plotting` 的数据图流程，复用现有绘图脚本和 provenance 校验，不重建一套平行框架。

每次运行结束：

1. 完成清理和结果校验。
2. 立即生成单 run 诊断图或状态表。
3. 更新实验组的累计预览。
4. 写明目前重复数、是否 provisional、失败及缺失状态。
5. 给出当前解释和下一实验。
6. 再启动下一任务。

实验组完成后生成正式图/表、CI、数据 CSV/JSON 和来源 manifest。

安全中止或资格失败可用状态表，不为每次失败强造无意义性能图。预览可包含 incomplete 状态，但必须显著标注；正式排名拒绝把 incomplete 当完整性能点。

### 11.2 图还是表：固定选择规则

| 实验目的 | 首选表达 |
|---|---|
| 精确比较多个系统与多指标 | 主表，分模型/工作点，避免七系统密集柱图 |
| G1：达标时减少 GPU 时间 | 带达标标记的 GPU-s 点图＋资源–SLO 图 |
| G2：共同预算下尾延迟 | P95 TTFT 点图/CI，附预算与 SLO 合规 |
| Serverless 等待原因 | 请求序号/时间–排队曲线、分配间隔 ECDF、阶段分解 |
| 累积消融 | 顺序点图/绝对值面板＋完整表 |
| 四格协同 | 四格结果＋interaction estimate CI |
| confirmed state | 状态年龄/分歧图、false-ready/missed-ready 表与真实服务指标 |
| first-service | activation 时间线、分组分布和样本数 |
| 分层准备成本 | 分阶段点图或可加和堆叠条 |
| 加载争用 | 同步时间线与 TPOT/KV/加载事件 |
| 速率/GPU/带宽敏感性 | 折线＋区间 |
| churn/工作负载 | workload 描述表＋性能趋势 |
| CE/价格稳健性 | break-even 曲线、排名热图、Pareto |
| 正确性、资源失败、资格 | 表格，不伪造连续性能点 |

所有图围绕该节要证明的问题组织。主表保留完整比较，主图强调与贡献相符的资源–服务质量关系。

### 11.3 单栏尺寸与字体

- 默认最终宽度 **3.45 inch，约 87.6 mm**，按目标 IEEE 实际 `\columnwidth` 校核。
- 不先生成双栏大图再缩小到单栏。
- 普通单面板高度约 2.2–2.8 inch；根据数据与布局增加高度或拆图，不缩小字体硬塞。
- 四面板采用单栏分组/纵向排布；2×2 只有在最终尺寸仍清晰时采用。
- 全部拉丁文字、数字、图例、坐标和子图标题使用 **Times New Roman**。
- 数学字符使用 Times 系字体可覆盖的字形；正式输出检查缺字和字体替代。
- 默认轴标题 10.5 pt，tick/legend 9.5 pt，注释至少 9 pt，子图标题 10.5 pt 加粗。
- 在无碰撞前提下尽可能增大字体；密度过高先拆图，不用极小字解决。
- 新投稿图不得静默 fallback 到 DejaVu/Arial；字体缺失先修复环境。
- PDF 嵌入字体，PNG 至少 300 DPI；投稿优先矢量 PDF。

### 11.4 版面、图例和子图标题

- 子图标题采用 **`(a) 标题`、`(b) 标题`**，加粗，放在对应子图下方，与子图居中对齐。
- 不使用上方 `(a)` 与下方标题分离的模板默认格式。
- 图例优先一行；在单栏真实宽度和字号下放不下时，采用直接标注、拆图或必要的第二行，不压缩成不可读。
- 七个长系统名不强塞一行；主比较优先表格或直接标注点图。
- 共享图例靠近绘图区，默认间距 2–4 pt；子图间距由 label/标题实际包围盒决定，尽量缩小但保持分隔。
- 外边距尽量小；导出后检查最终页面宽度，避免 tight-bbox 改变单栏设计尺寸。
- 不让 legend、数值、CI、轴标题、tick 和子图标题互相覆盖。
- 数据曲线本身相交属于真实关系，不用移动数据解决。
- 少量浅色主网格，不使用 3D、阴影装饰、雷达图掩盖定量差异。

### 11.5 配色与 Prime 突出方式

使用低装饰、色盲友好的专业配色，优先 Okabe–Ito 及其经过可读性检查的组合。

- 主比较中系统颜色、marker 和线型固定。
- Prime 使用突出且稳定的暖色、稍粗线条/明确 marker。
- 其他系统保持足够对比，不把关键竞争者淡化到难以辨认。
- 不同实验家族采用不同配色结构：主比较为系统色，消融为机制分组色，成本为阶段色，状态传播为状态色，敏感性使用一致系统色加不同辅助色。
- 同一颜色在同一实验组中的含义不改变。
- 跨组改变语义时提供清晰图例，不随机换色。
- 同时用 marker、线型或纹理，验证灰度和色觉缺陷下可辨认。

### 11.6 诚实展示与视觉验收

允许通过选取与贡献对应的图型突出真实优势，但必须：

- 完整保留预登记比较点、失败和退化。
- 柱图零起点；跨度大时优先点图/log 轴，明确刻度。
- 不用未标注断轴或不同面板隐蔽尺度放大优势。
- 标注 lower/higher-is-better、单位、\(n\)、CI 定义。
- 只在实际数值支持时标最优；统计未分辨不能用“显著领先”注释。
- 两工作点不混画成同一配置。
- 多阶段堆叠只用于真正可加和项，重叠活动采用时间线。

每张正式图保存：

- 输入数据与 SHA；
- run-set 和分析键；
- 脚本/样式版本；
- figure specification、配色和字体；
- PDF/PNG；
- 对应数据表；
- 单栏实际尺寸预览和 QA 记录。

自动检查包围盒碰撞、裁切、缺字、字体嵌入、尺寸、标签映射；再查看渲染图确认无视觉问题。图表名中不出现 `Serverless-new`。

---

## 十二、两轮审稿意见的证据闭环与文档

| 质疑 | 处理 |
|---|---|
| 收益主要来自 elasticity | FixedFull、ElasticOnly、ElasticLocality、四格与累积消融 |
| confirmed state 未隔离 | A/A、Live、Shadow、LastKnown、S2 |
| Fig.9 解读或 Table1/Full 不一致 | P0、canonical run-set、正确符号和 run-level CI |
| CE/idle 系数主导 | G1/G2、原始延迟、Pareto、价格与 idle 敏感性 |
| TTFT 明显退化未解释 | 共同 SLO、预算比较、完整阶段分解与失败点 |
| Serverless 四分钟等待 | 优先原始日志、官方代码、最小修复与对照 |
| 缺最近邻 | dLoRA 有限性能、其他官方制品与障碍审计 |
| 算法常规、目标缺推导 | 解释需求加权准备时间近似、可分离条件和失效范围 |
| budget/位置/动态性/最优性 | 公式合同、owner/registry 文档、动态重检查 |
| queue/affinity、active feasibility、admission | C1/C2/C3 具体文档与实现证据 |
| 500 adapters 与轮换代表性 | 权重身份审计、S7/S8、实际 unique/churn |
| Prime/S-LoRA E2E 不公平 | matched-output、原生 token、阶段分解和 S12 |
| readiness 不是 dispatch-time | 精确快照、epoch/dispatch 绑定及 A4 分组 |
| 小规模、低速率 | S4/S5 的真实压力与 GPU 规模，不冒称大型集群 |
| 3090/网络局限 | 实际 profiling、真实 remote、S6，不外推未测硬件 |
| 单次微小提升证据不足 | 独立运行块、CI、多重比较与未分辨结果 |

文档交付：

- B1/B2 场景表、budget owner、tier registry、routing/handoff/residency/admission 伪代码；
- C1 每个 queue/affinity 字段的含义、owner、存储、更新、过期与恢复；
- C2 active-LoRA feasibility，与 residency、batch-adapter limit、queue capacity 区分；
- C3 planner 慢时间尺度选候选、admission 快时间尺度按当前资源重检查的原因和示例；
- 需求加权准备时间目标的近似解释、逐层可分离的条件和失效场景；
- Fig.3 新颖性说明：既有机制与本系统协调关系分开；
- baseline 官方机制、适配、最小修复和不可复现障碍。

每份设计文档分“论文规范语义”和“当前实现证据”两栏。预算可行不等于全局最优；标准背包不包装成独创新算法。

不能在现有硬件条件下消除大型现代 GPU 集群证据不足这一限制，论文应如实限定适用范围。

---

## 十三、接口、测试、数量与最终交付

### 13.1 现有框架的最小必要扩展

复用 runner、fetcher、analyzer、绘图、scope 和清理入口，增加：

- 真实服务资源域、外置 watchdog、整树清理；
- 远端认证和 `status/start/stop/restart/health` 管理；
- R0–R3 逐指标复用审计；
- arrival/control/init/profile 身份；
- 原生 token、请求及 attempt/dispatch ID；
- activation 类别、plan hash、准备原因和旁路检查；
- allocation/release、互斥窗口和活动重叠；
- 跨进程 aggregate bandwidth；
- M1/M2 不可行诊断 selector；
- `Serverless` 显示名映射，内部身份不改；
- 逐实验图表回调、显式 publish directory、严格来源校验；
- 压缩清理回执、图表 QA 和执行状态。

区分：

- `slo_runtime_hash`：影响在线行为的 SLO、超时与控制配置；
- `slo_eval_hash`：只影响离线评价。

execution key 包括实际代码、环境、输入、生成、初态、资源限制、runtime 策略、监控和运行块。analysis key 包括纯离线统计、价格、显示名和图表样式。

### 13.2 启动正式实验前的测试

必须通过：

1. GPU 重叠活动不重复计费。
2. initial/natural/controlled 与 first-dispatch 正确。
3. M1 空可行集、零完成、全部失败不产生伪优胜。
4. B6 选择域和工作点不混用。
5. backend 变慢时 open-loop 到达不被重新排队生成。
6. S3 不向 ElasticLocality 注入主动策略，并保留延期工作。
7. 小型无 GPU cgroup 测试验证限制、子进程继承、事件和整组清理，不耗尽主机。
8. 容器/Ray 实际 worker 限制与总量可读回。
9. watchdog 独立、只清理本实验进程。
10. 截断 run 的分母、状态和资源正确。
11. Serverless ready/empty/full/cancel 路径无无条件每请求等待，原调度语义保留。
12. 远端免交互启动/停止/重启和真实下载通过。
13. 图表单栏宽度、Times New Roman、下方加粗子标题、无遮挡和 Serverless 名称检查通过。

### 13.3 正式验收

- 前后 GPU idle/cleanup gate；
- 无无关 GPU 作业污染；
- 完整 offered 集合，达标比较 4000/4000 正确；
- 输入、adapter、prompt、target、generation hash 对齐；
- 无 fallback，原生 token 符合对应合同；
- 时间分解和 GPU 积分正确；
- tier、引用、reservation 和物理容量无冲突；
- actual dispatch 字段完整；
- 远程来源与资源生命周期可追溯；
- high/max/OOM、CPU 节流、spill/retry 可解释；
- 失败、安全中止、不达标不丢弃；
- 标准监控低扰动，详细 profiling 单独执行；
- 图或表已生成并检查，结论已回写 tracker。

### 13.4 数量与排程

默认核心记账小计保留：

\[
210+140+82+10=442.
\]

它不是最终独立执行总数：82 已扣部分复用，M1/M2 尚待精确去重。

额外单独登记：

- 安全自测、资格、开发 pilot；
- Serverless 最小修复开发对照；
- warm SLO 标定与 Resident 参考；
- 受控 A4；
- 3B 自然 A4 从三块补五块：七变体最多 14 槽，精确复用 Full 两块时新增 12；
- 独立 Full、额外 W2、profiling；
- S1–S13；
- 如另行选 B6 的 G2 配置，另计，不默认增加。

registry 分别统计：

- planned execution keys；
- total attempts；
- valid completed runs；
- failed/safety-stopped/not-run；
- 历史直接复用数和离线重算数。

\[
GPU\text{-hours}=\sum U/3600.
\]

同时记录 wall time、机器独占时间、磁盘新增峰值和内存峰值。先 pilot 再估算耗时，不用旧运行时间承诺完成日期。

### 13.5 目录、发布与持久执行规范

工作目录：

`/home/qhq/serverless_llm_experiment_retry14_baseline`

分支：

`retry14_continuous_queue_v2`

产物：

```text
docs/ieee_tc/
results/ieee_tc/<campaign>/<model>/<run_block>/<run_tag>/
paper_results/ieee_tc/<experiment_group>/
figs/ieee_tc/<experiment_group>/
```

交付包括：

- 本完整计划、执行规范与主线 tracker；
- 两轮审稿证据矩阵；
- 公式—实现—测试映射；
- 自适应变量、模型配置和优化记录；
- Serverless 审计、修复证据及版本说明；
- 各 baseline 资格、适配、失败和历史复用表；
- B1/B2、C1/C2/C3 文档；
- 冻结配置、环境和输入 SHA；
- 请求、资源、失败、清理回执；
- 逐实验图表、数据和 QA；
- 非敏感远端自动管理说明。

实施开始时将本计划及所有非敏感优化/绘图规范写入项目持久文档和记忆；每轮启动先读取，不能只依赖对话上下文。

保护 dirty/untracked 用户文件，尤其不 stage：

`configs/generated/lora_manifest_1000.json`

旧 `final_v2/`、旧 `figs/paper/` 和原始证据前后做 hash 清单，零覆盖。原始大日志不入 Git；文档、脚本、curated CSV/JSON/PDF/PNG 经 diff/checksum/test/smoke/secrets 检查后推送：

`faaslora_origin/retry14_continuous_queue_v2`

baseline harness 独立提交，在 manifest 记录对应版本。

明确不做：13B 新实验、SGLang 新报告、伪造 H100/高速网络结果、非官方冒名复现、论文源文件修改、历史 commit 修复、按胜负筛选结果、明文凭据入库、删除唯一原始证据。

本版使用 `experiment-plan` 保持主线与证据对应，使用 `ablation-planner` 复核消融和资源合同，并按你指定的 `academic-plotting` 固定逐实验学术图表交付规范。
