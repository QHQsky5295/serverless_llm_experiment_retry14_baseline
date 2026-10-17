# D662：3B HOST/GPU 请求路径的保留回执分解

2026-10-18。复用 D660 的完整 W0 和 D661 的共同 SLO 投影；无新推理、
生产代码/配置修改、阈值调整或图表。主线仍为 3B W0，尚未达到联合 SLO。
结果分析技能用于区分观察、解释和后续验证；GitHub 同步技能用于保护性备份。

## 观察：不是把整个 HOST 路径都算作加载

D660 的 4000 条原生请求合同通过，联合 3575/4000；独立数值正确性未知，
`n_correct=null`、`ranking_eligible=false` 保留。下列是同一次运行的条件分组，
并非随机化 tier 对照，也不是可以直接从 TTFT 中扣除的反事实收益。

| 实际选择来源 | 请求数 | 联合达标 | 最后实际加载数 | 加载均值 ms | acquired→engine 均值 ms |
|---|---:|---:|---:|---:|---:|
| GPU 原生副本 | 1875 | 1748 | 0 | N/A | 93.927* |
| HOST 原生副本 | 1441 | 1281 | 1426 | 15.680 | 245.599 |
| 本地文件 NVMe | 552 | 496 | 548 | 40.098 | 197.123 |
| 真实远端 | 132 | 50 | 131 | 44.057 | 166.204 |

\* GPU 引用在 controller source-admission 前取得：42.401 ms 在 admission 前，
51.526 ms 在 admission 后。不能只列后者并称为全部 acquired→engine。
加载数少于来源数是等待中合法复用/重检查的情况，不能将无加载项填成假测量。
远端行的最后 native load 不包括真实传输；完整 remote admission→engine
仍为 2122.797 ms，没有删除传输。

严格同钟、PID、各自 owner、HOST lease、adapter 和时间顺序校验后，HOST
1441/1441、GPU 1875/1875 的链条完整；加和误差小于 0.001 ms。
pending 的 scheduler owner 与 residency owner 是不同身份，不能要求相等。
HOST 保护到 GPU 加载期间 owner epoch 可以推进；校验具体来源和租约，而非
要求 epoch 不变。原始请求/结果没有修改。

| HOST 同钟相邻事件 | 覆盖 n | 均值 ms | Type-1 P95 ms |
|---|---:|---:|---:|
| 全局许可恢复→pending 登记观察 | 1441 | 182.897 | 641.666 |
| pending 登记观察→HOST 引用取得 | 1441 | 75.998 | 365.285 |
| HOST 引用取得→source admission | 1441 | 16.468 | 47.890 |
| admission→最后一次加载开始 | 1426 | 128.653 | 298.415 |
| 最后一次加载开始→GPU 引用取得 | 1426 | 15.680 | 17.433 |
| GPU 引用取得→post-load source snapshot | 1441 | 83.012 | 111.705 |
| post-load source snapshot→engine dispatch | 1441 | 162.587 | 528.455 |

另外 15 条 HOST 最后没有加载，admission→acquired 均值 651.680 ms，单列保留，
不与 1426 条强行配成一条平均堆叠。HOST 的 15 条 stale-recheck=1，其余 1426
为 0。114 条 HOST TTFT 违约中，最后加载均值 15.931 ms（109 条），
acquired→snapshot 81.756 ms，snapshot→dispatch 461.988 ms。

GPU 路径：resume→pending 148.866 ms，pending→acquired 89.340 ms，
acquired→snapshot 1.106 ms，snapshot→admission 41.295 ms，
admission→dispatch 51.526 ms。不能据此断言 GPU-hit 无控制开销。

RPC 记录另作非加和说明：HOST pending/hold/acquire/release 的 wait-response
均值分别为 94.877/53.361/255.477/75.713 ms，channel acquire 分别
3.194/6.268/3.101/7.428 ms。它们与上述事件链重叠；不再次相加，
不把 RPC wait 当成纯 socket 或 CPU 计算时间。HOST-release 没有独立单调钟
完成时间戳，故目前不能把 snapshot→dispatch 精确分成所有互斥内部阶段。

## 解释、源码和历史边界

现有 `IEEEBackendGPUReferences._load_and_acquire` 在 acquired 时间之后进行
HOST allocation 后检查，随后生成 post-load identity snapshot。原生 HOST
加载走 fresh inventory；D352 仅对已经确认的 GPU-hit 事务去掉第二次完整
HOST 遍历。因此 83.012 ms 的区间包含值得验证的重复遍历，但不是其独占
CPU 时间测量，不能把整个区间宣布为可消除成本。

`_ieee_prepare_selected_adapter` 在消费上述回执、更新准备成本后，还同步
await HOST 引用释放；实际等待存在，但本轮不做异步释放/合并 RPC，因为这会
涉及容量与取消语义。D333 已明确 pending ACK 后、HOST 保护前的 pending-load
计数会影响其他请求路由，不能用简单合并抹去该顺序。

本轮联网核对固定 v0.30.0 官方实现并查看本机已装版本：

- [worker_manager.py](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
  `load_inplace=False` 且 CPU 已注册时，走已有对象访问后激活；原生变更属于
  单线程 core loop。该事实只支持考察受限事务，不证明整个 HOST 路径免费。
- [model_manager.py](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)：
  dense activation 从注册对象向 GPU slot 设置权重；LRU 激活可替换 GPU slot，
  不能因 CPU 对象不新建而删除 GPU 完成栅栏或物理容量检查。
- D329 的 storage resize、tensor replacement、alias、pinning 反例仍有效。
  不按 model ID/epoch 建跨请求物理事实缓存，不跳过首次完整测量。
- D352 同事务 GPU-hit 优化已经存在；候选只研究此前明确排除的 HOST 激活。
  D595 并行压力和 D628 锁外校验的完整回放退化已经撤销，不重复引入。

## 唯一下一候选：受限 HOST 激活事务内复用刚测得的存储事实

假设：在已受保护、CPU 已注册、非 inplace、dense vLLM0.30 的同步激活中，
CPU 权重图不改变。若能对该执行路径给出明确证明，就可以复用本次首次完整
HOST 测量的 storage/alias/pinning 事实，后检查仍重新读取 allocator occupancy
并计算当前 workspace 约束；只省第二次完整遍历，不复用旧容量决定。

进入生产前的必要条件：

1. 核查 native loader、dense layer `set_lora`、GPU LRU 替换、completion fence
   及 HOST/staging 所有权，证明两次检查之间不发生 CPU graph 变更、不让出 owner。
2. 复用 D352 夹具，检查新鲜 allocator 增长/不可用、workspace 拒绝、异常 poison、
   transport 重放，以及冷加载/staging/inplace/未确认路径不得误触发。
   未证明的路径保留原本的测量，不以异常兜底或静默过关实现提速。
3. 保留 D329 反例与每次新调用重新测量；任何身份或图结构证明不成立则不启用。
4. 最小原生资格须观察真实触发、完整计量和正确绑定；通过后仅一次新执行键
   完整 3B W0，沿用 D648 SLO 和 D660 非目标配置。不能按组件收益宣布达标。
5. 若无法证明语义或完整回放无收益，保留失败并撤销这项候选；不堆叠其他参数。

本轮仅选择下一项可证伪工作，没有实现候选，也没有宣称它必然令联合 SLO
超过 95% 或降低 GPU-s。D661 所见旧 TTFT-low 控制门槛仍原样保留。
3B W0 前不启动 W1/W2/新基线；既定 3B/dLoRA、同版本 7B、A1–A5、S1–S13、
成本/CE、两轮审稿文档及最终图表全部仍在主线。

## 分析修订、安全与来源

首次分析错将独立的 scheduler/residency owner 及不同阶段 epoch 要求相等，
故所有目标链条明确报错、未填零；原文件和日志保留。第二次修正分析器身份
规则后 3316 条目标链全部通过。它不是生产错误修复或性能重跑。
第二次 manifest 的末尾脚本引用因执行命名空间变量复用而写错路径；发布时
仅修正这一条来源路径并再次验 SHA，保存原 manifest 和修订说明，统计不变。

analysis/analysis2 实际峰值 77,918,208 / 94,105,600 B，high/max/OOM/kill/swap
均为零，实际 scope 已递归空并关闭。147 项旧投稿文件未变。推理机磁盘约
166.5 GB 可用（约155 GiB）；没有删除或再生成工件/负载，不虚报已空缓存回收。
远端服务未启动，不接触非本项目 k3s。无运行间 CI、无图表、无新 GPU 结果。

原始证据：`results/ieee_tc/p2_backend_qualification/d662_20261018/`。
汇总：`paper_results/ieee_tc/p2_backend/d662_3b_native_source_partition/`。
PLAN SHA `d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。
D660 result SHA `d26e59fe78ef4ba4000837b07d9d62ba3a282a923d9f9a962f8a619256557c76`。
