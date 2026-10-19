# D697：3B 完整 W0 单飞扩容候选（预登记）

日期：2026-10-19  
状态：已完成并拒绝候选；原始证据与失败分析保留

## 科学问题

D695 的完整 W0 观测显示，同一控制评估可能在同一请求前沿同时提交多个
activation，随后就绪顺序改变初始副本承载分布，并伴随 admission→engine 和
native decode 退化。D697 只检验一个可证伪变化：在 `ieee_confirmed` 路由下，
每个控制评估至多提交一个物理 activation；前一 activation 登记 ready 后，后续
控制评估才可继续提交。该变化不修改论文扩缩容公式、预算、handoff 或 SLO。

## 冻结合同

- 模型：Llama-3.2-3B-Instruct；4 张 RTX 3090，TP=1。
- 负载：既有 4,000 请求 W0 trace、500-adapter 子集、seed 42；不生成新负载。
- 生成：`fixed_length_greedy_v1`，原生 vLLM token-id 计数，最多 256 输出 token。
- 资源、远端真实交付、GPU 生命周期、共同阈值和分析口径与 D695 逐字段一致。
- 代码：生产候选含已提交的 `325fe9822c694eee277f6c2d0fac9101e93cd9a4`；
  其余生产回退基线为 D693 父版本。
- 配置：`configs/ieee_tc/20261019_d697_3b_full_w0.yaml`；入口脚本为既有
  `run_d293_7b_prefetch_validation_100.sh d697`。
- 所有输出使用独占 D697 路径；若路径已存在，启动器必须拒绝覆盖。

## 预登记判定

必须保留 4,000 请求计划分母和全部失败。完整运行只有在正确性、原生生成、
adapter 绑定、请求阶段计时、物理 GPU 分配/释放及远端交付证据均完整时，才可
进入候选比较。候选至少需要实际出现可审计的 scale-up submission/ready 序列；
若未触发，标记 `qualification_only`，不据此宣称性能改善。

若触发并完成全量回放：联合 SLO、P95/平均 TTFT、TPOT、E2E、生命周期 GPU-s
和阶段分解与 D682/D695 并列比较。只有联合 SLO 或目标资源/尾延迟方向改善且
无新的正确性/资源回归时，候选才可进入下一轮开发冻结；否则回退到 D693 父版本。
无论结果如何，D697 原始日志和分析保留，不重复同一执行键寻找更好结果，不画
论文图。

## 实际收尾与裁决

D697 于 2026-10-19 完成 4,000/4,000 个请求终态；4,000 条均有原生生成合同，
共 458,224 个原生输出 token，132 次远端传输、306,360,162 bytes，且请求期
远端打包计数为 0。4 个 natural-scaleout activation 均到达 `runtime_ready`
（1 个 initial、3 个 natural scale-out），因此本候选不是“机制未触发”的资格
结果。远端 3B/7B/monitor 单元均为 `inactive/MainPID=0`，本机 GPU 进程为空，
watchdog 的 high/max/OOM/oom_kill/swap 事件均为 0；结果文件、transfer journal
和 remote monitor 均通过远端 SHA-256 校验。

严格分析首先在同一 owner 的最终 GPU action 间隔上失败：4 个 owner 共 5,005
次动作中发现 955 个重叠，最大示例为 473.019 ms。为区分“该合同失败”和
“其余观测是否完整”，随后在不改变输入或时延数据的情况下只放宽这一条断言做
数据诊断；阶段恒等式、adapter 绑定、HOST→GPU lease、远端交付、物理 GPU
积分和共同 SLO 字段均通过。该诊断不是排名资格，`n_correct` 仍未知。

| 观测点 | 联合达标 | 平均 TTFT | P95 TTFT | 平均 TPOT | 平均 E2E | 生命周期 GPU-s |
|---|---:|---:|---:|---:|---:|---:|
| D682（父候选） | 3694/4000 (92.350%) | 804.280 ms | 1992.432 ms | 16.778 ms | 2723.519 ms | 15896.254 |
| D695（同 owner 候选） | 3618/4000 (90.450%) | 844.735 ms | 2177.395 ms | 17.450 ms | 2823.288 ms | 15904.504 |
| D697（单飞候选） | 3683/4000 (92.075%) | 791.383 ms | 1999.252 ms | 16.711 ms | 2698.748 ms | 15897.828 |

D697 相比 D695 改善了平均 TTFT、平均 TPOT 和平均 E2E，但相对 D682 联合
达标少 11 条、P95 TTFT 高 6.820 ms、GPU-s 高 1.574；它没有改善预登记的
联合 SLO、尾延迟或资源目标，故按停止规则拒绝并回退 D693 父版本。不能把
均值改善单独写成系统性能提升。

此外，D697 运行中加入的 submission-side 字段没有出现在最终
`mechanism_events._ieee_activations` 或 `scale_up_events` 中（注册函数只保留
既定字段）。因此本轮只能确认发生了 3 次 natural scale-out，不能从最终结果
证明每个控制评估的 submission/ready 序列；这是一项可观测性缺口，不用后处理
猜测补齐。严格失败日志和放宽诊断分别保存在 D697 raw 目录，机器可读摘要为
`paper_results/ieee_tc/p2_backend/d697_3b_full_w0/qualification.json`。

本候选不生成论文图，不改变 D682/D693/D695/D696 的身份，不启动第二个同键
运行。下一轮若仍检验控制序列，必须先修复事件注册的字段保留并提出新的、可
证伪的瓶颈假设；修复本身不构成 D697 性能通过。

## 启动前资源与远端门禁

启动时重新检查 GPU 空闲、主机内存公式、每个文件系统的 150 GiB 启动线与
32 GiB 预测增长、CPU/cgroup 归属、owned 远端 3B 服务健康和 1-Gbps/full 链路。
若任何门禁不通过，本执行身份保持 `not_run_safety`，不伪造结果。运行结束后
按既有 close-launch → 受限停止远端 → close-management → 受限 native 分析 →
封存顺序收尾。
