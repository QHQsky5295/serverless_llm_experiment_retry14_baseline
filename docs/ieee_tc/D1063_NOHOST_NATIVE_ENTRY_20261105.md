# D1063：NoHOST 的单因素配置与真实后端验证

2026-11-05 01:12 更新：唯一完整NoHOST已结束，4000/4000成功终态，实际GPU
释放，collector和comparison均退出0。3B A1现在11/11；详见
`D1063_NOHOST_OUTCOME_20261105.md`。Full对NoHOST为资源—时延取舍，不是
明确端到端胜出；旧反例全部保留。下文启动/等待记录均为历史，禁止据此重启。

接续 `D1063_NOHOST_CONTROLLER_SOURCES_20261105.md`，父提交
`987124985454683a02135527e9c1bb5fb43b65cb`。完整复核目标、原计划、V1/V2 和
历史检查；不接入或修改不属于本项目的 k3s。HydraServe 的隔离适配及已有
3B W0/W1/W2 结果继续复用。

## 待回答的问题

在相同 GPU/NVMe 预算和控制规律下，取消独立 HOST 复用是否改变远端/文件
加载、尾延迟、联合 SLO 与物理 GPU 时间？保留 GPU backing、真实引用和临时
staging 的安全生命周期，不以零 HOST 预算人为破坏 vLLM 加载。

新配置 `configs/ieee_tc/20261105_d1063_3b_nohost_w1.yaml` 直接来自 D1047。
解析后只允许 `preloading.independent_host_reuse=false` 和三个独占输出/缓存
路径不同，其他预算、profile、扩缩容、路由、生成、输入全部不变。论文九公式
不修改；所有既有反例和失败保留。

## 有限的新增检查

80 项 CPU 检查已通过且源 SHA 未变，直接复用，不重新跑整套测试。
但新 staging/private-object/lease 路径尚未经过真实 CUDA 后端，因此只执行
一个既有 W1 前缀 100 请求诊断，身份 `d1063_3b_nohost_w1_smoke100`。
不以此比较性能或替代 4,000 请求。检查模式实际生效、原生计数、来源授权、
加载/释放和资源收尾；未触发路径明确标注，不额外扩成全资格矩阵。

通过后才启动 `d1063_3b_nohost_w1` 的唯一 4,000 请求，复用 D1047 Full。
如果发现具体执行缺陷，先保存失败及阶段证据，只修对应原因，不盲目重复，
不放宽预算、超时或正确性合同。n_correct 仍不能从 HTTP/长度或零更新工件
推定；本轮不新增 LoRA 工件。

在线复核的官方依据为 vLLM 0.30.0 的
[worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
和 [model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)：
文件构造、CPU 注册和 GPU 激活分开，消融不能把物理 backing 假装抹掉。
具体实现与 80 项检查见前一交接页。

## 运行与保护

复用现有 D1000 runner、vLLM 0.30 环境、W1 变换、既有 trace/subset、D78/D80
远端只读 gzip 缓存和 fresh D747 HOST descriptor。tmux 托管；服务 72/80 GiB、
swap 2 GiB，外置回放/监控 3/4 GiB、swap 0，推理机磁盘 150/100 GiB 不变。
运行期间不压缩、构建、哈希扫盘或操作远端服务。

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1063_20261105/`。
`fresh_launch_check.py` 复用已有 preflight 工具验证单因素差异、80 项源码 SHA、
旧投稿 147 项 seal、用户两项 dirty 和服务 health；不新增控制框架。
实际结果另附，当前登记不是通过证明。暂不生成图。

## 后续主线

完成 NoHOST 后 3B A1 才到 11/11；随后综合负面和正面机制证据决定 Full 的
一个公式保持候选。仍未完成的 7B A1/累积点、受控 A2–A5、M2、S1–S13、CE/
价格/Pareto、两轮审稿设计文档、图表与最终 V2 集成不因本诊断删除。

## 第一次真实入口检查：请求前停止

`d1063_3b_nohost_w1_smoke100` 在读取 profile 时停止，N_arrived=N_submitted=0，
service exit 1、资源域清理和原生 GPU 释放已确认。错误是新增
`independent_host_reuse=false` 被当成不同后端配置；不是性能结果，不隐藏。

在现有 profile identity 中明确区分来源可用性开关与实际计算配置：NoHOST
保留在完整运行身份、owner 回执及来源筛选中，不改变原 profile 的 SHA、样本、
类别或均值。两变体对剩余合法来源采用同一初始估计，实测仍照常更新；非法
HOST 来源由既有 NoHOST 可行性排除，而不是把 HOST 样本变成其他层样本。
模型、dtype、slot、预算、allocator 和其他后端字段继续严格匹配。

新增检查实际 D1047 配置经 runtime child 解析后，service/preparation 两个
原文件均可绑定 NoHOST；模式不从 child 丢失。错误类型开关及其他 backend/
预算变化仍拒绝。25 项定向检查通过，source SHA 前后一致，内存峰值
621,506,560 bytes，high/max/OOM/kill/swap 均零，未执行 CUDA。第一次测试
启动因手写 scope 名不符合既有 UUID 身份规则，在测试前退出；随后按现有
uuid4 方式启动，不改资源或身份校验规则。

仅修复此可复现的入口原因，下一 attempt 为 `d1063_3b_nohost_w1_smoke100_a2`。
80 项旧检查按未变生产源复用；25 项覆盖本次新增的 profile identity 修改。

## 第二次短回放及闭合核验

`smoke100_a2` 完成 100/100，原生输出数逐请求等于目标；现有 analyzer 的计时
重算误差为零。100 条原生回执均为 `independent_host_reuse=false`。dispatch
来源为 GPU 69、remote 29、NVMe 2，无独立 HOST；原生 acquisition 为 GPU 69、
file 30、staging 1，确实执行了新 staging join。文件准备 58 组（53 completed、
5 superseded），GPU 准备 15 组（10 completed、5 superseded），原生 CPU 准备
21 组/48 attempts。需求回执中的退役 ID 计数为零，不声称该前缀覆盖了所有
替换/引用冲突场景；相关边界 CPU 检查复用，完整 W1 继续记录实际触发。

29 次传输与远端日志匹配。168 个 watchdog 样本无外来/逃逸 GPU 进程；服务
内存峰值 13,796,933,632 bytes，launch/service/replay/watchdog 正常结束，资源域
已退出且物理 GPU 释放确认。旧投稿 147 项无变化。分析内存峰值 692,781,056
bytes，high/max/OOM/kill/swap 均零。

共同冻结 SLO 在此前缀观测为 99/100，物理 GPU 时间 213.849634 秒；它只是
路径诊断，不作 Full/NoHOST 性能比较或 G1/G2 达标证明。零更新工件的独立
数值身份限制保留，`n_correct=null`；远端未连续监控的干扰未知限制同 D1047。
实际核验产物在 RAW `smoke100_a2_analysis1/`，所有失败和源码身份保留。

下一动作：按 github-sync 保存 tested checkpoint，然后仅运行完整
`d1063_3b_nohost_w1`。CPU/短回放不替代最后一个 3B A1 性能点。

## 00:30 实验启动交接

按 github-sync 检查后提交 `0ad0f12cb1c90084c8f37f8070ad662e535b7852`，已核验
origin 的 V2 分支及 faaslora_origin 的
`backup/d1063-nohost-native-entry-20261105`；没有 force，也未宣称分叉的
FaaSLoRA V2 已完成集成。用户两个 dirty 文件未 stage、SHA 未变。

`tc-d1063-nohost-3b-w1` tmux 于 2026-11-05 00:30:49 启动唯一完整 4,000 请求
NoHOST，RAW 子目录 `d1063_3b_nohost_w1`。启动前复核单因素配置、源码、旧稿
seal、空闲 GPU 和远端 health；实际资源准入由既有 launcher 执行。不可因暂未
看到输出而重复启动。后续只读 `postrun_pipeline.sh` 等待本 session 结束及
正常资源释放，随后复用 D703 共同 SLO/原生计量、profile、远端与资源审核。
新增检查 NoHOST 的逐请求模式与合法来源，失败即保留，不自动重试推理。

完整结果尚未完成；待分析后再和 D1047 Full 比较。当前 3B A1 仍记 10/11。

## 00:41 持续运行与离线比较准备

本轮重新完整读取目标、计划、V1/V2 及最新交接。上一轮为实际进展：原生
100 请求验证、profile 身份修复/25 项检查、提交备份和完整回放已真实执行。
本轮通过 tmux、进程与新增长度核实同一回放仍存活，不把观察超时当失败。
launcher/service/watchdog PID 为 687433/687449/687452，post-run 等待 PID
693692；后续必须重新核实这些身份，不能只依赖此页。

截至本次观察，已到达 1000/4000、完成 993、失败 0；这不是最终指标。最近
watchdog 样本 555：主机可用约 89.0 GiB、服务内存峰值 17.96 GiB，服务
high/max/OOM/kill/swap 全零，无外来/逃逸 GPU 进程。磁盘 149.54 GiB，低于
下一任务 150 GiB 启动线但高于本轮 100 GiB 停止线；不在测量中清理。结束
后先核查工作缓存释放与必要的无损压缩，再判断能否启动后续重型任务。

RAW 新 `compare.py` 从 D1050 的已有比较脚本调整，只做离线分析，未修改
任何运行中的服务代码/配置。它复用现有原生指标和 first-service 分析函数，
核查 Full/NoHOST 逐请求 adapter/prompt/target/输入/输出身份，再读取已封存
九组比较，生成包括 Full 和 NoHOST 的 11 点 A1 表。每组的 Full 指标必须
精确等于 D1047，已有不利结果保留。NoHOST 语义区分 GPU backing/staging 与
独立 HOST 服务，不将前者假装不存在。脚本语法检查通过，但尚未执行；需等
完整回放与现有 `postrun_pipeline.sh` 的闭合审核通过后，在既有 3/4 GiB
辅助 scope 内执行一次。当前 post-run 只自动做 collector，不自动跑比较，
不自动重试推理、画图或提交结果。

没有新增性能运行、优化候选、外部 baseline 或远端操作。整体目标仍 active，
3B A1 完成数仍为 10/11；剩余完整范围同上。

## 00:50 集群授权核对与后置汇总排程

用户关于非项目 k3s 的回复与既有
`HYDRASERVE_ISOLATION_DECISION_20261012.md` 一致。核对 D807/D808/D809：
HydraServe 已使用隔离 rootless Podman 路径完成 3B W0/W1/W2，各 4000 请求。
这不是本轮新完成的实验；其 pp1、未证明 pipeline/consolidation、局部数值
参考而非全量独立正确性等限制继续保留。不再等待 kubeconfig，不修改共享
集群，不重开基线资格或性能搜索，继续当前 NoHOST 主线。

本轮完整读取目标、PLAN、V1/V2、最新交接和相关基线边界。轻量读取九份
已封存比较的元数据：Full 均为 D1047、3836/4000 联合达标、物理 GPU 时间
6182.715970784891 秒，协议 SHA 一致。不将这些元数据读取冒充新性能实验，
也不提前解释尚未完成的 NoHOST。

在既有 post-run collector 后增加独立等待入口 `compare_after_closure.sh`，
只复用已准备的 `compare.py`。tmux `tc-d1063-nohost-comparison` 于 00:49:45
启动，bash PID 737353；此时它只等待 `tc-d1063-nohost-postrun` 退出。
随后要求 collector 的正常结束记录和完整闭合证据，再在 CPU 2,3,26,27、
3/4 GiB、swap 0、CUDA 不可见的既有辅助资源域内汇总。脚本语法检查通过；
不是已经生成结果。没有自动推理重试、下一实验、图表或提交操作。
两小时等待上限仅停止汇总等待，不终止或重启推理。日志为 RAW
`compare_after_closure.log`，失败输出保留；若只是离线汇总失败，不重跑推理。

- 等待脚本 SHA：`684fb0c0fcb8d652177c711ed9cb58eca259fa1b5b5f0cfa5aa50850dcebee3d`。
- 比较脚本 SHA：`4435b0423bff0a39a52b458d9a6ec046974c94be4626072a1c9ce3ee82e9373c`。

00:50:11 再核实 launcher/service/watchdog 及 collector 等待进程仍存活；
已提交2081，终态记录2076，尚非4000条完整结论。watchdog样本1145显示
主机可用95414108160字节，服务峰值20012142592字节，high/max/OOM/kill/
swap均零，无外来或逃逸GPU进程。两项用户dirty文件SHA仍未变，未stage。
没有修改正在运行的服务源码/配置，没有维护I/O或远端操作。

下一动作：检查同一轮完整回放、collector与comparison三个实际进程/输出，
不能仅凭本页PID判定存活；若都已结束，读取闭合证据与十一点A1汇总，客观
解释Full/NoHOST及已有负面消融。随后核查缓存释放和磁盘余量、保存测试后的
检查点，依据完整机制证据再决定一个公式保持候选。全部其他未完成范围不变。
本轮为后置汇总排程进展和已验证等待，不是目标完成；整体目标保持active。

00:55:20 后续 goal turn 为 VERIFIED_WAIT：重新读取目标并核对 PLAN/V1/V2
SHA 未变，复核 D1047/D1052/D1061/D1062 的既有解释边界。实际上述五个
进程仍存活，提交2816、终态记录2812；watchdog样本1450，主机可用88.90GiB、
服务峰值19.81GiB、保护事件/swap及foreign/escaped均零。磁盘可用
158802886656字节，仍高于本轮停止线；测量中未维护。collector和comparison
仍等待同一回放，不重启、不提前产出完整结果。下一继续核查同一实际运行。
