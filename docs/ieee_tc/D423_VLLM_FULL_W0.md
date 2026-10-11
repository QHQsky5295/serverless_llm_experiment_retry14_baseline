# D423：vLLM 7B 完整 W0 结果与 Resident 参考候选

2026-10-11。一次原始 4,000 请求回放，启动前登记；不是再次优化或短测试。
唯一原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/vllm_qualification/d423_20261011`。

## 目的、身份与复用

完成计划 §6.7 的完整回放，补齐与已封存 PrimeLoRA D371/D397、ServerlessLLM D414
同一 W0 的描述性比较证据，同时预登记为该模型/场景的 Resident-vLLM 预算参考候选。
只有完整正确性/输入/计量/物理租约审计通过后才能冻结 U_ref 和 Q_ref；不从此前
100/500/1000 请求线性外推，不按结果挑选重复。W1/W2 参考及 G2 仍开放。

D419 原生路径、D421 全500-ID执行、D422 原始1000兼容、D385/D422 有界独立数值
证据全部复用，不重复这些 GPU 检查。不修改原始未知正确性标志，也不将原生绑定
等同于完整数值证明。大部分权重为零或重复的限制继续披露。

使用既有 seed42 W0，`trace_prefix_v1` 的完整4000行，132实际ID、458224目标token，
原到达窗口3963.9089031219482秒。源trace SHA
`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`，
完整视图SHA `a5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d`，
pool index SHA `e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c`。
不生成新trace、权重或整池副本。

## 冻结执行条件

原生配置与 D419/D421/D422 相同：vLLM0.30.0，DP4/TP1，原请求序号round-robin，
native LoRARequest/LRU，无 PrimeLoRA 策略。FP16、GPU fraction0.7、max-seqs2、
max-loras4、CPU-loras16、rank64、1024 context/token budget、非eager图执行、
chunked prefill、无prefix cache、seed0、原生统计启用。配置SHA
`361dc7645a8eef8ee304addfaa9e1611093dae1c6678b108970c1adc2e7fe3a5`。
HTTP配置SHA `32b80e2227a8826943d03abf66de98dfd230f1288dc39840e6973c3d1b9b9ced`。
baseline版本 `e646f9106c8fd6fd15f54b69041bd54234fa47e1`，本次无运行代码改动。
实际main/baseline执行commit由启动前input_identity固定，后续备份commit另列。

共同60秒部署通知、pre-ready独立open-loop、fixed_length_greedy_v1和1800秒
开发资格保护。保留qualification_only=true、formal_run=false、development41、
source_seed42身份，不冒充未见工作负载的正式留出评估。

服务内存high72/max80GiB、swap2GiB、CPU4–23/28–47；外置回放监控4GiB、swap0、
CPU2/3/26/27。预计新增峰值8GiB；启动≥150GiB、运行≥100GiB，独立安全监控。
全新 `/tmp/tc-d423-7bv` LoRA缓存，现有原生编译磁盘缓存单独记录当前SHA；
不预启动引擎。真实远端沿用 D78/D80 一次发布只读缓存，不打包或重新发布；
保留请求触发读取、传输、竞争与物化延迟。业务期间不改远端、不清理或批量哈希。

## 判定与收尾

先要求4000 offered/submitted/terminal/native/source/prompt/adapter/content/timing
合同完整，无fallback、actual==target、时间重算误差≤1ms、四真实worker的来源与
资源约束可核验、HTTP/内容收据相符、完整物理GPU allocation/release和保护结果零变化。
复用既有分析入口 `--scheduler-contract resident_rr_v1 --population full4000`。

离线采用已冻结D394共同SLO，SHA
`2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3`。
按V1+单轮V2报告TTFT/TPOT/联合达标、原始延迟、生命周期GPU-s及互斥窗口。
不达标也保留本次完整结果；不重跑同键追求更优值，不输出CI或显著性。

顺序：实际GPU释放 → 按InvocationID关闭本次远端服务 → 保存原始证据 →
严格完整审计 → 封存/备份 → 同合同关联比较。比较需逐字段检查初态/输入/
生成/资源/计时，不能仅把三份摘要并列后称公平比较。

跨W0/W1/W2的resource_opt、W1/W2/G2、剩余7B基线、3B、核心消融/机制/敏感性、
CE补充与论文文档/图表均未因此完成。后续仍按主线，完整目标保持进行中。

## 已启动检查点

2026-10-11 17:17:37 Asia/Shanghai 启动，tmux `tc-d423-7b`。main实际执行
`d45d544c939b9d8b1a41aa332b979b74ee0baa3c` 已推送并核验远端；baseline如上，
同样核验远端。后者一次TLS查询失败，有限重试通过，没有绕过证书或强制推送。
27项既有分析CPU测试通过，0.095秒，原始测试日志保留。

服务InvocationID `09c12668f1554ed0af12e9435539fe02`；独立回放已注册4000条合同，
17:19:20时3条请求均完成，四worker收据存在，没有失败或保护事件。当前原生
编译磁盘缓存628文件/54387041字节，SHA
`92cc5289f1f91ebf0149086a133d359549ee311ca65d6bc2ebc380dba25a58d6`，与D422相同。
这是运行中检查点，不是完整结果、预算参考冻结或性能判优。继续原进程，不重启。

## 完整结果与证据边界

2026-10-11 18:25:32 Asia/Shanghai 确认终态。上节为历史启动检查点，
该运行已结束，不得重启。原生请求、源输入、prompt、adapter、内容、输出长度
及计时合同全部通过；仍保留原始 `n_correct=null` 和未知数值资格标志。
这是一次已知 W0 开发观测，不是独立留出或跨场景最优配置。

| 指标 | 完整 7B W0 单轮 |
|---|---:|
| offered / submitted / terminal / 完整原生合同 |4000 / 4000 / 4000 / 4000|
| 失败 / 原生输出 token / 实际 adapter ID |0 / 458224 / 132|
| TTFT / TPOT / 联合达标请求 |3979 / 4000 / 3979|
| 联合 SLO 达成率 |99.475%|
| 生命周期物理 GPU-s |16115.377162149176|
| GPU-s / 合同有效请求 |4.028844290537294|
| 平均 / Type-1 P95 TTFT（ms） |358.019243 / 1200.506860|
| 平均 / Type-1 P95 TPOT（ms） |24.797976 / 27.177332|
| 平均 E2E（ms） |3182.813643|
| E2E / TPOT / 原生计时最大重算误差（ms） |0 / 0 / 0|

四个物理持卡者的区间与实际释放证据核验通过。互斥 GPU-s 分项为：
业务前233.34046448394656、到达窗口15855.635612487793、drain
1.8053004760295153、收尾24.595784701406956；四项相加等于总量。
峰值服务内存14824255488 B，主机最低可用99514773504 B，磁盘最低
160266399744 B；high/max/OOM/swap均为零。运行期间磁盘低于150 GiB
新重型任务启动线，但高于100 GiB运行底线，没有现场清理或第二个重型任务。

132次真实已发布gzip交付，共131480060线上字节、3300789780核验物化字节，
3868次请求命中本轮本地缓存，零请求时打包。没有重新发布远端缓存。

原始前100条与既有D385 HF/PEFT独立参考离线关联：100个身份匹配，97个完整
输出序列一致，3个差异保留；其中2个有可复用的既有首差异/margin证据，
`req_00084`的对应margin未观测，不一概解释为舍入。相对D419有95个序列相同、
5个变化。覆盖4个独立权重类别，但不推导另外3900条的独立数值证明，不要求
跨后端逐token相同，也不因差异重跑性能点。多数零/重复权重限制保持披露。

实际GPU释放、服务资源域移除、退出码0已确认；远端三个本轮服务按InvocationID
停止，端口关闭；本地空的回放资源域已停止。远端日志复制SHA核验通过。
旧投稿保护清单147项零变化，启动前27项CPU合同测试通过（0.095秒，日志保留）。
随后完成两仓库备份和引用核查，仅回收本轮1320个下载副本（3300789780 payload
字节），实测释放3305713664字节，可用磁盘163415580672字节。原池当前字节/
独立inode、无symlink/嵌套mount、已保存元数据SHA、可读本用户进程FD/cwd/root、
fuser和未来受跟踪配置引用均检查；三个不可读proc条目明确保留在回执，不称
系统级穷尽扫描。唯一工件、远端缓存、原始日志均保留，206文件旧manifest未改。
本轮0002至0013辅助资源域均核验同一身份且空后停止。

## 封存位置与下一步

`paper_results/ieee_tc/baseline_audit/d423_7b_full_w0/` 保存：

- `audit.json`：SHA `7d9ac56d0db8cf3af0f87c7a5314eb272771ff6408c419bba616e8f7eceeb310`。
- `bounded_reference.json`：原始前100条的独立参考复用和全部差异。
- `evidence_manifest.json`：206个保留文件、226970169字节；SHA
  `b3796a4ea857e0a5cc8bb105397bd511f6c8548065b1c20a73ca9242975be516`。

复用D422的离线关联/封存脚本，仅调整本轮路径和完整请求数，没有新框架或
GPU数值实验。原生严格分析使用现有入口，主/基线运行代码均未改变。
本轮满足共同阈值下的原生绑定观测可行性；尚需独立完整合同关联后才与
PrimeLoRA D371/D397、ServerlessLLM D414比较并冻结W0的Resident U_ref/Q_ref。
不能只凭三份摘要直接宣布公平比较、G2达成或PrimeLoRA已最优。
先封存备份，再继续7B主线；W1/W2参考、其他7B基线、3B和全套消融/敏感性仍开放。

封存备份已推送并核验远端：主仓库`4c700e70a2b5d2e5ec93bd2d8b1dfc3bc46c0dff`，
基线仓库`b477b12622b8f095dd4d6d753c36ebe12391e7f7`。它们不是执行commit。
空间回收回执：`paper_results/ieee_tc/storage/20261011_d423_download_cache.json`。
