# D423：vLLM 7B 完整 W0 与 Resident 参考候选（预登记）

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
