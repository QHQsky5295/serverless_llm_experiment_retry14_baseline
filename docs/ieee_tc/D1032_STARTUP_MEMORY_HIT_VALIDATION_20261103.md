# D1032：同配置启动显存记录命中验证

2026-11-03，3B 主线。承接 D1031 已完成的冷画像，不重跑 423 请求。
保留 D1031 的模型、loader、admission/profile ID、初始化和全部 runtime
设置；仅复用其前两波 16 个既有请求，仍由原 native_source_matrix 收集。
这不是 Full 性能点，也不导出其 timings 为 service/preparation profile。

唯一未决问题：同一成功记录能否在下一真实原生 worker 命中，并维持
8092388312 B 原生 KV 预算及 8092385280 B block 池、正常图捕获、原生
token/binding 和最终资源释放。读取实际 startup_memory 回执，不因更快
就推断命中，不改 cache JSON/free-memory 基线/编译因子来强行命中。
若因真实身份或空闲显存不匹配未命中，保留原因；不追加暖启动搜索。

依据仍为 D1031 及官方 vLLM 0.30.0 gpu_worker.py 的
determine_available_memory：显式 KV 容量仍运行 profile_run 编译，
但跳过内存估计与临时图；compile_or_warm_up_model 仍执行服务图捕获。
2026-11-03 再次联网核验原始源码。没有新性能调参或公式变化。

已读取完整 PLAN、目标、D1031；run-experiment 与 serving-llms-vllm 技能
用于复用受限启动和原生观测。执行前后保留既有磁盘、内存、GPU、旧结果
保护及远端状态检查，不修改共享 k3s、远端配置或用户 dirty 文件。

通过后下一步为 D1031 新画像加相同 Full 配置的唯一新键 4000 请求 W2。
Full 的 loader/admission 身份不同允许首次冷启动，不能复制画像记录伪造命中。
仍按冻结 G1/G2 评价，不保证达标，不在临界点改变 SLO 或挑选运行。

两模型同最终版本 W0/W1/W2、独立数值正确性、保留外部基线比较、
A1–A5、S1–S13、CE/Pareto、两轮审稿与 B1/B2/C1/C2/C3 文档、达标后
图表和最终 V2 集成全部保留。D810 的基线已完成部分不重开适配。
