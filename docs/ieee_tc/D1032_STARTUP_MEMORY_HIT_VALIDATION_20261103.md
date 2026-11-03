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

## 完成，禁止重复本验证

12:48:30 启动，约12:50完成，source commit e01ee413。16/16原生合同通过，
worker4108335确实命中D1031producer4043841，实际回执和原生跳过估计日志
一致。KV预算、block池、空闲4409 block及32 LoRA槽全部保持。
complete startup54693.177062 ms，原生init16.26s/compile0.26s，单次真实
capture11秒、日志显示0.69GiB。D1031冷画像compile30.52s也不同，因此
不能把冷/热全部启动差值归因于显存复用，也不把capture日志当精确峰值。

93个资源样本，service peak5757472768 B、host最小105836855296 B；
high/max/OOM/kill、服务swap、foreign compute及逃逸均0。物理GPU占用
78.5752496943秒，完整release、service scope及workspace移除通过。
16次客户端真实传输37185660 B，发布归档/内容核验通过，无请求打包；
本最小验证不宣称额外双端逐条审计已完成。完整W2仍要求双端核查。

缓存原文件SHA保持b708cc9f11c4a4169340bd17b60253f42c56c42b7d61d3be2b88f0f281662e39。
4个worker小文件归档逐成员与原件一致，完整出处在completion.json。
实际engine-memory身份相同；原始model_config仅观测元数据
safetensors_mem_available_bytes不同（109662965760→109634867200），
不能把原始字典说成逐字节相同。没有为命中删除此策略的任何身份因子。

旧collector收尾仍出现_retire_ieee_host_budget协程未await的警告；独立
native release、物理allocation和workspace检查均通过，保留原日志，不隐去。
纯离线汇总第一次用了无numpy的系统Python、第二次误要求观测元数据相同，
均在输出前停止；改用既有环境并显式报告唯一元数据差异。没有GPU重跑。
Full配置草拟一次遇manifest字段名KeyError，也未生成输出，已按实际字段修正。

47测试通过（6既有警告、5.02秒）；分析器只为原生明确skip分支接受一次
真实capture，其他仍保持原约束。D1033实际strict loader验证160/96类通过，
只14处预期配置差异。下一步直接唯一新键完整W2，见D1033文档。
