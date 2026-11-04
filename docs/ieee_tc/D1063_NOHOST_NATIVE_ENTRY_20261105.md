# D1063：NoHOST 的单因素配置与真实后端验证

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
