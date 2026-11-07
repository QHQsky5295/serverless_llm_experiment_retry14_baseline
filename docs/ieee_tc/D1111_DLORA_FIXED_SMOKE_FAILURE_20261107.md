# D1111：dLoRA fixed-length smoke 的资格结论

本记录只描述 dLoRA 的当前公平协议资格，不把失败运行转成性能点。

## 身份

- model: Llama-3.2-3B-Instruct
- workload source: 既有 4,000-request / 500-adapter trace 与 subset；未重新生成输入或工件
- generation: `fixed_length_greedy_v1`，本地 replay 读取 native token IDs
- dLoRA source: `vendor_new_baselines/dLoRA_artifact_main_20260519`，HEAD `75f1c43`；该 checkout 的既有适配修改保持不变
- runtime: migration type 3，two groups，TP=1，GPU 0--1，`gpu_memory_utilization=0.92`，`max_num_seqs=2`，`swap_space=2 GiB`
- frozen content manifest: `paper_results/ieee_tc/inputs/20260927_3b_remote_content_index.json`

## 观测

远端认证通过，500/500 adapter 的 4,000 文件完成逐文件 materialization，耗时约
70 秒。随后 native Ray/vLLM worker 开始加载 500 个 PEFT adapter 到 dLoRA CPU
pool；在 1,800 秒启动期限内始终没有暴露 loopback HTTP 端口，因此没有发出请求，
没有生成固定长度 token、TTFT、TPOT、SLO 或 GPU-s/request 结果。

启动期间 cgroup 峰值约 93.6 GiB；`memory.events` 的 high/max/oom/oom-kill 均为
零。GPU 0/1 曾各占约 7.4 GiB，随后在停止时释放。该运行不是 OOM，也不是请求
失败，而是当前资源和文件系统条件下的启动资格失败。

历史 dLoRA 3B 运行日志显示同一 500-adapter pool 在没有本轮 cgroup/磁盘压力的
条件下曾完成启动；因此不能把 D1111 的启动超时直接宣称为 dLoRA 算法固有性能，
但也不能把历史 legacy-variable-output 结果冒充本轮 fixed-length 公平基线。

## 处理

D1111 被排除，不进入主比较或任何平均值。由于 W1/W2 使用相同 500-adapter
pool，启动阶段与 workload rotation 无关，本次资格失败足以阻止在同一资源包络下
把 dLoRA W1/W2 作为正式性能点；历史结果和障碍说明保留。后续若改变机器的
可审计存储/CPU 资源包络，必须重新登记新的 execution key，不能覆盖 D1111。

完整日志与失败证据位于 baseline results 的 D1111 目录；请求前的 materialization
cache 已精确回收，输入、日志、manifest 和失败记录未删。
