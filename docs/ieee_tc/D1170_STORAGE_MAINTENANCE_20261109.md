# D1170：旧失败回放大文件的无损压缩登记

日期：2026-11-09  
性质：存储维护，不是推理实验；不改变任何性能、SLO、正确性或运行身份。

当前推理机可用空间约 149.3 GiB，低于既有 150 GiB 重型任务门槛。为恢复
下一基线回放所需的临时空间，本轮只处理已经关闭、已经完成流式审计、且已有
curated 小型证据的两个旧开发失败回放的请求级 JSON：

- `results/ieee_tc/p2_backend_qualification/d96_20260928/3b_outputs_attempt8/experiment_results_full_vllm_dedicated_a500_r4000_c8_tc_ieee_full_d96_3b_full_w0_attempt8.json`
- `results/ieee_tc/p2_backend_qualification/d97_20260928/3b_outputs_attempt4/experiment_results_full_vllm_dedicated_a500_r4000_c8_tc_ieee_full_d97_3b_full_w0_attempt4.json`

两份文件均为历史失败/资格诊断，不是当前 Full、baseline 或留出结果；对应的
完整失败证据、curated 摘要、源码、日志和运行文档均保留。压缩规则：

1. 逐文件核对普通单链接文件、无活动进程引用和当前 SHA256；
2. 生成同路径 `.gz`，执行 `gzip -t`；
3. 两次独立流式解压 SHA256 必须等于原文件 SHA256；
4. 仅在上述核对全部通过后移除未压缩副本，保留 `.gz` 作为完整可恢复原件；
5. 保存原始/压缩 SHA、字节数、回收量和恢复命令；不处理模型、工件、负载、
   当前运行、用户修改文件或任何唯一结果。

若任一项验证失败，则保留原文件和失败证据，不继续压缩其他目标。维护结束后
重新核对磁盘、GPU、服务和资源域状态；恢复空间只用于后续已登记的 S-LoRA W2
或其他主线实验，不修改评价口径。

