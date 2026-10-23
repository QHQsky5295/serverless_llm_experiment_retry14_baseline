# D782：保留 dLoRA 缓存能力的必要存储适配

2026-10-24。本轮按run-experiment/monitor-experiment复用受限入口，
analyze-results区分实际验证和容量推算，github-sync备份经测试的有限修改。
不修改论文，不生成图、不改变冻结评价协议，不触碰其他项目的k3s。

## 为什么不缩小CPU KV

D781的失败证据全部保留。原生调度器可跨批次保留READY/swapped请求，
最大调度batch4不代表全部上下文只有4个；直接缩小CPU KV可能改变基线能力。
因此以新的源码/工件证据修正适配方向，而非无依据重复旧配置：既有完整500池
都只包含q/k/v/o权重，CPU MLP组在保留加载路径中从未写入、始终为零。

采用显式opt-in的CPU零值共享视图，GPU分配/拷贝/驻留/迁移、CPU KV8GiB名义
配置、500池和原生调度不变。运行时检查全部工件配置和tensor header，含MLP
或其他不支持权重时拒绝，不以零值替代真实权重。只读视图不作为写入目的地。

| 证据 | 结果及含义 |
|---|---|
| 全池元数据 | 500/500 attention-only；不是全文件内容重新哈希 |
| 针对性测试 | 5项通过，默认旧路径和GPU方法保持 |
| 原生环境真实张量 | 两个既有adapter、各28层；448项CPU检查差异0 |
| GPU拷贝 | 原顺序及反序槽位896项检查差异0，含GQA K/V偏移 |
| 只读零值表示 | 112个张量均pinned、索引切片连续、底层storage共享 |
| 微测资源 | 服务峰值907112448bytes，high/max/OOM/swap均0，已退出释放 |
| 完整500池运行及模型输出数值证明 | 尚未完成；不升级为性能/正确性结论 |

静态容量推算：每worker CPU LoRA31.5→17.52734375GiB，CPU KV保留14GiB；
DP2合计91→63.0546875GiB，不含运行时和其他开销。没有把这约27.95GiB减少
当成完整服务实测节省，更不能据此推出TTFT/GPU-s改善。

依据：[官方dLoRA分配/copy](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/worker/lora_engine.py)、
[固定PyTorch pinned allocator](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/aten/src/ATen/cuda/CachingHostAllocator.cpp)、
[expand只读使用条件](https://docs.pytorch.org/docs/2.1/generated/torch.Tensor.expand.html)。
baseline仓库同名文档记录实现范围。RAW为
`/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d782_20261024`；
小型证据在`paper_results/ieee_tc/baseline_audit/d782_dlora_zero_mlp_views/`。

## 返回实验主线

下一D783使用D782新source_view及其manifest、D779既有环境、D780私有Ray
按需启动与清理修正，原DP2/TP1、500池、同100请求前缀；仍实际真实远端，
不在推理机免费预加载、不改变60秒部署通知。D782本地权重小测仅证明张量
表示/拷贝，不冒称真实远程主实验。此后依次限定dLoRA3B W1/W2、
Loquetier/HydraServe3B、PrimeLoRA完整对照优化、同版本Full、A1–A5、S1–S13、
成本/CE、审稿文档和最终图表。完整计划不删减，整体目标active。
