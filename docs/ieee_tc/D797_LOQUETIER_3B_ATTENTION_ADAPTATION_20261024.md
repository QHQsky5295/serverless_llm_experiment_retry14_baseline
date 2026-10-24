# D797：Loquetier 3B 原生注意力的必要模型适配

2026-10-24。一次隔离构建、一次受影响注意力数值检查；不是新的性能实验，
也不表示完整模型、LoRA 或 500-adapter 工作负载已取得数值资格。

## 问题、改动和范围

D796 已实测原生 GQA3 prefill 非有限，且全部六条注意力路径不符合实际
Llama-3.2 的分频 RoPE。本轮复用原生源码、环境、KV 接口及 SMLM，修正
完整 packed row 的 token/head 商余数索引，并由已有模型配置生成分频变换。
不填充额外查询头，不回退另一后端，不修改调度、批处理、静态分片或性能参数。

依据为 [Loquetier 固定提交的原生 prefill](https://raw.githubusercontent.com/NJUDeepEngine/Loquetier/aae33baeeb19777129c1ccbff99a898d4a0e2c63/kernel_src/third_party/flashinfer/include/flashinfer/prefill.cuh)、
[FlashInfer v0.2.0 的完整 packed-index 约定](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/v0.2.0/include/flashinfer/attention/prefill.cuh)
及 [Transformers v4.45.2 的 Llama-3 频率定义](https://raw.githubusercontent.com/huggingface/transformers/v4.45.2/src/transformers/modeling_rope_utils.py)。
本轮再次读取官方来源；没有安装新依赖或上传私有实验内容。

隔离构建保留原项目扩展目标，因此编译器也重新编译其未改动的 SMLM/RMS
组件；不是只编译两个文件。原已安装二进制、7B 环境和原始 vendor 均不改。
新的模型专用二进制只用于这份配置，不能无条件应用于 7B 或其他 RoPE 配置。

## 实测结果

沿用 D796 的固定数值测试：FP16、两个序列长度 17/257、decode 后18/258，
FP32 显式因果注意力参考，`atol=rtol=0.005`。没有重新生成论文工作负载。

| GQA group | 阶段 | Llama-3 参考 relative L2 | 最大绝对误差 | 通过 |
|---:|---|---:|---:|---|
| 1 | prefill | 0.000308980 | 0.001288414 | 是 |
| 1 | decode | 0.000202411 | 0.000385761 | 是 |
| 3 | prefill | 0.000303790 | 0.001322344 | 是 |
| 3 | decode | 0.000204883 | 0.000466466 | 是 |
| 4 | prefill | 0.000306909 | 0.001263618 | 是 |
| 4 | decode | 0.000205072 | 0.000442028 | 是 |

全部输出有限，六条 KV 内容逐元素一致；普通未缩放 RoPE 参考均不通过，
符合这份模型必须使用分频缩放的预期。GPU 峰值 allocated 为121,020,416
字节。这是算术诊断，不用它比较吞吐、TTFT、资源效率或宣布系统最优。

构建与诊断的 service/watchdog 均返回0，实际 GPU 上下文释放、服务路径
移除，两个辅助资源域均已按身份检查并关闭。原始全部日志保留。

## 身份和可回退性

- 原始目录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d797_20261024`。
- 新二进制 SHA：`b1f4b70ef8d816183a9d3ca0211b79a022f05af16a9228f34346bbe4ef2e1d41`。
- 原二进制 SHA：`100efcf29d3f8a53d8b2fe31b877d1589634024c742c8fff6ff30110f5a57ec9`。
- 模型配置 SHA：`39fb36dc5416f445ebc4e71cb71fbcf6727e80a35836d8ba1a1474c318467b7a`。
- 源适配器 SHA：`32c9a9f37e85675741f6dd2e2d6184c7d4fde0f69829d72623e7c19c56cff931`。
- 诊断 runner SHA：`bf6c7e5e7782ef95d19ba7d15b5a18494f1d084f64746c0f34915cce274bc424`。
- PLAN SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。

小型证据及脚本快照保存至
`paper_results/ieee_tc/baseline_audit/d797_loquetier_attention_adaptation/`，
具体提交、测试、保护清单和推送状态以该目录 manifest / 原始备份回执为准。
不覆盖旧 D796 失败，不把成功检查覆盖成旧版本也正确。

## 下一主线

一次受影响的真实输入完整模型/LoRA 检查，核对实际共享输出 head 和模型绑定；
随后按已有配置完成 Loquetier 3B W0/W1/W2，再推进 HydraServe 3B。
不得把注意力检查当成所有权重、所有请求的数值证明，也不继续无依据调优基线。
他项目 k3s 保持不变。全基线比较、PrimeLoRA 优化、同版 Full、A1–A5、
S1–S13、成本/CE/Pareto、两轮审稿文档、图表与最终 V2 交付均未删减。
