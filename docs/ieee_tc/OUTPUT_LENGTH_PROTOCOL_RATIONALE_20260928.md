# 固定输出长度：适用范围、依据与执行决定

核查日期：2026-09-28。本文是已批准协议的解释，不是新实验计划或协议修订。
核查代码：`d6733aa8607e1fc4f66be3b261f555253c1a1f87`。
冻结指标协议：`primelora_tc_metrics_v1`，SHA256
`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`。

## 1. 结论

当前主实验要求每条请求的原生实际输出 token 数等于该请求预先规定的 target。
这是“完成相同推理工作量”的受控性能实验条件，不是要求不同系统生成相同文本，
也不是预测模型自然停止时会输出多少 token。不同请求仍可以具有不同 target。

自然 EOS/stop 请求不应要求 actual==cap；正常提前结束可以是有效完成。
原计划 S12 已明确采用自然停止、无需长度等于 cap，继续保留该补充实验。
因此，保留现有主协议是合理的，但不能把它称为所有生产请求通用的成功定义。

## 2. 为什么概率生成不妨碍控制长度

主协议逐请求使用：

\[
n_r^{target}=\min(n_r^{source\ expected},256).
\]

实际参数为 temperature=0、top_p=1、ignore_eos=true、空 stop/stop_token_ids，
max_tokens=该请求 target。原 trace 的 expected 字段在这里被转为“声明的生成工作量”，
不是对自然输出长度的准确预测。只设置 max_tokens 则通常只是上限，不能保证长度。

token 的选择与停止条件是两个问题：greedy 减少采样差异；忽略 EOS、取消 stop 并
由长度上限终止，才控制生成数量。即使不同后端的数值细节导致内容不同，也可以
执行相同数量的 decode 步骤。不要求跨后端逐字一致，更不以返回文本重分词代替
后端实际 token IDs。异常、取消、上下文越界、服务端拒绝仍可能导致未完成合同。
官方参数语义见 [vLLM 0.30.0 sampling_params.py](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/sampling_params.py)。

## 3. 与同类工作的对应及证据边界

| 来源 | 核实到的设置 | 能支持什么、不能支持什么 |
|---|---|---|
| [HydraServe，NSDI 2026，§8.4](https://www.usenix.org/system/files/nsdi26-lou.pdf) | Pipeline consolidation 实验将每条请求输入和输出均设为 512 token | 直接支持固定长度机制实验；不据此声称该论文所有实验均固定为 512 |
| [S-LoRA，§7.2](https://arxiv.org/html/2311.03285v3) | 合成 workload 的每条请求输入、输出长度预先从均匀分布采样 | 支持变长但预先指定工作量；该段本身不证明其使用与本项目完全相同的 EOS/100% 验收实现 |
| [Anyscale：Ray Serve LLM benchmarking，Handle variability in token outputs](https://docs.anyscale.com/llm/serving/benchmarking/benchmarking-guide) | 建议固定输入/输出长度并使用 ignore-eos 以获得一致运行 | 工业评测实践的明确依据；是官方指南，不是同行评审论文或普遍强制标准 |
| [vLLM 0.30.0 参数源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/sampling_params.py) | temperature=0 为 greedy；ignore_eos 忽略 EOS；max_tokens 是数量上限 | 证明控制机制可实现；不证明所有 benchmark 默认开启 ignore_eos |

这是针对具体问题的原文核查，不是穷尽性文献综述。不把其他论文未说明的
验收细节补写成事实，也不因某个设置被发表过就自动断言本项目设置最优。
本轮 research-lookup 的 Parallel 限定查询在 300 秒超时；上述结论来自直接
打开并核对原论文/官方页面，不依赖失败检索的推测内容。

## 4. 对 PrimeLoRA 评价目标的意义

若系统 A 实际生成 32 token、系统 B 生成 256 token，直接比较 E2E、GPU-s/request
或 cost/request，会混入工作量差异。长度会影响 decode 工作、KV 驻留、后续请求
并发和排队，因而也会间接改变 TTFT、扩缩容及生命周期资源占用。

固定相同 prompt/adapter/arrival/target，能减少这项混杂，适合检验 G1（共同 SLO
下的生命周期 GPU 占用）和 G2（共同预算下的尾延迟）。它不是保证完全相同硬件
指令数或相同内部执行：批处理、缓存、后端优化的真实差异仍是测量对象。
声明的逐请求输出上限可以供所有系统使用，但不得给 Prime 提供独享的未来请求
或自然生成结果信息。

不能以“输出长度相同”证明答案质量、正确 LoRA 的数值应用或 SLO 达标。至少区分：

1. 生成/传输合同：正常完成、原生实际 token 数、正确 request/adapter 元数据。
2. 数值正确性：期望 adapter 确实应用；仅名称正确或长度相同不够。
3. 服务质量：在有效完成基础上判断共同 TTFT/TPOT SLO。

当前已有工件的数值可区分性限制仍保留，不能以 token 检查通过替代该证据。

## 5. 局限及原有补充实验

- 忽略 EOS 可能使生成越过自然答案结束，不能代表文本质量或所有实际产品体验。
- 固定工作量下的资源结论限定于该 workload；自然停止会改变长度、排队和扩缩容。
- 保留原有逐请求变长 trace，不把全部请求强改为 256；cap 与截断比例需要报告。
- 原计划 S9 保留长度/KV 敏感性；S12 保留两模型 Prime/S-LoRA/Loquetier 的自然停止
  三块实验。S12 不要求 actual==cap，长度不同也不能直接把全部 E2E 差异归因于系统。
- 固定长度和自然停止分别回答受控工作量与应用表现问题，不混入一个执行键或排名。

## 6. 已核对的实现和继续执行决定

`scripts/run_all_experiments.py` 中，fixed_length_greedy_v1 设置实际 SamplingParams；
原生完成要求 finish_reason=length，并检查实际 token_ids 数等于 safe_max_tokens。
控制器记录 requested/actual tokens 和 output_contract_match。
`faaslora/metrics/metrics_collector.py::PhysicalGPUDeployment.terminal`
进一步从原请求恢复 target，独立记录 success 与 native_contract_matched；注释
明确指出该合同不是 LoRA 数值正确性证明。

继续当前固定工作量协议；不修改 V1、计划、正在运行的配置、超时或成功分母。
不为提高完成率而将提前结束、缺 token 或未知原生终态改为成功；自然停止实验
则依其单独合同判断。后续其他基线要验证等价原生停止/计数，不能假定参数名相同
就已经实现同一工作量。

08:55:52 的 D97 实时快照：4000 请求已提交，2984 终态，其中 1389 条成功且
native-contract-matched、1499 条 TimeoutError、96 条返回失败（终态摘要 error_type
为空，完整原因待运行结束核查）。此时还未结束，也未完成资源释放。
已成功请求均长度匹配；这些计数不支持把主要失败解释成“随机文本长度不一致”。
不得把成功子集当完整 Full 性能，或把上述实时快照当最终失败归因。

09:44 完整提取补充：本轮 4000 终态为 1609 成功/原生合同匹配、2243 超时、
148 RuntimeError。148 条均报告原生 RPC 所有权未决而拒绝本次新生成，并记录
本次 generation 未提交；未发现它们因输出长度不匹配而失败。超时请求的具体
阶段不能从缺字段推断。详情见 [D97 完整失败诊断](D97_FULL_W0_ATTEMPT4.md)。
