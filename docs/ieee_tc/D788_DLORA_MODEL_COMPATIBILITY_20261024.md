# D788：dLoRA 3B 必要模型兼容适配

本项不是性能调优。D786/W1及D787/W2保留为受模型错误影响的R3记录；
D787已主动停止并保存，备份commit为0d196af750075bc9d8f4d5e77861b36840881fff。
仅在新source view适配，不修改vendor或D782视图，不更换权重/负载。

## 历史、假设与最小改动

D774–D777已处理计时、canonical输入、私有Ray/worker生命周期；D782处理
缺省MLP CPU存储重复，D783–D785验证其拷贝和执行合同。这些检查不等于
模型数值正确。D787全部747条成功响应只有token0，引出具体模型检查：
3B配置要求tied vocabulary，检查点没有独立lm_head，但旧模型未共享；
另外llama3 RoPE配置未传递。实际Torch2.1.2/FP16小算式复核旧频率首项
为1、后续项退化为0，因此适配时必须按官方方法先以FP32计算频率。
不重新审计全零LoRA池，不产生新权重，不声称已证明所有异常只有一个原因。

可证伪假设：补齐模型要求的共享输出权重与Llama3位置编码后，真实native
prefill和paged decode应能与同checkpoint/adapter的Transformers/PEFT参考
达到预先指定的数值一致条件，且不再产生这一全token0退化。

适配只包括：配置驱动的PP1共享词嵌入；llama3频率缩放及FP32频率初始化。
原有调度、迁移、merge、采样、并发、KV和原生attention kernel不变；
未缩放/未共享的Llama2路径保持原样，不扩大为其他模型或PP部署的资格。
不通过检测输出token0回退到其他后端，不替换模型计算。

官方实现依据（本轮已联网核验）：

- [dLoRA保留版本Llama](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/model_executor/models/llama.py)。
- [vLLM v0.6.3.post1 Llama](https://raw.githubusercontent.com/vllm-project/vllm/v0.6.3.post1/vllm/model_executor/models/llama.py)：共享词嵌入并跳过独立head加载。
- [Transformers v4.45.2 RoPE](https://raw.githubusercontent.com/huggingface/transformers/v4.45.2/src/transformers/modeling_rope_utils.py)：FP32基础频率、三频段缩放。

## 验证范围与判据（运行前登记）

六项新增CPU单元检查已通过。完整既有dLoRA合同回归随后执行。
实际模型检查复用D787日志中按记录顺序前两个不同adapter的已有prompt IDs；
用其原有本地工件作为参考，不建立新工件池、不测远端性能。
每个prompt检查prefill和随后两步原生paged decode；参考实现使用相同
native选择token作为共同续文，避免首处分歧改变后续输入。

逐点要求finite、relative L2≤0.01、cosine≥0.999且greedy token一致；
另检查真实共享参数与各层RoPE cache和官方参考完全一致。这些是有界
实现回归判据，不是语言质量指标或全部4000请求的数值证明。
失败保留实测误差，不放宽判据，不直接进入性能回放。
先顺序执行native再释放后加载参考模型，复用已有资源限制和watchdog。
无需再次跑无关100/500/1000覆盖；通过后只重跑受影响W1/W2。

## 状态

源码与六项单元检查、46项完整回归已完成。D788实际检查在加载模型前因
参考依赖PEFT缺失停止（服务1/watchdog0，GPU和服务域已释放）；失败日志
保留，不是模型数值失败。D789仅以明确的只读package overlay复用本机已有
PEFT0.17.1/Accelerate0.34.2及其metadata，255个来源文件记录SHA；没有
安装、改动dLoRA环境或借入另一环境的Torch/Transformers。CPU import确认
仍使用原Torch2.1.2+cu121/Transformers4.45.2。相同模型检查已重新启动，
模型代码与数值判据不变；未重新执行无关资格。D789随后在native cache
kernel因probe提供int64 slot、而内核要求int32停止，GPU/服务/辅助域均已
释放。此为新增验证入口错误，不归因为dLoRA性能或模型错误；当时probe
完整源码快照SHA ef7aea3d46b17fa926e732958559895dfbcd67cbf73f05276b3ce933403b913d
及日志保留。D790改为直接复用实际Worker._prepare_inputs，连同原生padding/
adapter mapping一并复用；模型代码、参考依赖、数值判据不变。D790检查已
启动并已通过：2个既有prompt、2个既有adapter，各prefill+2步paged decode，
6/6 greedy token与Transformers/PEFT相同，全部logits finite；relative L2
范围0.00130277–0.00220894，最低cosine0.99999177，最大绝对误差0.0234375。
两条输出分别为[578,19114,374]及[578,2326,1288]，不再是此前全token0。
实际共享head已确认，各层RoPE cache与官方参考最大误差0。结果只支持
这些路径的模型兼容修正，不是4000请求逐项数值证明，不证明dLoRA最终性能。
服务/监控返回0，GPU和服务域已实际释放，probe辅助域已闭合。原失败记录
全部保留；新W1/W2仍未开始，下一冻结备份后仅重跑这两个受影响点。
后续Loquetier/HydraServe 3B、PrimeLoRA全比较及优化、同版本Full、A1–A5、
S1–S13、CE/Pareto、两轮审稿文档和图表/V2仍全部保留。
