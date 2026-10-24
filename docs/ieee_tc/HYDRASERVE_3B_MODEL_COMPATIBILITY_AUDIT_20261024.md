# HydraServe 3B：现有模型路径的静态核查

2026-10-24，D801 Loquetier W1运行期间的只读核查。没有启动HydraServe、
导入Torch、安装环境、修改运行代码或接管其他项目k3s。本文记录实际差异，
不是新的性能结果、模型正确性证明或重新开始平台适配的理由。

## 可直接复用的范围

D574/D576/D579/D582已保留原生后端、独立rootless Podman、真实worker
资源计量与生命周期适配。继续复用，不改其他项目的Kubernetes服务、权限、
网络或GPU分配。3B仍须采用原有3B工件、固定prompt/target和真实远端交付。
本次使用vLLM serving技能的模型加载诊断思路，不采用其中通用示例吞吐/
TTFT阈值、参数扫描或升级整个后端；冻结V1/V2及基线有界适配规则不变。

D576的`native_body.py`把`installs/hydraserve_native_d484_20261013`只读挂载
到worker，并通过已有source overlay执行原生入口。该次16项overlay没有
替换`rotary_embedding.py`或`models/llama.py`；下列两个文件的vendor与
已安装副本SHA一致。不能把7B已运行等同于3B模型数学已受支持。

## 已确认的静态事实与未证明部分

| 项目 | 现有事实 | 对后续3B的约束 |
|---|---|---|
| 模型配置 | 28层，24个Q头/8个KV头，head dim128，theta500000；`rope_type=llama3`、factor32、low1/high4、原始长度8192 | 不能删掉RoPE配置或套普通/线性缩放来使模型勉强启动 |
| 原生RoPE分派 | `get_rope`支持linear/dynamic/yarn/su，没有llama3，未知类型抛错 | 当前路径不能未经适配直接用于该3B配置 |
| 配置解析 | 当前环境的Transformers LlamaConfig已经包含llama3校验 | 没有证据要求为此升级整套Transformers；配置能读入不等于旧vLLM算术支持 |
| 输出权重 | 配置要求`tie_word_embeddings=true`；检查点索引没有`lm_head.weight`，有`model.embed_tokens.weight` | 原生初始化/加载/共享路径必须恢复这项模型约束 |
| 原生Llama实现 | 只在首PP stage构造embedding，末stage另外构造LM head；未发现按`tie_word_embeddings`建立关联的分支 | 需同时检查PP1以及末stage/合并的参数归属，不能只绕过缺少head的错误 |
| 分片元数据 | `materialize_metadata`和`collect_witness`明确仅接受原untied32层7B，其他模型拒绝 | 现有拒绝是已知资格边界；不能删除检查后宣称3B合格 |
| 原生provider | `transform`本来会按tied配置把embedding派给首末stage，并跳过独立head | 应对齐该原生语义，不另建权重副本或改成其他loader |

上述是源码与已有配置/索引的证据。尚未运行3B初始化、真实加载、prefill、
paged decode或consolidation；不能据静态检查给出其数值误差或性能结论。
也不把已确认的RoPE缺口扩大成整个HydraServe不支持Llama架构的结论。

## 适配依据与第一性原则

已重新联网核对[vLLM v0.6.3.post1的RoPE实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.6.3.post1/vllm/model_executor/layers/rotary_embedding.py)
与[Transformers v4.45.2的频率定义](https://raw.githubusercontent.com/huggingface/transformers/v4.45.2/src/transformers/modeling_rope_utils.py)：
基础频率采用FP32，按模型参数对高/低频段分别保持/缩放，中间频段连续插值。
可借鉴这一模型数学，不迁移新版本调度或替换HydraServe的专用后端。

[vLLM同版本Llama实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.6.3.post1/vllm/model_executor/models/llama.py)
明确按模型配置共享输出权重，并处理末stage所需embedding与重复head加载。
它提供语义参照，不表示可把现代PP实现直接覆盖到HydraServe的原生合并流程。

历史D788–D790已证明dLoRA需要这两项模型适配；D796–D798另核查了
Loquetier的GQA索引和RoPE。可复用既有真实prompt、参考依赖和数值判据，
但不能将其他后端的通过结论转授给HydraServe，也不重复已关闭的无效方向。
HydraServe原生attention是否正确处理GQA3尚无本轮运行证据，不能因为
Loquetier出现过索引错误就对它盲目复制同一补丁。

必要实现应在新、可回退的source overlay中由模型配置驱动：保留原生加载/
启动重叠、LoRA manager、pipeline及consolidation，只补实际模型契约。
分片与共享元数据由现有张量header和原生分配/映射函数推导，不复制整个
checkpoint，不新增权重、trace或人为未来热点。原7B路径和旧结果不变。

最小验证应覆盖实际模型输入、prefill和paged decode、共享head及受影响的
分片/合并身份；复用原生输入准备函数，避免重演D789验证入口的slot dtype
错误。只根据具体受影响路径补测试，不重新做无关平台资格或重复100/1000
回放。仅数值小样本通过仍不等于全部4000请求独立正确。

## 核查身份与下一主线

- 原生根：`/home/qhq/serverless_llm_baselines/vendor_new_baselines/HydraServe_main_20260520`。
- `rotary_embedding.py` SHA：`00652e44bc6bc5c8ad0f529c359c6a646ed888890f9477117fbcc4d5166e4ea7`。
- `models/llama.py` SHA：`13b7e5f17b678ea73d35901649b856ca771b632ae9f81c986676606fe18a558a`。
- `runtime/native_delivery.py` SHA：`1ee3e448f71528700f3014d4d5bbe3ef135adc30733a0b2556fdbd6d9645bcc5`。
- `scripts/probe_layer_identity.py` SHA：`ba15fe5f54a9d3d7a296b6fbfaeccd7a06262dc5ddb561889b50de6389e86d7a`。
- 模型：既有`models/LLM-Research--Llama-3.2-3B-Instruct`，未改动配置/索引/权重。

当前先完成并保存同一D801 W1，再运行同配置Loquetier W2；随后才实际执行
HydraServe 3B必要适配和完整场景。本文随下一安全检查点备份，不在性能测量
期间push。完整比较、PrimeLoRA优化、同版Full、A1–A5、S1–S13、成本/CE、
两轮审稿文档与最终图表/V2全部保留。
