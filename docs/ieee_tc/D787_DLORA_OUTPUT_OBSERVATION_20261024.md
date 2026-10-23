# D787：运行中的 dLoRA 输出退化线索

更新：本轮已因下述模型兼容缺口主动结束，属于 R3 截断诊断。最初选择等待
完整回放，后在确认两个实现缺口后调整为受控停止，避免继续占用资源采集
不能用于主比较的数据。以下运行中观察保留原时间语境；最终结局见文末。

2026-10-24。只读、有界日志观察，不是新性能运行或独立数值证明。
完整 W2 仍在运行；原配置、输入、超时、调度和远端未改。

## 实际观察

读取 D787 `launch3b.launch/replay.jsonl` 按写入顺序的前 25 条
`http_raw_response`，不是请求编号最小的 25 条。所有样本的原生
`completion_token_ids` 的不同值集合都是 `[0]`，实际长度为 6–256。
其中出现 finance_lora / finance_lora_0025，原生 model_id 为 125 / 127。
这些响应仍带 `lora_numerical_correctness_qualified=false`。

以相同只读方式抽查已经封存的 D786 W1 前 8 条响应，不同值集合也均为
`[0]`；样本中包括 finance_lora 和 code_lora。未覆盖全部请求，不据此
宣称两个完整运行的所有输出都相同。没有修改 D786 的旧结果或结局身份。

来源：

- `/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d787_20261024/launch3b.launch/replay.jsonl`
- `/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d786_20261024/launch3b.launch/replay.jsonl`

复核方法：`rg -m 25 '"event":"http_raw_response"'`（D786 取 8），
以 jq 提取 request_id、native_adapter_id、native_model_id 和
completion_token_ids 的长度及 unique 集合。未输出完整 prompt 或权重，
未扫描/哈希整个工件池，也未新增 GPU 请求。

## 当前判断与边界

这是具体的输出退化线索，不能只用“数值正确性尚未知”略过，也不能仅凭
它断言 NaN、错误权重、采样器错误或原论文算法缺陷。既有内容审计的零
LoRA 更新与模型连续生成 token 0 是两个不同事实：零更新不意味着模型
应生成零 token。用户关于不重复检查全零工件的规则不被用作忽略新输出
异常的理由；本次不重开整池审计、不生成新权重。

原生 adapter 绑定、计数与时钟合格仍不等于模型计算数值合格；当前结果
不进入“全部正确完成”的 G1/G2 达标排名。保留完整失败、时延和资源数据，
不得作为正常数值语义下的基线性能劣势证据。

## 后续范围

先让已启动的一次 W2 按既定有限超时、安全与收尾规则结束并保留证据。
此后优先做有界来源核查：既有同请求原生输出、实际加载/采样路径、3B
模型兼容适配及原生 logits/算术证据；区分输出记录错误、模型数值问题和
可解释的真实生成结果。只有出现具体待验证路径才做最小正确性实验，
不是基线性能搜索，不盲目提高并发、重跑完整矩阵或改变超时。

如果确认适配错误，保存旧记录为受影响诊断，修复必要路径后只重跑受影响
执行键；否则明确已排除什么与仍未知什么。不得在没有此核查的情况下将
W1 的异常等待全部归因 dLoRA 原算法，或宣布 dLoRA 的全部资格已经完成。
其余 Loquetier/HydraServe 3B、PrimeLoRA 优化、A1–A5、S1–S13 与最终交付
仍完整保留，当前不并行启动任何重型任务。

## 后续只读核查：两个具体的 Llama-3.2 兼容缺口

同日继续核对实际 D787 配置、D782 source view、保留的官方源码和模型文件
头部；没有导入 torch、分配 GPU、改变运行源码或追加推理请求。以下是来源
和实现语义证据，不是对运行中 GPU 参数、logits 或 NaN 的直接测量。

### 1. tied output embedding 没有建立

- D787 的 `engine_args.model` 与 HTTP backbone 都指向现有
  `models/LLM-Research--Llama-3.2-3B-Instruct`，`use_dummy_weights=false`、
  `use_np_weights=false`、DP2/TP1/PP1。模型配置明确
  `tie_word_embeddings=true`，架构为 `LlamaForCausalLM`。
- 检查点索引有 254 个 tensor 名称，只有 `model.embed_tokens.weight`，没有
  `lm_head.weight`。又只读两个 safetensors 文件的头部（分别 21,680 和
  7,704 bytes，未读 tensor 内容），得到 187+67=254 个 tensor，确认不是
  只由索引缺项得出的推断。输入嵌入是 BF16、形状 `[128256,3072]`。
- 实际 source view 的 `vllm/model_executor/models/llama.py` 是保留 vendor
  文件的符号链接；构造器始终创建独立 `ColumnParallelLinear` 输出层，
  `perform_initialization=false`。其 `load_weights` 只按枚举到的检查点名
  复制权重，没有 tied 分支或加载后的共享操作。采样直接使用该输出层。
- `model_loader.get_model` 在 `use_dummy_weights=false` 时调用该原生
  `load_weights`，随后 `.cuda()`；原生 LORA worker 使用这一架构路径。
  wrapper/source-view 修改也没有建立 tied 关系。

因此，**当前声明的真实权重路径没有给独立输出层加载检查点权重，也没有
满足模型要求的共享关系**，是模型兼容缺口，不是 dLoRA 调度算法的性能
特性。它是 token 0 现象的具体候选原因；不能由此直接断言当前 GPU 上的
该 tensor 全零、NaN，或声称修正后所有数值问题都已解决。

官方参照：

- [dLoRA 保留版本的 Llama 实现](https://raw.githubusercontent.com/LLMServe/dLoRA-artifact/75f1c439446fe194b1df8a24982ef9067841fab5/vllm/model_executor/models/llama.py)：当前文件与本地该版本无 diff；旧实现没有上述 tied 处理。
- [vLLM v0.6.3.post1 的 Llama 实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.6.3.post1/vllm/model_executor/models/llama.py)：构造时按配置 tie head，加载时跳过独立 head。
- [Transformers v4.45.2 的模型基础实现](https://raw.githubusercontent.com/huggingface/transformers/v4.45.2/src/transformers/modeling_utils.py)：`tie_weights` 将输出层与输入嵌入关联。

这些参照支持必要模型语义适配，不授权替换 dLoRA 调度、迁移、合并或采样
策略，不需要升级整个原生后端，也不需要生成/复制权重。

### 2. llama3 RoPE 配置未传递

同一模型配置含 `rope_type=llama3`、factor=32、low/high frequency
factor=1/4、original context=8192、theta=500000。实际 Llama decoder 只
传 `rope_theta`；`PagedAttentionWithRoPE` 直接建立未缩放的频率缓存，
没有读取/传递 `rope_scaling`。当前文件与保留 vendor 版本同样无 diff。

[Transformers v4.45.2 的官方 RoPE 实现](https://raw.githubusercontent.com/huggingface/transformers/v4.45.2/src/transformers/modeling_rope_utils.py)
按波长分别保留、缩放和插值，并用 FP32 构造频率。按本模型配置做纯数学
计算，64 个频率中 29 个保留、29 个缩放、6 个插值；这不是仅在长于 8192
的请求才生效的配置。此数量是配置推导，不是当前运行的缓存数值测量。
不能因本轮上下文不超过 1024 就忽略这项兼容缺口，也不能在没有验证时把
它认定为 token 0 的唯一原因。

### 来源固定与下一次最小验证

| 小型来源 | SHA-256 |
|---|---|
| 实际 Llama 实现 | `beeeb731d1c79534726c7c9228810695000ef530ae51159dcf0dfecdf85e4ce1` |
| 实际 attention 实现 | `01967f67d5996e56cc773632f8c3e56d3f186fb04644e72daf250c3ee71e083f` |
| 模型 config.json | `39fb36dc5416f445ebc4e71cb71fbcf6727e80a35836d8ba1a1474c318467b7a` |
| 模型索引 | `0a57c30afc07c81610ae7fe013bea3b9d50365f9a8145ae98e8714220f3d7c7c` |
| shard 1 原始 header bytes | `48bea735386a79849b314920a28485d8aa59eab8ee67f7a4bd25083cb55cca7c` |
| shard 2 原始 header bytes | `3362160bc1e9babdfc4d5069629a78bfef214caffd28f98e60187b08bc847565` |

先按原协议收尾 D787，再复用现有 source-view 生成器作新身份的最小模型
适配，保留 vendor 和 D782 文件不变。一次有界正确性验证应核对：实际
输入/输出权重共享与加载、对应配置的 RoPE 数值、有限 logits/原生输出，
以及既有请求在可信实现下的模型计算一致性。只针对改变路径测试，不重做
整池零权重审计、不重开 baseline 性能参数搜索、不重复无关平台资格。

本次已发现具体实现缺陷，D786/D787 的绑定、计数和时间审计不能升格为
正确模型执行。旧运行完整保留为受影响诊断（R3），后续仅在修正且验证后
重跑受影响的 dLoRA 3B W1/W2 键；不是低 SLO 驱动重跑，不增加 dLoRA W0。
新的适配尚未实现或验证；当前不能声称已经修好、能够提高多少性能，或
由这些源码证据判定其他 baseline 存在同样问题。

## 最终结局：主动停止、完整保存实际发生的记录

2026-10-24，通过核验 PID/start_ticks、service invocation 与命令身份后的
pidfd，仅向本轮 native-http-service owner 发送 SIGTERM。复用其已有
finally 清理和外置 launcher 收尾，没有强杀其他进程、操作共享 k3s、改
采样/调度配置，或重复启动。停止原因是已证实的模型语义不匹配，不是低
SLO、结果胜负或观察超时；决策、发送时间和源码证据 SHA 已单独保存。

| 项目 | 实测记录 |
|---|---:|
| 原计划请求 | 4,000 |
| 实际到达 / 终态 | 1,364 / 1,364 |
| 实际发送 HTTP headers | 1,345 |
| 返回符合绑定/计数合同的响应 | 747 |
| 停止前失败 / 停止后失败 | 0 / 617 |
| 未到达，不能计作失败 | 2,636 |
| 原生输出 token 数 | 85,456 |
| 仅含 token ID 0 的响应 | 747 / 747 |
| 截断观察 GPU 时间 | 2,657.656482404098 GPU-s |

停止后的 617 条失败为 597 条 HTTP 错误对应 ValueError、19 条连接失败、
1 条连接断开。597 条错误状态的 raw HTTP 响应不能当模型成功响应。一次
离线核查曾将全部 raw HTTP 与服务保存的成功响应比较而失败；失败脚本和
回执保留，第二次在原数据上明确区分两者后通过。没有重跑 GPU 实验。

服务保存的 physical terminal 快照有 1,360 条，独立回放器最终终态有
1,364 条；差额为 req_01360–req_01363，发生在服务保存快照至回放器完成
停止之间。没有改写旧快照补齐；实际到达/终态以最后的独立回放日志为准。
物理 lease 均已 release，观察 GPU 时间由两卡 acquire/release 直接重算
并匹配已有 summary；完整 workload GPU 时间、GPU-s/correct-request、
正确完成数和正式 SLO 排名不成立，不用短运行的低资源量制造优势。

service/replay 返回 1，watchdog 返回 0；实际 GPU 已释放，service/launcher
已关闭，本轮两个远端 artifact 服务和 monitor 已停。500 个既有 adapter
已按发布 gzip 获取，线上 1,160,474,851 bytes，按请求打包为零。保存 54 个
Ray 日志、远端日志、resolver 记录和全部本地原始证据。模型数值适配仍未
修正，不能把此处的记录核查通过写成模型正确性通过。

后续复用原源码生成器、环境、模型与 LoRA/负载，仅修正必要模型语义并做
有界数值验证，再重跑受影响的 dLoRA 3B W1/W2。D786 和本次 D787 原结果
保持原样；其余基线与完整后续实验范围不变。
