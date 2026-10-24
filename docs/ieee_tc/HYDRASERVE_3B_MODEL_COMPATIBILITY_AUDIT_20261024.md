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

## 16:32补充：共享权重的元数据与实际checkpoint dtype

D802运行期间继续只读核查，没有构建/加载模型、改动运行源码或启动新性能
实验。以下事实细化必要适配，避免照搬dLoRA的PP1补丁而遗漏HydraServe路径。

### 两个原生映射接口必须区分

本地保留源码以及本轮重新核查的
[HydraServe官方download实现](https://raw.githubusercontent.com/LLMServe/HydraServe/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/scripts/kubernetes/vllm/download/download.py)
表明：download侧`transform/get_layer_distribution`接受`_tied_weights_keys`；
provider侧接受`tie_word_embeddings`布尔值。现有环境Llama类的
`_tied_weights_keys`确为`["lm_head.weight"]`，不能向两种接口机械地传同一参数。

在tied情况下，embedding的owner是首、末stage列表，head owner为空列表；
decoder仍是单个stage。现有7B metadata生成器传`None`，并按`owner==rank`
判断归属，只适合已经限定的untied模型。3B扩展须保持原生partition heuristic，
使用实际共享关系，按所有owner成员生成首末stage及合并duplicate映射。PP1的
`[0,0]`是同一归属而非两个独立参数；不可由此重复计量或复制权重。

这不是现有7B结果错误的证据。旧入口明确拒绝3B，说明此前的资格边界正确；
必要改动是扩展真实模型语义，而不是去掉拒绝条件掩盖未实现路径。

### 保留原生head模块，仅共享其参数

重新核查[vLLM v0.6.3.post1 head实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.6.3.post1/vllm/model_executor/layers/vocab_parallel_embedding.py)：
非GGUF路径保留`ParallelLMHead`对象，只让其weight关联embedding参数。
这与dLoRA旧PP1适配中直接复用整个embedding模块不是同一个结构。

HydraServe现有LoRA manager按模块身份登记embedding/head，head另关联
logits processor；consolidation又复用旧模块对象。后续实现应保留这些关系，
在首/末stage按配置提供embedding，并检查共享参数在原生CPU加载、GPU转移、
LoRA包装及destination注册之后仍成立。不能仅凭两个tensor值相等宣称同一
物理参数，也不能为了让绑定通过替换native LoRA manager或关闭合并机制。

必要测试范围因此包括当前参数alias、stage名字映射和实际prefill/paged decode；
不重新开启已完成的Kubernetes/容器平台资格，不把未发生的多stage性能当作已测。

### 原checkpoint为BF16，不应沿用7B的FP16-header限制

本轮复用`_tensor_header`函数，仅读取两个现有safetensors的头部与索引，
没有读取完整tensor payload、批量扫描工件池或导入Torch。结果为254个tensor，
全部BF16；embedding形状128256×3072，788004864 B，仍无独立lm_head。
头部身份：

- shard1：`58e68073fda43c094068939b0508e97f6e295cab178a83ec9464ade55688b42a`；
- shard2：`a13e271ba00a58d5a4d1ce7fe08c1178b8d6299fa5a7302a696a58ce06d7f2d0`。

这些SHA只标识header，不是完整权重SHA或数值证明。原生provider本来就显式
`dtype="float16"`加载，并按实际参数序列化给worker；应保留该运行时转换，
不重写原模型、保存第二套FP16 checkpoint或把原BF16文件标签改成FP16。
BF16/F16元素同为2字节，header可以用于原生partition的大小计算；最终加载
值及实际服务dtype仍由必要模型资格确认，不能只靠header判正确。

当前结论仍是静态适配范围已经更明确，HydraServe 3B运行正确性尚未证明。
本补充随D802结束后的正常检查点备份，不在性能测量期间push。

## 16:49补充：独立代码准备及27项CPU检查，尚无GPU资格

D802继续原配置运行期间，仅编辑HydraServe独立适配代码并在CPU2执行不足
一秒的标准库单元检查；没有模型/Torch导入、构建、GPU probe、远端操作、
大文件扫描、提交或push。没有修改Loquetier运行源码或本轮输入/配置。

新增可显式启用的`runtime/llama_model_compat.py`，经原生文件SHA绑定，在
新source overlay里加入Llama3频率变换，首/末stage的tied embedding及
参数级head共享。原attention/cache kernel和`check_duplicate`函数不变，
旧`native_delivery.render_overlays`默认输出不变；不升级整个后端。
共享embedding若已经被原生LoRA包装，取其真实base参数，不替换head模块。

本次还发现并修正了上一静态推断的不充分之处：provider显式dtype=float16
**不等于已保证转换**。原生`default_weight_loader`、embedding和部分linear
路径直接`param.data = loaded_weight`，而另一些packed路径用copy_，3B的
BF16原检查点可能因此形成混合dtype。因此在原Llama权重分配之后、调用
原weight_loader之前，按该参数选定的实际dtype转换；同dtype保持原tensor。
这保留原零拷贝/共享加载顺序，不新建FP16检查点，也不改原BF16文件标签。
另外，原流水线按checkpoint config.torch_dtype接收，而服务选定dtype可
不同；接收类型改取实际模型参数dtype，保留原socket与激活传输机制。
这些是实际源码证明需要处理的模型契约，运行效果仍待GPU资格确认。

现有`probe_layer_identity.py`扩展为显式选择既有7B/3B模型及匹配工件池，
仅读取header；仍用原生get_layer_distribution和transform。BF16/F16权重
允许用真实两字节大小参与分片，非权重F32旋转buffer单独记录。元数据按
owner成员关系处理tied embedding首末归属，并为末stage到末stage的共享
head保存模块别名，使原生合并与LoRA logits所有权一致；不增加独立head
payload。所有权重、tokenizer、索引仍为既有文件引用，未复制数据池。

验证命令为baseline仓库下`python3 -m unittest`三个模块：
`tests.test_ieee_tc_hydraserve_llama_model_compat`、
`tests.test_ieee_tc_hydraserve_native_delivery`、
`tests.test_ieee_tc_hydraserve_layer_identity`，绑定CPU2、禁写pyc。
最终27/27通过，0.811秒。包括原7B元数据/overlay回归、真实3B header的
28层/254个BF16 tensor和112个LoRA模块坐标、首末归属、标量频段定义、
构造器参数别名、加载dtype及接收dtype。对象与dtype测试使用明确的惰性
Python替身；它们不证明实际GPU alias、prefill、paged decode或数值一致性。

文件身份、命令与资格边界保存于
`paper_results/ieee_tc/baseline_audit/hydraserve_3b_model_compat_cpu_20261024/receipt.json`。
当前BASE HEAD仍8594922093ee4625dcc6614ecc11746af277219d，六个任务文件
尚未提交。待D802测量结束再作安全检查点备份；BASE新提交之后须相应更新
D802后置备份的跨仓身份核对，不把当前准备代码冒称D802实际执行版本。
下一先完成D802收尾，再以这份隔离适配做必要GPU模型资格和HydraServe完整
场景；不增加无关平台资格或为低SLO追加baseline参数搜索。

## 17:08补充：复用数值验证入口，统一provider与worker实际精度

D802仍在执行原W2，本轮未启动第二个GPU任务。仅扩展原有
`probe_layer_identity.py`，默认仍为CPU header检查；新增显式数值模式，
复用D790的两个真实输入和Loquetier已有`_retained_model_inputs`校验器。
这些输入是两个逻辑adapter ID，不因此称为两份独立训练权重。旧生成结果
不作正确答案；拟逐请求比较prefill及后两步paged decode的完整logits。

使用实际HydraServe Worker/ModelRunner的cache分配、输入准备、LoRA激活和
sampling，只在`compute_logits`处取只读观测。没有重写slot mapping、padding
或KV layout，避免重演D789验证器手工输入dtype错误。该检查为PP1模型算术，
不能代替provider/shared-memory、真实pipeline/consolidation或4000请求证明。
实际GPU共享参数、dtype与RoPE cache都仍待运行，当前不报告数值通过。

重新联网核对[HydraServe原ModelRunner](https://raw.githubusercontent.com/LLMServe/HydraServe/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/vllm/worker/model_runner.py)
和[Transformers Llama实现](https://raw.githubusercontent.com/huggingface/transformers/v4.45.2/src/transformers/models/llama/modeling_llama.py)，
沿用先前相对L2≤0.01、cosine≥0.999、greedy token一致的数值判据。
RoPE cache单独预定绝对误差≤1e-3，容纳旧Transformers标量循环和向量实现
运算顺序造成的约一个FP16 ULP差异；不是看到GPU失败后放宽。

HydraServe已有环境是Transformers4.43.1且没有PEFT；不升级该后端。参考
阶段拟复用`loquetier_20260520/bin/python`现有HF4.45.2/PEFT环境，在同一
受限服务域内顺序执行；先释放原生模型引用，只存约3MiB非pickle logits，
不复制checkpoint或LoRA。原生和参考两个入口都先核验真实service边界，
之后才导入Torch。参考依赖可用性与真实运行尚未新验证，不能凭源码准备
宣称成功；异常保留原生logits及参考日志，不打依赖兜底补丁。

另外确认了一个此前未覆盖的精度差异：
[官方provider](https://raw.githubusercontent.com/LLMServe/HydraServe/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/scripts/kubernetes/vllm/storage_server/remote_server.py)
显式选FP16，而[原app](https://raw.githubusercontent.com/LLMServe/HydraServe/8ae605de354ccfa2e9095514cdb4a9e9c56aa56b/scripts/image/app.py)
三个initial/destination EngineArgs均未指定dtype。原`dtype=auto`对BF16的
3B配置会选BF16，不能由provider声明推断整个路径同为FP16。新的模型opt-in
从SHA绑定的原provider AST读取dtype，在三个app入口一致声明它；不是搜索
精度或改原权重。旧默认overlay字节不变，先前27项检查的历史身份保留；
此次新增opt-in除两个模型模块外，也改变app的这三个dtype参数。

最终31项标准库CPU检查通过，0.840秒；没有Torch/GPU运行、构建、安装、
平台维护或基线性能调参。新回执单独保存为
`paper_results/ieee_tc/baseline_audit/hydraserve_3b_probe_preparation_20261024/receipt.json`，
不覆盖上一份27项回执。六个BASE任务文件仍未提交，当前执行HEAD仍8594922。
待D802真实结束，先封存其原版本结果/回收重复缓存/MAIN备份，再单独备份
HydraServe适配并核验最小受影响模型路径；不要把新源码追记为D802执行代码。

## 17:21补充：受控数值检查的启动材料已准备，尚未执行

复用现有`gated-launch`、D576的环境/动态库配置、原`materialize_metadata`
和`render_overlays`，在新目录`hydraserve_qualification/d803_20261024`准备
三个小入口：prepare、run_probe和原有精确归属close包装。没有新建实验框架，
没有运行入口、物化metadata、导入Torch、复制模型或启动GPU。D802仍在同一
W2测量；本轮只在CPU2检查这三个源码的语法与静态绑定。

prepare明确在D802结果封存及本地资源释放后运行，沿用辅助域3/4GiB和实际
推理机启动门槛。metadata只有现有权重的链接及原生分片元数据；运行入口
读取新manifest，使用已验证HydraServe安装及NCCL，不安装新依赖、不改k3s。
模型probe仍仅PP1，参考为已有Loquetier环境的顺序HF/PEFT计算；真实远端
交付、provider/shared-memory、pipeline/consolidation不由此自动获得资格。
这些范围须在后续实际完整场景及受影响路径中继续验证，不能省略主线。

准备源码SHA：

- prepare.py：`b0e46ef7fcc2f87162dd5dc831dde50c6162d1fcb366520272b5aa8282cf5c89`；
- run_probe.py：`580b5251e1e51ff7e207ca28f4194af2f63a47e65a595b51bad68213148d8379`；
- close.py：`acdfcccb84761199785d00d675cd2c52bc283eb4f724799e82a09774e19c8c13`。

本次重新核对官方ModelRunner的实际input preparation、采样输出和本地原生
worker/cache加载接口；仍使用此前预定数值阈值，不把既有两个逻辑adapter
解释成两份独立训练权重。当前没有D803结果，不能声称数值验证已经通过。
