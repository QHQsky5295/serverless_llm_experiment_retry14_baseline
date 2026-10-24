# Loquetier 3B：必要模型兼容性核查（只读阶段）

2026-10-24，D792 dLoRA W2运行期间。此记录不是新的性能实验、资格通过或
Loquetier调参决定；没有启动模型、构建内核、改变环境或修改运行中的系统。
主线仍是D792完整收尾后，完成3B Loquetier/HydraServe必要适配及既定矩阵。

## 可复用的已有实现

- 官方来源：`NJUDeepEngine/Loquetier`，commit
  `aae33baeeb19777129c1ccbff99a898d4a0e2c63`。
- D589原生配置：
  `/home/qhq/serverless_llm_experiment/results/ieee_tc/loquetier_qualification/d589_20261016/native_7b.json`。
- 源码观测视图为D560，manifest SHA
  `58a57e6be6027e40b2c06fc324e0f50c56ffba496bb64ecd3dc78790e0525c0c`。
  其中`loquetier_src/models/mixed_llama.py`链接到原生vendor源码，非`src/`子目录。
- 已登记原生内核SHA
  `100efcf29d3f8a53d8b2fe31b877d1589634024c742c8fff6ff30110f5a57ec9`；
  这是D589身份记录，不表示本次已重新哈希或验证3B运行。
- 7B使用原生SMLM、静态四分片、顺序启动、batch 8；这些已完成的机制/计量
  适配优先复用。不重新搜索7B配置，也不把7B容量测量当作3B容量证明。

## 发现：推理RoPE参数存在具体的模型语义缺口

本地3B模型配置为24个query heads、8个KV heads、head_dim 128、
`rope_theta=500000`，并指定`rope_type=llama3`、factor 32、低/高频系数1/4、
原始上下文8192；`tie_word_embeddings=true`。

原生`mixed_llama.py`的训练路径使用`self.rotary_emb`，但推理prefill/decode
路径分别调用`batch_prefill`/`batch_decode`，只传`self.rope_theta`。
已安装Python ops接口也只有`base`，没有上述分频参数。官方内核接口固定
`RotaryMode::kLlama`和`rope_scale=1.f`。因此，不能由训练路径支持RoPE或
成功载入config，推断原生推理已经实现Llama-3.2的分频缩放。
[官方模型源码](https://raw.githubusercontent.com/NJUDeepEngine/Loquetier/aae33baeeb19777129c1ccbff99a898d4a0e2c63/src/loquetier_src/models/mixed_llama.py)、
[官方内核入口](https://raw.githubusercontent.com/NJUDeepEngine/Loquetier/aae33baeeb19777129c1ccbff99a898d4a0e2c63/kernel_src/csrc/flashinfer_adapter/flashinfer_all.cu)。

Transformers v4.45.2的`_compute_llama3_parameters`依据波长处理低频、中频
和高频，中频做平滑插值；该计算不是仅改theta，也不以当前请求是否超过
8192 tokens为触发条件。因此，context上限1024不能自行消除这项差异。
[官方参考实现](https://raw.githubusercontent.com/huggingface/transformers/v4.45.2/src/transformers/modeling_rope_utils.py)。

这是对源码与模型合同的判断，尚未测得该缺口对3B logits或性能的量化影响。
不据此否定Loquetier的算法或已有Llama-2-7B实验。

## 尚不能认定的其他问题

- GQA group size应为24/8=3。本地生成的`dispatch.inc`已经列出3，虽然
  `kernel_src/setup.py`的默认组列表为1/2/4/8。不能仅凭默认值宣布本机二进制
  缺少group 3，更不能因此盲目重编译；后续先复用构建身份/已有资格证据。
- 原生模型有`_tied_weights_keys`、输入/输出embedding访问器和`post_init`。
  不能把dLoRA此前的tied-head缺陷直接复制为Loquetier缺陷；需要实际载入证据。
- 本次不重开已结束的工件池填充/零权重审计，不生成权重或负载；这是基座模型
  数值语义检查，与既有工件范围限制分别记录。

## 有界后续与停止条件

### 同日历史复核补充

已读取baseline仓库的`docs/NEW_SERVERLESS_BASELINES_2026-05-18.md`及
`Loquetier_project/evidence/real_adapt_2026-05-20.json`：当时明确构建了
`LOQUETIER_GROUP_SIZES=1,2,3,4,8`，3B的256-adapter/1024-request小规模
检查有1024个返回，累计2048输出tokens；另有500-adapter单卡物化OOM记录。
这支持复用现有环境/真实LoRA装载适配，并避免重复单卡500池失败，但不证明
固定输出主合同或模型数值正确。旧单卡/小规模结果不替代新的静态分片主实验。

进一步查阅本地native FlashInfer的`prefill.cuh`：query读写及RoPE位置推进
使用整数`8/group_size`、`16/group_size`，两个相关kernel中的
`static_assert(8 % group_size == 0)`当前是注释。group 3虽有dispatch入口，
但这些索引假设仍需验证，不能仅以成功编译/返回输出作为正确性证明。本轮
没有确定该注释的修改来源，不将其归因于本地补丁。该疑点与RoPE检查合并为
同一次有界3B原生prefill/decode数值核查，不另外增加无问题依据的重放。

待D792完全结束并释放资源后，先确认实际使用的内核和模型路径。若确证需要
兼容修改，只在新的隔离版本中补齐模型计算语义，保留SMLM、静态路由、队列、
批处理与冻结性能配置。复用既有probe流程验证受影响的prefill/decode、LoRA
绑定和共享head；不另加没有具体问题的全套重复门禁。必要正确性和计量通过后，
进入既定完整3B矩阵，不因为SLO低而进行baseline参数搜索。

本次按`research-lookup`的原始来源核查原则进行有界检索；默认服务无认证，
使用可用网页工具验证官方源码。未上传私有项目数据、安装依赖或改变运行环境。

## 09:14补充：分组约束的来源与可检验索引例子

继续只读核查官方固定commit的
`kernel_src/third_party/flashinfer/include/flashinfer/prefill.cuh`，其single和
paged prefill中同样已有注释形式的`static_assert(8 % group_size == 0)`。
本地该完整路径是git tracked，针对该文件的git diff为空；因此不再把注释
来源列为本地补丁的未解怀疑。官方`.gitmodules`仅列CUTLASS，不应在这个
FlashInfer目录得到顶层HEAD后将其误认成独立FlashInfer子模块版本。
[官方prefill源码](https://raw.githubusercontent.com/NJUDeepEngine/Loquetier/aae33baeeb19777129c1ccbff99a898d4a0e2c63/kernel_src/third_party/flashinfer/include/flashinfer/prefill.cuh)、
[官方submodule清单](https://raw.githubusercontent.com/NJUDeepEngine/Loquetier/aae33baeeb19777129c1ccbff99a898d4a0e2c63/.gitmodules)。

存在一个无需执行GPU的具体索引检查：`load_q_global_smem`初始行映射使用
`token=floor((tx/4)/group_size)`、`head=(tx/4)%group_size`，下一8行通过
增加`floor(8/group_size)`个token stride实现，head不变。当group_size=3、
初始打包行为0时，这会从(token0,head0)移至(token2,head0)，而行8对应
(token2,head2)。同一kernel的causal mask却按完整行偏移再除group_size。
这说明只补RoPE参数不足以消除prefill的具体索引疑点；成功dispatch和返回
结果不能替代数值对照。它是源码算术分析，不是本轮已测logit误差或性能损失。

下一次有界原生数值核查需同时覆盖非8整除的GQA3、prefill及decode、实际
Llama-3.2 RoPE与LoRA绑定。先复用现有环境和已登记内核，不盲目重建或换成
其他推理系统；没有新增性能参数搜索、输入工件或GPU实验。D792仍运行中，
本补充不改变它的配置和测量路径。
