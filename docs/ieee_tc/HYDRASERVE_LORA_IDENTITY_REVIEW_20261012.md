# HydraServe：流水线与合并过程中的 LoRA 身份核查

2026-10-12。承接20261011只读复核；这是源码证据补充，不替代完整实验计划，
不增加性能矩阵，不表示已完成资格或观察到错误输出。当前唯一重型任务仍是
D445 Loquetier 完整4000回放。此次没有导入运行后端、执行测试、启动容器/GPU、
构建、安装、准备模型分片、修改远端或改变任何baseline算法/配置。

## 1. 核查身份与客观结论

官方checkout `vendor_new_baselines/HydraServe_main_20260520`，HEAD
`8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`，本次git status为空。
12个直接读取源码的SHA在
`paper_results/ieee_tc/hydraserve/20261012_readonly_lora_identity_review.json`。

官方实现以流水线缩短模型准备、随后合并流水线并重叠准备阶段为核心；应保留
这些机制及其修改后的vLLM 0.4.2，不因版本旧排除，也不以裸vLLM代替。
[官方仓库](https://github.com/LLMServe/hydraserve)、
[原论文入口](https://www.usenix.org/conference/nsdi26/presentation/lou)。

本次将“需要验证层映射”细化为以下可证伪的源码判断：

| 环节 | 固定源码事实 | 资格含义 |
|---|---|---|
| 基座分片 | `download/download.py:111`的transform把全局层号减去前一stage累计层数；:227将config层数改为本stage层数 | native模块名是局部编号，不能直接等同完整adapter的全局层名 |
| LoRA解析 | `vllm/lora/utils.py:77`只去掉PEFT前缀及A/B后缀，保留原始层号 | 完整LoRA字典仍是全局坐标 |
| LoRA绑定 | `vllm/lora/models.py:333`按当前模型的module_name取LoRA；缺项reset；:430按named_modules注册 | 仅“adapter ID正确”不足以证明每层tensor正确 |
| 合并目标 | `vllm/worker/model_runner.py:162`加载dest_model，但没有:178初始load_model的manager创建/注册步骤 | 新增层与复用层都需要真正LoRA绑定，不能只验证合并前 |
| 物理共享 | `vllm/model_executor/models/llama.py:513`把旧decoder对象引用给目标模型 | 不能直接假设两个独立manager对共享对象各自注册就安全 |
| 执行选择 | `vllm/worker/model_runner.py:796`先调用同一个manager的set_active，再依use_dest选择模型 | manager的覆盖范围、slot及batch映射必须和实际执行模型一致 |
| 请求传播 | `vllm/sequence.py:570`保留lora_request；`async_llm_engine.py:308–325`序列化整个ExecuteModelRequest并广播 | 底层已有身份载体；上层缺字段不等于必须重写原scheduler |
| 迁移 | `async_llm_engine.py:227–295`迁移KV后改变use_dest，保持原sequence对象 | 迁移前后的每层adapter数学意义必须一致，不能只看request_id不变 |

这里download路径相对`scripts/kubernetes/vllm/`。行号对应固定HEAD，不泛指未来版本。

## 2. 为什么只接通LoRA参数不够

设stage的全局起始层为o，本地decoder层号为i。正确绑定必须满足：

`backbone(stage,i) = backbone(global,o+i)`，
`LoRA(stage,i,adapter) = LoRA(global,o+i,adapter)`。

例：仅用于解释、不是实测分片结果，若两个stage覆盖全局0–15与16–31，
第二stage的`model.layers.0`对应全局16。完整adapter直接加载仍含全局
`model.layers.0`，现有按名查找会取全局0的LoRA，而不是16的LoRA。
这在形状相同的decoder层上可能不触发形状异常。没有运行数值见证，不能声称
已经测得该错误；但源码已足以否定“只加enable_lora就完成资格”的推论。

应使用官方实际分片映射验证全局→局部的双射及覆盖；不能假设平均分层。
允许的必要兼容适配候选是读原工件时的名称/视图映射，保持tensor内容、rank、
alpha和模型数学运算不变，不创建另一套LoRA权重池，不利用未来请求分布。
这只是待验证的适配方向，尚未实施或证明足够。

已有500逻辑ID多为零/重复权重；adapter名称正确、零权重输出相同或生成长度
正确均无法揭示错层。资格应复用既有非零工件和已有参考，不生成新权重；分别
检查映射、实际tensor/指针绑定及数值见证。跨后端仍不要求逐token精确相同。

## 3. 合并不能只“再建一个manager”

原生check_duplicate复用整个decoder对象，初始LoRA注册已经改写其子模块。
目标模型既有共享层，也可能有新增层。manager的_create_lora_modules还会
对模块写入mapping张量；from_layer在没有可替换类型时会返回原对象。
据此需要检验的风险是：再次注册可能让共享对象指向另一套batch/slot映射，
同时原模型尚未停止服务。当前不是已观察到的数据竞争结论。

必须区分以下事实，不能相互替代：

- 旧视图每层正确；目标新增层正确；目标复用层仍正确。
- old/destination共存时物理共享对象、LoRA slot及active映射的owner清楚。
- 请求切换use_dest和迁移KV前后，adapter及全局层坐标一致。
- 只有实际停止并释放的物理持卡区间才结束计量。

必要适配仍须保留原pipeline/consolidation/准备重叠、实例选择和原生批处理。
若无法在这些边界内实现并证明正确，就记录具体资格限制，不用关闭核心机制
后的普通vLLM输出代替HydraServe，也不把源码推理写成性能失败。

## 4. 发现的原始资产保护风险

`download/download.py:151`的partition在pp_size=1分支先加载模型，随后删除
传入目录内的.bin/.safetensors/.index.json等文件，再写单文件模型（:164–181）。
因此严禁将现有唯一受保护基座目录直接传给官方准备脚本。该判断来自读取源码，
本轮没有调用脚本，没有删除、转换或复制任何模型。

正式准备只能先寻找可复用、内容一致的历史布局；确需路径/元数据适配时保留
原工件不变，并遵循不重复整池、磁盘增长及内存包络。不以“官方命令”作为覆盖
用户模型的授权；也不能把只读元数据视图冒充已实现原生异步权重加载。

## 5. 下一资格问题与证据范围

按已有完整计划，在D445结束并封存之后继续：先确认可使用且归属明确的平台与
实际worker资源限制，然后证明上述层映射/共享/新增层/迁移绑定，再进入既定
100→500-ID→1000→4000资格。沿用20261011平台结论：存在k3s等服务不等于当前
用户已有可用GPU集群权限；不得改动未知现有集群、驱动或全局网络。

本记录没有新的运行通过标志。source推理、CPU见证、实际GPU执行、完整数值
与完整性能是不同证据层次。不会为得到好看结果优化此baseline或增加重复。
当前D445期间只读源码和小型文档工作；无第二重任务、远端管理或后台安装。
