# D205：原生执行映射边界与检查器资格

## 本项回答什么

D204 发现两次相邻 7B Full 的 126 个输出摘要不同；提交记录中的原生 LoRA
整数标识均符合逻辑 ID，但这不能证明实际内核采用了对应的槽位。本项沿安装的
vLLM 0.30.0 追踪执行链，并验证一个离线映射检查器。**没有新 GPU 回放，
没有解释掉这 126 个差异，也没有改变任何 Full 的正确数或达标结论。**

复用 D166/D167 的 22 请求、12 逻辑 adapter、4 内容类别与 checkpoint/槽位
核验证据；不重复其 44 次 GPU 快照，不生成权重或负载。一次性交付缓存授权已由
D78/D80 完成，本项不连接远端、不重建缓存。九个公式、调度、预算、SLO、生成
合同与安装环境不变；新函数仅在已有 preflight 中，由测试/离线诊断显式调用。

## 找到的实际观测边界

| 环节 | 安装版本源码位置 | 能证明什么／尚缺什么 |
|---|---|---|
| 外部请求身份 | Prime `generate_prepared` 的 `begin_use`、backend request ID | 请求意图与持有引用；不能代替内核映射 |
| batch 排列 | `gpu_input_batch.py:make_lora_inputs` | 按当前请求顺序、scheduled/sample 数展开逻辑 ID；batch 重排后不能按旧行号关联 |
| 物理槽位转换 | `model_manager.py:set_adapter_mapping`、Punica `convert_mapping` | 元数据依赖逻辑映射和当前槽布局；安装版本已同时检查二者，不重复添加这个已有修复 |
| 设备元数据 | `punica_base.py`、`punica_gpu.py:update_metadata` | token 与 sampler 是两套映射，均需读回，不能只验证其一 |
| 内核分组 | `lora_kernel_metadata.py:prepare_tensors/meta_args` | 真正用于执行的是槽号、分组行索引、组计数、起点和执行标志；正确的高层 ID 不保证这些正确 |
| forward 边界 | `gpu_model_runner.py:_model_forward` | 已完成本 iteration 输入准备，实际 model 调用前；应从这里读回当前设备映射，而非只在 RPC 返回时观察 |
| 内核消费者 | `lora_shrink_op.py`、`lora_expand_op.py` | 按 active slot、count、start 和排序行执行；无 LoRA 分支、graph padding 与实际启动组数必须区别处理 |

官方依据：
[请求映射](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/lora_model_runner_mixin.py)、
[槽位管理](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)、
[内核元数据](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/ops/triton_ops/lora_kernel_metadata.py)、
[shrink 消费路径](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/ops/triton_ops/lora_shrink_op.py)。
本地 11 个实际文件的路径和 SHA 另存于结果 JSON；网页文本不替代安装身份。
Punica GPU 网页本次抓取失败，该部分以本地固定版本源码核查，不声称已联网读取。

## 检查器与可证伪假设

假设：正确提交身份到实际 token 分组之间可能出现错误或陈旧映射；当前记录
不足以排除。检查器必须能拒绝此类错误，而不是“看到任意标识就算通过”。

新增 `validate_native_lora_execution_metadata`，在同一批真实行上核对：

- 独立提供的 backend request ID→预期 adapter 绑定，与观测 batch 的绑定一致；
- 当前物理槽布局唯一、目标存在，token/sampler 槽号按请求计数展开一致；
- 实际内核分组覆盖每个真实行恰好一次，无遗漏、重复、越界或错误槽位；
- 有效组全部处于实际启动组数内，不把 padded 行冒充真实请求；
- token 与 sampler 两套元数据均一致；错误 no-LoRA 分支拒绝；
- 全 base 分支中，原生实现会重置分组后提前返回，不复制 mapping/order。
  此时这些未使用的陈旧数组不能误判为错误，也不能作为执行证据。

当前检查合同明确限于 dense、TP=1、非 speculative 的每请求一个 sampled row。
其他路径显式拒绝/未验证，不静默猜测。函数仅返回“所给元数据一致”，始终保留
`gpu_observation_qualified=false`、`kernel_arithmetic_qualified=false`、
`full_pool_qualified=false` 和 `n_correct=null`。真实设备/采集完整性另行资格，
不能由离线数组自行宣称。

## 结果与失败记录

| 项目 | 实际结果 | 解释 |
|---|---:|---|
| 新增映射检查器测试 | 10/10 通过 | 包含错身份、换槽后陈旧数组、重排、错 sampler、遗漏/重复行、错误分支等 |
| 首次扩大回归 | 384 项，2 failure、3 error | 4 项因用模型环境运行需 pidfd 的系统安全测试；1 项旧动态导入未登记模块，影响 dataclass |
| 更正运行职责后的模型环境检查 | 337/337 通过，21.462 s | 包含新增测试、旧工件检查、相关 preflight 与 smoke |
| 系统 Python 安全检查 | 47/47 通过，0.096 s | 不改安全检查，也不在不支持 pidfd 的解释器伪造该能力 |
| 安装版本元数据构造代码的 CPU 微测 | 8/8 场景通过 | prefill 混合、decode 重排、槽位重新分配、all-base；specialization 开/关 |
| CPU 微测错误对照 | 34/34 拒绝 | 外部身份、分支、重复行、启动组数、sampler 错槽；不是模型输出样本 |
| 前两次证据封存 | 均在写入汇总前拒绝 | 第一次未处理既有 INFO 前缀，第二次前缀正则转义错误；独立第三次分析，原记录保留，不重跑测试或 GPU |
| GPU 回放／性能样本 | 0 | 不宣称正确性或延迟已改善 |

原失败脚本、日志和源码归档全部保留。修复只涉及测试的正常动态模块登记方式，
并将系统安全测试交给原协议指定的 `/usr/bin/python3`；不改变生产实现和门槛。
微测直接加载安装文件中的 `LoRAKernelMeta`，在 CPU Torch 2.8.0 上运行：
它不是服务环境 Torch 2.13 的 GPU/编译图试验，不据此认定两种执行等价。
全部工作置于 3/4 GiB、swap=0、CPU 2,3,26,27 的资源域；无 high/max/OOM/swap。
模型环境测试末尾的只读保护校验与系统安全测试有短暂重叠；没有并发 GPU、构建
或推理测量。结束后逐资源域检查移除，147 项历史保护验证通过。
依计划 §11 与 academic-plotting，本项用精确状态表/CSV，不画性能排名。

## 下一步和不得外推的结论

下一项只增加隔离诊断采集：复用现有 worker extension、preflight 与这 22 条请求。
先在单 owner、TP=1、dense、非 speculative 路径验证，启动后装入 opt-in observer，
普通 Full 默认不安装、不付出同步读回开销。至少需要：

1. 外部请求绑定来自提交/引用合同，不能从观测 batch 反向构造“期望”。
2. 在实际 forward 边界读回设备元数据，记录 iteration、真实行数、padding、
   graph 模式、设备/进程身份；逐层 wrapper 和权重 buffer 引用需绑定到已核查槽位。
3. 不只包住 Python 的 LoRA 算子调用，因为 CUDA graph replay 可能不再执行该
   Python 入口。不能把仅 eager 诊断称为原 graph 路径合格。
4. 记录 prefill/decode 全部目标 iteration，缺失、提前停止和未支持路径保留；
   覆盖率不能用“收到一个快照”代替。内容核验与执行映射分别报告。
5. 实际采集资格通过后才进行一个小样本诊断；新 GPU 启动前重新核查磁盘，
   当前低于 150 GiB 不放行，先审核可重建缓存，不删除唯一证据或降低门槛。

这不是新的性能优化候选，不为检查器再跑一轮普通 Full。7B 数值正确性、126 个
输出差异、共同参考与旧/新 Prime 的 G1/G2 仍未闭合；之后才是 3B、外部基线、
主比较、消融和敏感性。不得把这次 CPU 资格写成完成这些实验。

原始目录：`results/ieee_tc/p2_backend_qualification/d205_20261003/`。
汇总：`paper_results/ieee_tc/p2_backend/20261003_d205_execution_metadata.*`。
