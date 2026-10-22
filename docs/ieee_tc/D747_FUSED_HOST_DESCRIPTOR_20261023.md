# D747：同次实时 HOST 清点的原生描述读取（独立组件候选）

当前仍为3B优化主线。D745完整W0退化至3687/4000联合时延达标，已在D746
精确撤回D744；生产保持D741。D742为3727/4000、U14868.152527 GPU-s，
历史D703为3744/4000；均未达3800。D711数值限制继续，不能宣布G1/G2合格。
本次不是完整性能实验，不改变基线、共同SLO、九公式或既有数据。

## 有界生命周期核查的结论

1. 当前 `ieee_pending_gpu_reference` 先登记pending，再做一次demand事务。
   pending登记不清点HOST物理图，不能据“合并入口”假定存在另一遍可复用清点。
2. 原生 `_add_adapter` 在注册前完成packing；`_load_adapter`、staged对象的
   packing和后续registration均可能替换CPU字段。native `clone`共享layer/
   tensor，`list_adapters/get_adapter`暴露原对象；当前owner监视removal与
   model incarnation，不是所有storage修改的独占所有权证明。
3. 因此D329的跨调用缓存反例仍成立，本次不实现epoch/ID/TTL容量缓存。
   不把源码允许的变化声称为W0中已观察到的变化，也不重复D329微测。

## 新假设与历史边界

D684现成诊断中，在已有推理工作的240次registered图pending/GPU事务内，
HOST清点均值79.687399ms，父事务累计20576.597195ms中的清点为
19124.975687ms。该比例是同一host区间的描述，不是可直接扣除的TPOT。
D742 GPU命中路径的较长获取等待进一步支持检查这个同步控制工作量。

新假设：**每个张量仍实时读取全部必要属性，但将CPU/dense/2D/nonempty、
storage pointer/capacity和原生is_pinned融合为一次Python→C++调用**，可减少
每view多次通用Python接口封装。alias图、dtype、隐藏字段、staging、requested
scope、workspace和allocator仍使用原实现，不改变读取时机或决策规则。

与已关闭方向明确区分：

- D685/D686仅批量pin，产生含Tensor的临时tuple列表；本次不使用其library、
  不直接调用CUDA、不延后pin查询、不保留批量Tensor列表、不调整GC。
- D687仅优化Python描述格式化，完整D688退化已回退；本次保留原格式化、
  原dtype和图构建，不恢复该补丁。
- D352/D663/D675已经消除合法同事务后检查，不重复实现。
- D691控制交错已生效，D728/D737/D744完整失败方向不恢复。

原始依据（2026-10-23联网核对，本地安装源码另存SHA，不假定与网页同字节）：

- [vLLM0.30 manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
  和[LoRAModel](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/lora_model.py)：注册、packing和共享引用。
- [PyTorch2.13 Python tensor接口](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/torch/csrc/autograd/python_variable.cpp)：
  Python属性分别进入native Tensor；本次只融合读取，不改变底层状态。
- [PyTorch2.13 pinning实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)：
  查询storage基址的accelerator状态。本次保留Tensor.is_pinned dispatcher。
- [vLLM官方性能分析](https://vllm.ai/blog/2024-09-05-perf-update)：CPU调度/对象工作
  可能阻塞推理推进；只借鉴定位方法，不移用其加速百分比或GPU结论。

## 预先固定的验证与停止规则

先做CPU组件替换/既有图回归和独立native库构建，约束3/4GiB、swap0、辅助CPU。
使用实际Torch2.13环境、精确Tensor；不宣称支持Tensor subclass/Parameter或
其他ABI。只读描述的GIL不释放，不创建/修改/保留权重，不调用GPU kernel。

CPU通过后最多一次现有backend-pin-check原生微测扩展：复用9种实际CUDA
pin状态、四种完整图、8×32层tiny fixture和20组预定交替完整清点。
逐字段等价与实际registered/unregistered pin变化均需成立；计入全部清点，
不只计C函数。不删除长点，不重复以得到好均值，不输出CI。
若完整组件开销不支持收益或语义失败，归档并返回主线；通过才考虑生产集成，
仍需普通完整W0判定，不能从组件结果宣告联合SLO/GPU-s改善。

本轮不跑新的profiling前缀，不新增工件池或trace，不操作远端/共享k3s。
旧145项保护、用户manifest、失败记录和全部未完成矩阵保留，目标前不绘图。
使用system-profile保持组件/系统证据分离，run-experiment复用资源约束。

变更范围仅诊断：`native_host_descriptor.cpp`、`host_descriptor_witness.py`、
`test_ieee_tc_host_descriptor_witness.py`及本次原始执行/小型报告；无生产插桩。
原始目录 `results/ieee_tc/p2_backend_qualification/d747_20261023/`，
小型证据 `paper_results/ieee_tc/p2_backend/d747_fused_host_descriptor/`。

PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
V1 SHA 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22；
单轮V2 SHA 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。

## CPU组件核查完成，尚无性能结果

| 已执行核查 | 结果 |
|---|---:|
| 图、projection、freshness及替换边界回归 | 49项通过，0.417秒 |
| 实际Torch2.13构建扩展后的CPU情形 | 12项通过 |
| 完整/runtime/requested/staged共享图逐字段对照 | 4种一致 |
| 新诊断入口分离及原pin入口回归 | 8项通过，0.016秒 |
| GPU/完整工作负载运行 | 0 |

CPU情形包括offset、转置、dtype、同Tensor底层storage resize，以及空/非二维/
meta/sparse/非Tensor/subclass/Parameter拒绝。后3项是该独立扩展明确限定的
输入范围，不宣称兼容全部PyTorch对象；vLLM生产接入仍须验证其实际输入范围。
CPU-visible环境没有初始化CUDA；因此这不是实际pinned注册/解除的资格证据。
49项含复用的原有类，不称为49个独立新实验；扩展读取结果没有性能计时。

只生成195688B独立库；构建出现上游header的C++20 bit-field默认初始化警告，
当前C++17编译成功、原生CPU检查通过，警告原文保留；不隐瞒也不重复构建。
限定当前实际ABI，未安装或替换环境包。资格域峰值1200140288B，high/max/
OOM/swap均0；145保护项及用户manifest不变。原资格域已精确空核验释放。

`ieee_tc_preflight.py`仅扩展显式 `--host-descriptor-library` 诊断选择，互斥
于旧pin-batch/GC候选；默认入口不变，无faaslora生产修改。准备好的launch.py
仍未执行，下一只运行一次原有native tiny微测，不再做完整profiling前缀。
生产D741与完整D742/D703结果身份保持不变。
