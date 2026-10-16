# D559：Loquetier 原始表示的对象生命周期验证

2026-10-16。用户已明确授权针对 Loquetier 异常低性能开展有依据的优化。
本次仅扩展现有 `tests/test_loquetier_native_generation.py`，没有改变原生源码、
正式服务、batch2 配置、分片或加载时机，也没有新的 GPU 性能结果。

## 为什么先验证所有权

D438 实测最大 125-ID 分片同时保留约 5.09 GiB 原始 FP32 LoRA 和约 3.91 GiB
准备后 FP16 表示。D445/D544 的主要延迟发生在原生请求登记前，说明需要
检查实际服务容量，而不能把全部延迟归因于 GPU 算术或远端传输。
当前接入明确保留 `bridge.native_models`；原生 `_prepare_weight` 可能复制、
缩放、补齐，也可能保留相同存储的视图，不能直接根据删除 Python 对象推算
可用显存。虚拟模块还可能通过绑定方法形成引用环。

已核对 D437/D438/D439/D441/D445/D544、接入提交 `1fd6740` 及后续历史。
本机官方 HEAD 为 `aae33baeeb19777129c1ccbff99a898d4a0e2c63`。
本次联网读取了[固定版本原生工作流](https://raw.githubusercontent.com/NJUDeepEngine/Loquetier/aae33baeeb19777129c1ccbff99a898d4a0e2c63/src/loquetier_src/model_workflow.py)，
保留其原生登记/generate 流程。mixed_lora.py 和 virtual_model.py 的同版本
在线请求返回 Cache miss；对象关系依据本地固定 checkout，不冒称在线逐字节验证。

## 实际测试与结果

在既有 `loquetier_20260520` 环境、GPU 隐藏条件下执行六项 CPU 检查，全部
通过，22.241 秒、无 skip。其中四项是原有生成边界检查，两项是新增所有权
检查，采用微型 CPU Llama、两个非零合成 LoRA 测试对象，覆盖不同 rank：

- FP32 原始表示 → FP16 准备表示：原始对象及 Parameter 弱引用均可回收，
  准备后的权重内容与指针、原生 A/B 指针表以及共享基座全部保持不变。
- FP16 同 rank 的别名情形：原始 Parameter 对象可回收，但准备张量继续
  持有共享存储；准备内容、指针与基座同样不变。不能把对象回收直接等同于
  所有原始存储字节释放。
- 对每个 adapter 的 CPU 线性参考计算在回收前后完全一致。

使用实际 `create_virtual_model`、PEFT 和 `MixedLoraModel` 构造，不以自制
对象替代这些所有权关系；小型参考运算明确不是原生 SMLM GPU forward。
合成对象仅用于测试，没有写入或改变实验模型、LoRA 工件池、trace 或输入合同。
PEFT 提示共享基座已有 `peft_config` 的 warning 被保留；基座参数及指针检查
通过，未隐藏 warning，也不据此宣称完整模型数值正确性。

检查受 CPU2,3,26,27、memory.high3/max4 GiB、swap0 限制；峰值578465792 B，
high/max/OOM/kill/swap均0。实际进程树递归为空后按准确身份关闭，路径消失。
原始证据在 `results/ieee_tc/loquetier_qualification/d559_20261016/`。

## 下一关与不支持的结论

下一步复用现有 `--native-capacity-probe`，新增明确 opt-in 的所有权诊断：
同一最大分片、原有工件、同一基座/精度/静态映射，核对真实准备张量和共享
基座、回收前后原生 GPU 输出，以及实际 allocated/reserved/峰值变化。
垃圾回收如必要只在部署期进行，时间计入准备，不放进逐请求路径。
之后才根据真实 KV/workspace 余量确定安全并发，不照搬官方 batch16。

本次尚未释放生产模型对象、测得 GPU 显存节省或修改 serving；不能据此
宣布延迟、吞吐、SLO 或 GPU-s 改善。旧 D445/D544 及失败结果保持封存，
未来合格新配置使用新执行键，跨 W0/W1/W2 固定。Loquetier 的训练能力不在
本次纯推理生命周期改进范围，原生虚拟模块、SMLM 和生成机制必须保留。
