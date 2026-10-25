# D865：将已验证的新鲜HOST图接入完整服务路径

## 依据与唯一变化

本轮完整重读计划及冻结V1/单轮V2。D864已完成单次实际pin/图验证：9种状态、
4种图一致；20对全部保留，中位耗时4.609142→3.6443135ms，20/20更短。
均值含参照119.94ms长点，不用64.67%均值改善冒充稳定典型加速。生产接入
仍需完整W2才能判断是否改善D858的3746/4000联合SLO及11311.401341 GPU-s。

沿用D864同一个252008B库，SHA
30455d8f678483eb1adf3f7e594ffb6bef90a7aad462fbc03efbd98b969edf80。
`FAASLORA_IEEE_HOST_INVENTORY_MODE=graph`明确选择整图入口；默认descriptor
保留历史路径。绝对库路径、SHA、ABI核验及无静默回退规则保留。只缓存代码
入口，不缓存对象、storage、容量或pinning。已选择graph但缺库/入口时直接
报错，不以旧路径隐藏候选没有实际执行。

gpu_monitor保留原manager快照、requested/staged范围验证；两条路径共用
同一结果作用域与共享容量计算。原生图逐view实时检查，无延后Tensor列表、
GC修改、新线程、TTL或未来负载。九公式、路由/规划/准入/扩缩容配置不变。
整图持有GIL的潜在影响由完整负载判断，不由组件实验直接接受性能。

## 历史和官方实现核对

不恢复D329缓存、D685/686延后pin、D687字符串投影、D851–854全owner线程、
D844/856 ctypes-stat或D860文件projection已失败方向。D671/675/680/747/
751/857已有有效改进保留。D863控制交叠89.61%集中于四类共同HOST检查入口
是当前瓶颈依据；交叠时间不是可直接扣除的时延。

本轮重新联网核查：

- [vLLM CPU路径分析](https://vllm.ai/blog/2024-09-05-perf-update)：对象处理可能阻塞生成；不借用其加速数值。
- [vLLM 0.30 LoRAModel](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/lora_model.py)：clone共享底层Tensor，必须保留storage别名图。
- [PyTorch 2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)：pin状态使用当前storage的原生查询，不换成缓存标记。

## 已完成的必要接入验证

26项定向检查通过；原生选择、缺入口报错、作用域拒绝、图错误不回退、
完整/runtime/requested/staged四种生产图、即时替换/resize和既有异常检查
均覆盖。移除显式选择并展开共享结果函数后，与e36b530c父版清点AST完全
相同。实际vLLM0.30 from_lora_tensors及clone生成的图也与旧D747库参照一致。
构造fixture时关闭pinning，仅为CPU接口验证；真实9状态复用D864，不再重跑。

辅助域峰值651141120B，high/max/OOM/swap0，CUDA未初始化。147保护项、
用户manifest和旧D747库不变；没有新工件、负载、backbone、GPU微测或性能
回放。复用D748验证脚本，不重建框架。主改动：native_host_descriptor.py
的显式入口选择、gpu_monitor.py的同图接入与共同结果投影；无性能插桩。

下一只登记完整3B W2新执行键。沿用D858配置、D725 profile、D648阈值、
D657预算12083.063870GPU-s，唯一变化为HOST图实现。无1000前缀、无CI、
不调阈值/对手、不重跑同键择优。数值正确性D711仍未知，不能填n_correct。
同版本两模型全场景、baseline/dLoRA、A1–A5、S1–S13、数值证明、论文图文
和最终V2同步全部保留。此次接入检查不是目标完成。
