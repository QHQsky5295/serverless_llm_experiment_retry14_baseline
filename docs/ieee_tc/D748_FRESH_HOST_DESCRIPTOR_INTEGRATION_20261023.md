# D748：实时 HOST 描述读取接入原清点路径

沿 D747 唯一候选继续，不新增优化方向。D747 原生组件语义一致，20组完整
清点均值8.575053→7.853158ms；71.717910ms长点全部保留。结果已以
19a5ec5360870c36c02a2100e7d83fd9b241fd01备份至
`backup/d747-native-descriptor-complete-20261023`，远端SHA已核对。
V2仍未同步，无force。发布时CSV的CRLF lint问题单独保留，未改测量或重跑。

## 单一实现变化

- `native_host_descriptor.py`仅缓存已验证的函数入口，不缓存Tensor、storage、
  pinning或HOST图。按进程启动环境显式选择库及SHA，仅当前Torch2.13/C Python
  3.12 ABI；未配置保持原Python路径，明确配置失败则报错，不静默降级。
- 原清点每个view调用一次实时描述，保留原dtype、alias、hidden字段、staging、
  requested scope、allocator/workspace及全部预算逻辑。worker回执记录实际库身份。
- 不改IEEE九公式、路由/准入/扩缩容，不改GC、学习规律或共同阈值。不恢复
  D329跨调用缓存、D685 pin-only、D687格式化、D728/D737/D744失败方向。
- 复用D747的195688B库与9种实际CUDA pin状态证据，不重建、不新跑微测。
  原诊断witness仅兼容新的显式分支，继续严格校验原访客结构，不用于生产改写。

2026-10-23再次核查官方
[vLLM0.30 LoRAModel](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/lora_model.py)
的Tensor构造/clone，以及
[PyTorch2.13 pinning](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)。
使用serving-llms-vllm的接入/测量原则，不采用其通用示例吞吐、TTFT或配置
替换本项目冻结指标。历史瓶颈和已拒绝方向见D747，来源代码与结果不改写。

## 有界验证与主线

只核查新loader配置/无fallback、生产图等价、状态新鲜度、实际原生loader
Tensor和clone输入；AST去掉显式选择后须与父版清点完全相同。复用已通过
CUDA状态和原图回归，绝不从CPU资格或组件均值推算完整SLO。

通过后冻结一次普通3B W0完整4000请求候选，复用工件、输入、D725 profile、
共同真实远端和资源合同；不做profiling前缀。保留原D74293.175%/U14868.152527
及D70393.6%比较，不只对比已撤回D745。D711数值限制继续，未达标不闭口。
全部3B后续/基线/dLoRA、7B最终同版本、A1–A5、S1–S13、CE/Pareto、文档/
图表均继续；共享k3s不动，既有HydraServe隔离结果复用。

## 最小接入核查已完成

9项接口/诊断兼容检查0.025秒通过；原清点AST去掉显式descriptor选择后与
19a5ec5父版逐结构相同。实际Torch2.13下完整/runtime/requested/staged四种
图对照一致；同一个Tensor改变storage容量后立即读到新值。

实际vLLM0.30 `from_lora_tensors`及clone产生的CPU Tensor图也与父版一致。
这项CPU检查显式关闭构造时pinning，不冒充新的CUDA证据；pinning资格直接
复用D747含注册/解除的9状态。没有重建库、运行新CUDA微测或完整性能回放。
资格峰值654299136B，high/max/OOM/swap0；145保护项和用户manifest不变。

使用system-profile/analyze-results保留组件与系统证据边界；本阶段唯一新增
生产路径是实时属性读取及身份回执，没有新的性能插桩。下一登记并执行一次
普通完整W0，资源状态须按已有规则重新检查；不重复上述资格，不开启支线。
