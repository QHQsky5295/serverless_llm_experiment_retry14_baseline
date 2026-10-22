# D749：实时HOST描述读取的完整3B W0验证

## 问题与预登记判定

延续D747/D748唯一候选：保持每view实时读取和原物理清点图，减少Python/native
接口封装，检验其能否改善完整服务，而非仅组件耗时。D747的71.717910ms长点
保留；不得用组件均值直接减去请求延迟或推算SLO。

直接父版D742：3727/4000=93.175%，U14868.152527434286 GPU-s；历史D703
3744/4000=93.6%同样保留。失败D7453687/4000已撤销，不仅以胜过它为成功。
仍按冻结V1/单轮V2的G1/G2、共同warm阈值判断；4000全完成与数值正确性
分别记录。D711对现有3B池的数值证明限制继续，不能自动填n_correct=4000。

本轮是已知固定负载上的开发完整验证，不改写为独立留出或正式统计证据。
同键仅一次，不添加CI、无关profiling前缀或短回放。未改善则保留并撤销候选，
不改SLO、GC、控制参数或对手，不筛请求/长点后宣称达标。

## 已完成证据直接复用

- D748备份890f1976f9849c038b71a03af638cc9b88fbee14，原清点AST等价；
  9项入口检查、4种图、实际vLLM CPU构造/clone和storage变化均已验证。
- D747实际CUDA9状态及4图、20组组件测量完整保留；复用195688B库，不重建。
- 代码历史、已拒绝方向和原始源码核查见D747/D748；本轮不新增生产优化。
- D725准备/服务profile、D648共同warm阈值、原3B seed42输入与500-ID工件池
  直接复用。真实远端沿用D80通知前发布的gzip；请求不打包，真实传输仍计时。

## 唯一执行差异

YAML与D742仅output/NVMe/HOST三个唯一目录不同。现有runner新增d749分支，
无prefix，原4000请求、60秒准备、TP1、FP16、c8、共同资源包络均不变。
显式候选环境也作为执行身份冻结，不是不可见调参：

```
FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so
FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044
```

启动前核对源/库/协议/保护SHA、既有资格释放、GPU idle及原150GiB/102GiB
准入；只新增两个runner合同检查。进程资源域、独立watchdog与真实GPU释放
复用既有入口，监督进程使用支持PID-handle signaling的系统Python。
worker原生回执必须显示同库SHA；未启用不能归因该候选。

原始目录`results/ieee_tc/p2_backend_qualification/d749_20261023/`；
汇总`paper_results/ieee_tc/p2_backend/d749_3b_fresh_host_descriptor_full_w0/`。
新缓存不覆盖D742/D745。运行中不构建、测试、hash扫描、压缩、推送或更改远端。
失败证据完整保留，结束后释放、分析、备份，再返回完整主线。

保留3B后续W1/W2/全部baseline/dLoRA、7B最终同版本、A1–A5/S1–S13、
CE/Pareto、两轮审稿文档和最终图表。共享k3s不动；HydraServe隔离结果复用。
