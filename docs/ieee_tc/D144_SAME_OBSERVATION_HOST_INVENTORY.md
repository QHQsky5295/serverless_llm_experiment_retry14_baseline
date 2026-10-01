# D144：同次 HOST 观察内的等价复用

2026-10-01。单一开发候选，CPU 正确性检查已完成，普通 Full 尚未执行。
不声称 TTFT、TPOT、SLO 或生命周期 GPU-s 已改善。

## 1. 证据与可证伪假设

D143 已备份在 `f70595dc98fb9b2ae013171142bbb87085faa991`。它只覆盖
原 W0 的前 1,000 条请求，存在 GIL 采样偏差和 7 条截断栈，不是性能排名。
四个 GPU core 业务期 HOST inventory 栈分别出现 162/665、144/644、
156/643、157/641 次；`add` 的直接 HOST caller 分别是 150、135、140、142 次。
同名 GPU pool helper 的少量出现已单列，不能全部算作 HOST 操作。

静态代码和最小计数对照共同确认：同一串行 source_snapshot 中，registered
与 registered+staged 在 staged 为空时是同一个当前物理集合，却被遍历两次；
新 allocation 的同一个 view 又被查询两次 is_pinned。这里没有中间修改或
await，需要的是两份独立的计量数据，不是两次相同的原生属性检查。

假设：保留本次所有真实观察与校验，仅删除同次重复查询，可减少阻塞 native
worker 的同步工作。它不预言全部排队由此造成，也不预设完整回放一定获益。
若 Full 未改善，保留结果并回到阶段分解，不继续堆叠未验证优化。

## 2. 原始来源与适用边界

- [PyTorch v2.13.0 Memory.cpp](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/Memory.cpp)
  的 is_pinned 查询 storage 指针的原生 pinned 状态，并非读取 Python 缓存布尔值。
  本候选每个 view 仍查询一次；别名 view 仍分别检查，不缓存跨调用的结果。
- [vLLM v0.30.0 model_manager.py](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/model_manager.py)
  的 list_adapters 返回注册映射副本，不做 LRU touch；本机对应函数已核对。
- [vLLM v0.30.0 UniProcExecutor](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/executor/uniproc_executor.py)
  经 run_method 执行本地 worker 方法。本项目的同 owner 桥接采用该串行边界，
  不把它推广为任意 TP/PP、多 worker 或外部并发写入的保证。
- [vLLM CPU/GIL 分离说明](https://vllm.ai/blog/2024-09-05-perf-update)
  提供同步 CPU 工作可能妨碍推理推进的设计依据；本文不借用其加速比。

原始来源在本次优化前联网核查；PyTorch 2.13 文档页访问失败，改核对对应
tag 的实现，未以当前 stable 2.14 文档冒充所用版本。serving-llms-vllm
及 optimization reference 只作工作流参考，未采用其通用性能目标或扩并发建议。

## 3. 唯一候选与论文合同

修改只在 `faaslora/memory/gpu_monitor.py`：

1. 每个 HOST tensor view 的 pinning 值在 add 内观察一次，同时用于新
   allocation 记录及别名一致性检查。
2. source_snapshot 先生成当前 registered inventory。staged 非空时仍完整
   枚举联合集合；为空时深拷贝本次已测出的普通数据，保持两份返回值互不影响。

没有跨 epoch cache、过期判断替代、轮询周期调整、异常吞掉或兜底值；没有
改动 IEEE 九式、真实 footprint/alias 去重、owner/epoch/reference、content、
预算、admission、注册/驱逐/取消、动态 allocator 观察或 completion fence。
原有非法形状、额外 tensor、pinning 不一致仍报错。staged 和 registered
不能重名、不能把共享 storage 当作独占可回收字节的规则保持不变。

代价也保留：为空时仍复制 plain inventory，仍传输原有完整返回协议；
这不是消除了全部观测、序列化或 GPU 调度成本。

## 4. 验证结果与状态表

两项最小检查在旧实现上失败，准确复现 9 对 6 次 pinning 查询及 2 对 1 次
HOST 枚举；同测试在候选上通过（0.005 s）。扩展检查还覆盖后续调用新鲜度、
返回对象独立、staged 非空仍完整枚举。相关 628 项回归通过，77.150 s；
整条命令 86.41 s，峰值 RSS 1,122,388 KiB。

复用已有 NativeHostFootprint 微型 CPU 存储，与 Git 中精确旧 helper
逐项比较完整输出或错误，而非只对总字节相等：

| 样例 | 旧 / 新 pinning 查询数 | 完整输出或错误 |
|---|---:|---|
| 共享 storage | 9 / 6 | 相同 |
| 部分 view | 9 / 6 | 相同 |
| packed 缺省子模块 | 9 / 6 | 相同 |
| 空注册集合 | 0 / 0 | 相同 |
| registered/staged 共享 storage | 9 / 6 | 相同 |
| 不支持的额外 tensor | 4 / 2 | 相同拒绝 |
| 别名 pinning 不一致 | 7 / 4 | 相同拒绝 |

这是 CPU 测试环境 Torch 2.8.0+cu128、CUDA_VISIBLE_DEVICES 为空的组件
正确性/工作量证据，不是正式 vLLM0.30/Torch2.13 CUDA 性能测量；没有创建
真实模型、工件池或请求负载。计数减少不直接换算成服务速度百分比。
按 academic-plotting 和计划 §11，本阶段用表，不画暗示系统收益的性能图。

各 CPU 测试域实际 3/4 GiB、swap0、taskset2,3,26,27；退出后按已记录身份
核对空域并关闭，high/max/OOM 全零。失败的旧实现测试保留，不混成通过运行。

## 5. 下一步与未完成项

先完成 source/保护清单/秘密检查并备份候选；随后使用原 7B Full 4,000
请求、D137 配置和 D89 profiles，关闭详细观察器，进行完整普通回放。
后续以冻结 MetricV1 分析原生生成、全部请求、阶段时间和物理释放；不与
本次 prefix 直接拼成配对主结果。原始结果、失败与退化全部保留。

3B TPOT 和输出差异、两个模型的性能目标、数值 adapter 资格、共同
warm/Resident、基线资格及 M1/M2、A1–A5、S1–S13 均未由本候选完成。
不提高容量/超时，不重建已发布缓存，不跳到 baseline；普通 Full 之前不
叠加另一个优化，也不重复已完成的旧大日志投影。
