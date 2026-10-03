# D201：原生查询等待与引擎执行内容的关联

2026-10-03。离线诊断已完成；没有新增 GPU 回放或修改服务逻辑，不是性能验收。
沿用 D200 已封存数据与读取器，不生成工件、负载或交付缓存。一次性缓存授权
已由 D78/D80 实现，当前仍复用同一只读发布集合。

## 问题、方法和边界

D200 的 4,204 次完整查询中，frontend 发出原生查询到 native 开始平均
96.328020 ms；原生检查内部平均 12.315801 ms。两者不是纯排队和纯 CPU
时间，也不能从中直接推算请求 TTFT。本轮不重复 D197 的编码微测。

复用 D200 的有界 JSONL 读取器，先复核封存摘要和每份控制/采样文件 SHA，
再按调用 ID、实际进程 PID+start_ticks、原生执行线程和本机共同单调时钟匹配。
只有整个采样捕获窗口都位于 `[frontend_native_send, native_begin)` 内才计入；
跨边界窗口单列。同一进程的同一次采样即使遇到多个等待调用也只计一次。
完整成功、零 partial 的前置资格来自同 SHA 的 D199/D200 校验，不重新造分母。

| 原生进程 | 完整调用 | 有区间内采样的调用 | 区间内独立采样 | 最近项目帧为 HOST 深拷贝 |
|---|---:|---:|---:|---:|
| 2364655 | 1068 | 57 | 43 | 11 |
| 2368679 | 1050 | 49 | 34 | 12 |
| 2369096 | 1045 | 68 | 43 | 10 |
| 2369310 | 1041 | 52 | 39 | 15 |
| 合计 | 4204 | 226 | 159 | 48 |

159 个区间内采样均有完整主线程栈；另有 1 个跨边界捕获窗口。未采样调用
不是零成本。48 个深拷贝采样都落在 `gpu_monitor.py:940` 的 `copy.deepcopy(host)`；
其上层是原生 utility 调用。另有 59 个采样最近项目帧位于 HOST storage/pointer
检查的 `add:103`，以及其他加载、核查和无项目帧情况，全部保存在结果中。
这些计数不代表 CPU 百分比、墙钟占比或 48/159 的因果延迟贡献。

观察间隔为 2 秒且受 GIL 调度影响；不能观测 C/CUDA 内部，也不能将被较长
等待覆盖的样本视为随机请求样本。本轮仅支持“某些等待期间确实在执行这些
同步工作”，不能证明全部 96 ms 由它们导致。

## 官方实现、当前调用链和历史改动

- [vLLM 0.30.0 core](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
  在处理循环中串行处理 utility；`get_result()` 先执行，只有返回 Future 后
  才延后结果交付。不能把所有 utility 方法自动视为异步后台任务。
- [同版本 UniProcExecutor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)
  两种 non_block 分支均先调用 `run_method`。本机 TP1 的物理观察与准备因此
  可能占用服务原生查询的同一线程；这不是官方系统总体性能差的结论。
- [core client](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py)
  为 utility 建立 Future 并等待响应；client 的 async 不会把同步 owner 核查
  自动迁出 engine。以上均核对了本机安装代码，不仅依据架构图。
- [官方 CPU 优化分析](https://vllm.ai/blog/2024-09-05-perf-update)
  支持检查对象构建与同步工作这一方向；其加速比和旧版本配置不移用于本机。

Prime 当前已实现 async TCP、in-flight 共享、target-scoped routing、完整
planning 图和 HOST 事务；不重复这些已完成的优化。D144 用同次深拷贝替代
重复原生遍历，保持两份返回对象独立；D174 仅省去 routing 的 tensor 明细。
本次发现的是仍走完整 `source_snapshot` 的物理观察路径，不回退上述改动。

代码消费者核查：`NativeSourceSnapshot._footprints` 使用 allocation graph、
adapter edges、dtype 与 GPU pool；`_ieee_pinned_host_observation` 使用 allocation
容量、pinning 和 staged membership。仓库内 `host_tensor_views` 的显式引用
仅为生产者及测试，但不能据此省掉 GPU `pool_tensor_views`，后者确有消费者。

## 唯一下一候选：物理观察的按消费者表示

可证伪假设：在保持每次真实枚举、全部 allocation/alias/dtype/allocator 和
owner/epoch 检查的前提下，让物理控制观察不构建无消费者的 HOST tensor 描述，
可减少同步构建与后续同次独立复制成本。不是跨调用缓存、放宽新鲜度，也不是
直接共享两个可变返回对象；必须保留 D144 的返回隔离语义。

先核对所有动态传递消费者，复用现有真实快照与 CPU fixture 验证完整图、
预算/admission 输入、staged/non-staged、alias、替换和异常行为等价；确认
实际减少的工作后才允许改生产路径。合格候选再做普通 4,000 请求 Full，
同时看 TTFT/TPOT、物理 GPU-s、完整性和输出差异。若无净收益，不堆叠补丁。
本轮尚未实施该候选，不声称已减少延迟或满足 G1/G2。

## 安全、交付与主线

分析使用原 3/4 GiB、swap0、CPU 2,3,26,27 包络；峰值 109,821,952 B，
high/max/OOM/swap 全零，执行域已消失。8 个区间边界 fixture 通过；原有
147 个保护项通过。使用精确表格，遵循 academic-plotting 与计划 §11，
不为单次采样关联绘制“加速”图。原始关联栈与读取器随小型来源包保留。

7B 数值正确性、共同 warm/Resident、旧/新 Prime 的新指标比较及 G1/G2
仍未完成；没有将 4,204 次控制查询视为独立性能重复。顺序仍为 7B 验收→3B→
外部基线，M1/M2、A1–A5、S1–S13 全部保留。
