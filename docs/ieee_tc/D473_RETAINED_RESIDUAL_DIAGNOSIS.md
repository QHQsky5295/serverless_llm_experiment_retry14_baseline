# D473：CPU 准备后仍存在的解码与控制路径差距

2026-10-12。继续 7B 主线；复用 D472 已封存的完整 4,000 请求，不重跑
性能点、不改生产服务、不调 baseline。使用 analyze-results 和
optimize-for-gpu 的证据优先流程；此项是原因定位，不是新优化的性能验收。

## 1. 结论与完整阶段表

D472 达到共同联合 SLO 3807/4000，物理 GPU-s/request 为 2.726165169。
这仍高于 ElasticLocality 的 2.473178968（+10.2292%），不宣称已经最优。
这些已封存的共同合同指标没有重算或替换，见 D472_FULL_W0_RESULT.md。
本项重用现有十二段计时分解：4000/4000 字段齐全，恒等式最大误差 0 ms，
原始 TTFT/E2E 均值与 Type-1 P95 与封存统计一致。

下表是每请求阶段均值，单位 ms；各阶段均值可加和，分位数不可加和。
ElasticLocality 和此前 PrimeLoRA 的阶段值直接复用 D459/D466，未重复分析。

| 互斥阶段 | ElasticLocality D453 | PrimeLoRA D465 | PrimeLoRA D472 |
|---|---:|---:|---:|
| 计划到达到回放释放 | 6.136 | 72.897 | 85.803 |
| 释放到全局许可申请 | 14.925 | 30.469 | 27.251 |
| 全局许可申请到恢复 | 356.170 | 398.142 | 309.669 |
| runtime slot 等待 | 151.597 | 251.502 | 257.356 |
| runtime 到控制器 admission | 0 | 0 | 0 |
| admission 到引擎 dispatch | 173.306 | 289.730 | 276.989 |
| 引擎入口到原生队列 | 104.004 | 142.230 | 136.048 |
| 原生引擎排队 | 4.166 | 8.055 | 8.293 |
| 原生 prefill | 193.393 | 204.624 | 202.315 |
| 原生 decode | 3102.616 | 3594.094 | 3593.118 |
| worker 完成通知 | 92.415 | 120.582 | 115.044 |
| worker 到控制器完成 | 4.157 | 92.033 | 93.658 |
| E2E | 4202.886 | 5204.357 | 5105.544 |

1. D472 相对 ElasticLocality 的 E2E 均值差为 902.659 ms，其中 decode
   阶段差 490.502 ms。约 54.34% 是差值的算术分解，不是某机制的因果贡献。
2. 全局许可申请到恢复反而少 46.502 ms。不能沿用 D466 的局部队列现象，
   把新运行的主要剩余差距全部写成“等 GPU”或据此盲改许可/扩容阈值。
3. D472 相对 D465 的 decode 仅少 0.976 ms，而全局许可等待少 88.474 ms。
   CPU 前端提前准备取得资源改善，但并没有消除本次观测的解码差距。
4. admission→dispatch、runtime 等待和完成通知仍有差距。它们包含多个
   控制/执行阶段，不能把全部时间直接算作 CPU 计算或 adapter 复制。

## 2. 同副本窄区间复核

复用 D460 的原生区间算法，仅新增 D472 的分析。每个原生 owner 与已释放的
物理 GPU journal、PID birth 关联；15 个实际服务 owner 均获得唯一 TP1 卡身份。
所有 4000 请求保留，未完成 ready 的 3 个 activation 也保留。
当前共同 D394 阈值覆盖复用函数中历史 D170 的文字说明；没有改变阈值。

| decode 条件 | 请求数 | TPOT 未达标数 | 平均 TPOT ms | 平均其他原生请求占用 |
|---|---:|---:|---:|---:|
| 全部 | 4000 | 45 | 33.810 | 2.525 |
| 无其他副本 startup 重叠 | 3552 | 25 | 32.630 | 2.240 |
| 有其他副本 startup 重叠 | 448 | 20 | 43.163 | 4.777 |
| 无同副本按需加载重叠 | 2252 | 20 | 31.186 | 1.979 |
| 有同副本按需加载重叠 | 1748 | 25 | 37.190 | 3.228 |
| 无同副本 prefill 重叠 | 1239 | 10 | 28.213 | 1.287 |
| 有同副本 prefill 重叠 | 2761 | 35 | 36.321 | 3.080 |
| 无同 owner 主动加载重叠 | 3107 | 33 | 32.917 | 2.642 |
| 有同 owner 主动加载重叠 | 893 | 12 | 36.914 | 2.116 |

“其他原生请求占用”是同副本请求区间的时间加权数量，不是 kernel batch、
GPU utilization 或独占执行时间。慢请求本身更易与别的事件重叠，各组长度、
并发和请求成员不同；不能当作受控因果对照或从 latency 中扣去重叠 span。

33/45 TPOT 违约没有同 owner 主动加载重叠。因此“主动加载阻塞解释全部解码
问题”不成立，也不能据此删除 handoff/hierarchy/admission。35/45 与同副本
prefill 重叠，但这同样没有证明 prefill 是唯一原因。

主动操作保留计数：GPU 1623 plans、2128 attempts、1536 实际加载、524
deferred、17 缺 receipt；HOST 1013 plans、2427 attempts、967 实际加载、
1457 deferred、3 缺 receipt。缺 receipt 不补成完成，deferred 没有窄执行
区间的不推造时长。它们提示控制事务具有实际规模，不代表所有事务成本已测出。

## 3. 历史与原生实现核查，下一项边界

- D329 已证明 epoch/model identity 不足以缓存可变 storage/pinning；D352
  已实现同一次 confirmed GPU-hit 事务的 postcheck 重用；D355 已把完整
  原子控制事务穿插到 decode turn；D457 已优化文件视图。不能重复包装旧改动。
- D462 的逐指针 Python CUDA 查询没有获得有说服力的改善，继续保留拒绝
  结论；不重跑该微测、不直接取消 pinning、完成屏障或物理预算检查。
- D463 的单项性能假设未被 D465 支持；D472 是带 CPU 准备的新版本，不能
  倒推 D463 单独有效，也不静默回滚或继续机械扩大相同替换。
- 当前复核的 [vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  将 native LoRA LRU 加载放在单线程 core loop；load-inplace、加载后驱逐和
  registration 都可能改变实际对象。异步调用形式不等于 CPU 工作离开 decode。
- [PyTorch 2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)
  与 [CUDAHooks.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/cuda/detail/CUDAHooks.cpp)
  显示 pinning 查询读取底层 storage 指针及 CUDA 属性，不是一个可永久信任的
  Python 标签。这支持检查重复观测的执行位置，不支持省略物理事实核验。

**下一项仅聚焦：完整 native 控制事务与 decode 推进的争用。** 先核对真实
consumer 对 source identity、全局 allocation/alias graph、allocator、staging
各字段的依赖，以及 mutation/引用/取消的所有者边界，再决定能否缩短事务。
只在严格保留消费语义、实时容量和失效检测的情况下提出一个可证伪改动。
GPU-hit “不新增权重”本身不足以授权省去已有 HOST 预算/allocator 检查。
不根据本表改并发、token budget、SLO、层级公式或 baseline，不重做已完成的
许可容量分类、CPU import 实验或同配置性能回放。

本项没有证明具体 CPU 检查占用多少本次 decode 时间；D456 的旧采样仅是
历史定位依据，不冒称 D472 的 CPU 时间比例。性能假设仍需最小正确性验证
及新版本一次完整回放；未实现或未测试的方案不能称已优化。

## 4. 存储、安全与交付

现有压缩工具增加显式 `--allow-closed-failure`，只允许逐一指定、实际服务和
GPU context 已释放的失败记录。默认不变；未知 pass/释放状态继续拒绝。
6 项测试通过，覆盖默认拒绝、释放约束、真实 gzip 往返与 launch 字节不变。
D93 3B attempt2 和 D311 7B 的 main_outcome 已各经两次解压 SHA 校验，
共回收 986513408 B；失败仍是失败，不进入主排名，不删除唯一原始证据。
压缩后的路径与原始 SHA 见 storage/20261012_d473_closed_failures_lossless.json。
释放后约 150.50 GiB；每次重型任务前仍须重新核验原 150/100 GiB 门槛。

110 项既有区间/计时测试通过。最初 systemd scope 拒绝 CPUAffinity 属性，
分析未执行；保留该 launcher 记录，使用 AllowedCPUs 加 taskset 后正常执行，
没有降低 CPU/内存限制。三个分析/存储域均为 3/4 GiB、swap0、CPU2,3,26,27，
CUDA hidden；峰值 406626304 / 1688469504 / 831057920 B，high/max/OOM/swap0。
均按捕获的 InvocationID、inode 和实际递归空进程清理；147 项旧结果零变化。

原始衍生数据：`results/ieee_tc/main_comparison/d473_20261012/`。
curated：`paper_results/ieee_tc/main_comparison/d473_residual/`。
未生成图，没有运行 GPU 或远端服务，没有修改无关 Kubernetes 集群。
7B 优化、W1/W2/G2、HydraServe、3B、消融、机制与敏感性等完整范围仍开放。
