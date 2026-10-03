# D202 — 物理 HOST 观察使用完整存储图，省去无消费者的描述表

2026-10-03。状态：CPU 资格完成，普通 Full 待执行。尚无性能结论。父版本
`5fadd45fb6a8be47c7a6e39ed65d8b1f4f2ecc4d`。

## 问题与唯一假设

D201 已将 D200 的 4,204 个完整原生状态查询与同进程、同主线程的采样关联。
159 个完整落在查询等待区间内的独立采样中，48 个位于 HOST inventory 的
deepcopy，59 个位于每个 view 的物理 pointer 检查。它们不是 CPU 百分比，
不能量化这些操作贡献了多少请求时延。

D144 已删除同一次观察的重复 HOST 遍历，但保留两份返回值的独立性；D174
只对 routing 省去描述表；D194 已让初始化后的 planner 使用该 routing 图。
物理 source_snapshot、HOST 预算检查和 replacement 仍构造描述表。

可证伪假设：物理消费者也不读取 `host_tensor_views`，省去其构造及深拷贝
能够减少原生核心同步 CPU 工作，保持实际决策输入不变。端到端效果须由普通
4,000 请求 Full 验证；不能由表示字节减少或组件计时推断 TTFT 收益。

## 官方实现与消费者审计

- [vLLM 0.30 UniProcExecutor](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/executor/uniproc_executor.py)：
  collective_rpc 在返回 Future 前同步执行 run_method；async 客户端不等于
  原生 owner 工作已经离开主线程。
- [vLLM 0.30 model manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/model_manager.py)：
  list_adapters 返回注册映射副本。本项目逐张量描述表不是该原生接口所必需。
- [vLLM CPU 优化说明](https://vllm.ai/blog/2024-09-05-perf-update)：
  对象构造和同步 CPU 工作可限制推理推进。本次不搬用该文的性能数字。

`NativeSourceSnapshot._footprints`、`native_gpu_fallback_costs`、
`native_host_replacement_costs` 使用 allocations、adapter edges、真实容量、
dtype 和 GPU pool；allocator/workspace 使用实际 HOST 分配图及原生计数器。
源码搜索与这些直接消费者逐项检查均未发现 HOST view 描述的运行时消费者。
GPU `pool_tensor_views` 仍是必需输入，不删除。

## 候选边界

仅五个原生物理观察调用点选择既有 `include_tensor_views=False`。每次仍检查
当前注册/暂存对象、CPU dense A/B、pointer、capacity、dtype、每 view pinning、
alias、共享/独占字节。暂存非空仍重新枚举联合图；空暂存仍 deep copy 图，
返回对象互不影响。原生 allocator 仍实时读取；未知不填零。

helper 默认完整，孤立 `ieee_worker_observation` 的完整描述与内容资格接口
不改。IEEE 九式、owner/epoch、planner、admission、实际复制、CUDA fence、
配置、并发、超时、生成合同和阈值全部不改；不添加跨调用缓存或 fallback。

## 验证与后续

| 检查 | 结果 |
|---|---|
| 实际 extension/CPU inventory 路径及 HOST 预算、暂存、owner 定向检查 | 79 项通过，0.047 s |
| 完整请求、owner、规划、取消、诊断与 smoke 回归 | 1,122 项通过，156.731 s |
| 精确父版本 extension 对照 | 7 类微型 CPU fixture 的其他字段/错误全部相同 |
| D169 既有真实物理快照 | 4 份来源对象及原生 allocator 计量逐项相同 |
| 普通 4,000 请求 Full | 尚未执行，不能宣布服务性能提升 |

两组测试有重叠，不相加作为独立证据。组件对照使用父 commit 的原始 extension
函数，在同一 tiny CPU tensor、相同 owner/allocator 测试替身上执行，不只是
比较 helper 的选项。共享、partial、packed、空集、暂存共享、额外 tensor、
pinning 冲突均保留。正常非空观察仍有 6 次实时 pinning 查询；暂存共享仍有
14 次，只有 stride/element_size/offset 描述查询变为零。

| 保留快照 | 注册数 | 每份 HOST 表省去的 view 数 | 完整 JSON 字节（前→后） | HOST 深拷贝 ms（前→后） |
|---|---:|---:|---:|---:|
| 0 | 8 | 2,048 | 1,388,807→550,023 | 26.140382→9.928758 |
| 1 | 12 | 3,072 | 2,025,691→766,427 | 75.266975→14.411368 |
| 2 | 20 | 5,120 | 3,301,339→1,199,835 | 96.688978→60.121510 |
| 3 | 24 | 6,144 | 3,936,172→1,415,212 | 111.970420→30.927552 |

JSON 大小包括 registered 与 staging 两份图；不是 msgpack 线上字节。
深拷贝仅测空暂存分支所复制的 HOST 字典，不包括观测原生 pointer/pinning、
RPC、GPU 或完整请求。三块交替顺序、每块十次，表中取块均值的中位数；全部
24 行原始计时保留，不将这些循环当三个独立 workload，亦不出运行级 CI。
非单调耗时和波动保留，不能据此推算各请求实际节省时间。

资格环境为既有 CPU 测试环境 Torch 2.8.0+cu128，CUDA_VISIBLE_DEVICES 为空；
没有将它冒充原生 Torch 2.13/vLLM 0.30 CUDA 性能。D169 来源 SHA
`ead0c5c81acc52c502874e9d45cfba111665f875220e23467c2e716a0d770602`；没有
重新解析大原始回放、新建权重、trace 或远端缓存。

失败保留：首轮 79 项中 2 failure/1 error，分别是新增断言漏计 configure
阶段的一次真实检查，以及 callback 样例缺少合法同次 length snapshot；只
修正测试，生产候选不改。首个组件脚本的父函数拥有独立 globals，未接入
测试替身的 GPU pool observer，因此在第一个 fixture 报错，未生成组件数据；
第二个脚本显式接入相同替身后通过。两轮源码/日志全部保留，没有放宽检查。

全部资格任务为 3/4 GiB、swap0、CPU2,3,26,27；终态 high/max/OOM/swap 为零，
资源域退出核对。按照 analyze-results/academic-plotting 和计划 §11，采用
精确资格/组件表，不画暗示系统端到端收益的性能图。

先形成精确资格表并备份，再运行同 D195 配置的普通 Full；没有第二个候选。
当前 7B 数值正确性、128 个输出 hash 变化、共同 warm/Resident、新旧 Prime
G1/G2 均未完成；随后才推进 3B，外部基线继续暂停。全部后续矩阵保留。

Plan SHA `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`；
MetricV1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`。
