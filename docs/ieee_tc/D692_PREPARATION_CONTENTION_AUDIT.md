# D692：准备任务发起与同副本竞争的有界核查

2026-10-18。主线仍为3B PrimeLoRA优化；没有新GPU回放、生产修改、配置选择
或图表。复用D684已关闭的1,000请求诊断日志，不将此前缀写成完整W0性能点。
最新完整候选D688仍为91.125%联合达成、15897.107082 GPU-s；其改动已由
D689撤回。共同95%/75%、九公式、V1/单轮V2以及独立数值证明未知均不变。

## 1. 问题和方法

D691证明当前控制交错确实有效，并排除了重复D514身份查询替换。下一问题是
准备任务自身为何反复进入控制路径，而不是再缩减一次物理清点的描述字段。

本次采用system-profile与analyze-results的观察—解释—边界流程：从
`mechanism_events`四个权威journal流式读取，不重复计入shutdown快照副本；
保留取消、延后、过期和缺字段。最大一次只构建一个journal记录，派生投影
约6.89MB，未复制299.74MB原outcome。分析脚本复用D691统计/校验函数及
既有资源限制、保护清单和关闭入口；没有新监测代码插入推理路径。

## 2. 实际计数

| 对象 | 总数 | 完成 | 其他终态或回执 |
|---|---:|---:|---|
| residency epochs | 1200 | 907 | 289注册时过期、4取消 |
| GPU准备计划 | 803 | 510 | 289注册时过期、4取消 |
| GPU准备尝试 | 1513 | 1030 | 482延后、1在观察中随收尾取消 |
| GPU准备有回执尝试 | 1512 | 1030 acquired | 474 stale_snapshot、7无可用HOST victim、1收益不足 |
| 成功GPU准备的实际行为 | 1030 | 1018调用native load | 12复用已确认GPU目标 |
| HOST准备有回执尝试 | 675 | 356 held | 319 stale_snapshot |

GPU计划有1623个目标，其中584个属于注册即过期的计划；它们不能作为已执行
加载计数。完成的510个GPU计划包含1021个目标和1485次尝试，其中1010次
实际加载、11次GPU复用。另4个取消计划中仍有9次已完成动作；这些工作不能删。
HOST有361条准备记录、676次attempt，其中1次只到file_held，另3条记录没有
attempt；没有回执不填零加载，也不据此认定资源遗失。D684已有完整关闭证据。

这些数分别是计划、目标、attempt和实际加载，不是不同adapter数、服务命中数
或有用准备数。尚未证明1018次加载中有多少在驱逐前服务了请求。

### 排除最初考虑的捷径

“已GPU-ready仍走准备流程”仅占12/1512个有回执GPU attempt，不是此处最主要
的重复类型。更重要的是，`register_preparation_plan`明确只保护主动替换目标，
不pin物理GPU/CPU存储，也不阻止普通请求的原生LRU替换。D104已经覆盖这一
生命周期差别，当前代码和21项已有owner回归仍符合它。

因此不把注册回执当作执行时的GPU租约，不据它提前完成目标或跳过确认。
也不删除完整source_snapshot：D514/D516/D519已有失败与回退证据。本次没有
重新命名、重跑或接受这些旧方向。

## 3. 新的组件因果证据：同副本GPU准备相互失效

现有公共movement queue按物理目标/content去重，但不同adapter即使属于同一
native owner也可同时执行。每个最终GPU action包含观察→提交→release的多次
异步往返。后端单个操作串行，并不能保证两个action的整个观察/执行序列原子。
已有`GPUExecutionProjection`测试也明确允许兄弟目标造成stale_snapshot。

本次只在独立CPU测试进程中替换queue的`_pump`，未修改生产文件：

- 同一native owner最多执行一个最终GPU action；这是由该后端串行操作边界
  导出的资格限制，不增加可调并发参数或sleep。
- HOST/NVMe和其他owner仍可并行，全局并发上限保持3；不将整个系统串行化。
- 不让等待同owner的任务先占全局slot；保持原有demand、density和稳定身份
  排序，只在当前可执行集合中派发。
- 不改source/epoch校验、planner、admission、物理容量、准备目标、加载内容、
  注册过期、事件唤醒、取消/失联处理或控制交错。

使用现有真实runner/queue/native-owner逻辑及CPU缓存fixture：

| 同一两个目标的操作 | 当前实现 | 仅进程内候选 |
|---|---:|---:|
| 完成目标 | 2 | 2 |
| 真实fixture加载 | 1 | 1 |
| 复用GPU目标 | 1 | 1 |
| source查询/prepare尝试 | 3 / 3 | 2 / 2 |
| stale_snapshot延后 | 1 | 0 |
| 残留GPU引用/准备目标 | 0 / 0 | 0 / 0 |

这是一个直接操作顺序对照，不是GPU latency或数值正确性试验。它证明一条
自身竞争路径存在，不能断言D684全部474次GPU、319次HOST过期都来自它；普通
请求、跨tier工作也会推进epoch。289个注册过期同样未被这个候选消除。

40项组件/已有回归通过，覆盖同owner互斥、不同owner与HOST并行、无head-of-line
slot占用、defer后等待真实wake、取消必须先完成实际收尾，以及来源增长、GPU
复用、坏身份和HOST执行冲突。包含前述21项测试，不能相加为61个独立检查。
完整相关回归及GPU资格尚未进行，生产性能收益未知，候选尚未接受。

## 4. 历史与原始资料依据

- D104：注册不等于预留；普通需求仍能改变准备来源集合。
- D134：计划保护实际选中来源，不自动延长已退役旧副本寿命。
- D176：按需HOST路径已有精确copy lease接续；不能将其凭证凭空套给未持有
  租约的主动准备。本候选不放宽epoch guard、不引入陈旧快照缓存。
- D355/D691：当前原子utility后的模型step交错已经工作，不能复活多utility
  排空问题；本候选只改变哪些准备action能同时占用同一owner。
- D514/D516/D519及D687–D689：局部CPU开销降低并不保证完整SLO或GPU-s改善。

本轮在线核查的官方材料：

- [vLLM 0.30.0 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
  LRU `add_adapter`由单线程core loop调用，已注册对象复用而非重复加载。
- [vLLM 0.30.0 model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)：
  active/registered两类缓存与真实GPU slot属于同一manager，activation会改变
  slot状态；它没有替本项目保证多个外部RPC组成的事务原子性。
- [vLLM官方CPU路径分析](https://vllm.ai/blog/2024-09-05-perf-update)：
  支持检验管理工作对模型迭代的干扰，但不借用其速度数字或将本次CPU测试
  解释为端到端性能提升。

## 5. 安全、产物及下一步

两项CPU任务都使用3/4GiB high/max、0swap和CPU2/3/26/27。日志分析峰值
535871488B，组件验证523026432B；high/max/OOM/OOMkill和swap均0，实际
递归空后按精确身份关闭。GPU空闲，无remote/k3s操作。147投稿保护文件与
用户manifest未变，无清理或压缩回收；下个重型任务仍检查150GiB和新增峰值。

原始：`results/ieee_tc/p2_backend_qualification/d692_20261018/`。
发布：`paper_results/ieee_tc/p2_backend/d692_preparation_contention/`。
分析新增文件为`diagnose.py`、`owner_probe.py`和沿用关闭/发布wrapper；所有
运行派生数据、测试日志、源码SHA与关闭回执一并保存。推理代码修改清单为空。

下一步允许将本次**同owner GPU action资格**作为唯一新候选进入生产实现验证：
先扩展现有queue/混合准备回归，特别检验shared-interest取消、staging前置任务
不占GPU slot、不同owner并行和普通需求冲突仍严格拒绝；再按既定最小原生
资格及一次完整3B W0检查系统效果。不得以本组件次数下降直接宣布优化成功，
不得顺带修改HOST并发、source_snapshot、profile、autoscaler、SLO或预算。
若完整运行无收益，保留结果并撤回，不反复重跑同键挑胜者。

7B/3B剩余场景与G2、3B基线/dLoRA、独立数值证据、A1–A5、S1–S13、CE/
成本、两轮审稿文档和最终图表全部保留；不再重复已有baseline基础适配。
