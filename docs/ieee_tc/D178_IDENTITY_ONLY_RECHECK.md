# D178 — 将所选低层副本复核与全量物理盘点分开

2026-10-03。Prime 7B 单一开发候选，父版本
`8ae851c418a0a2d2090e73d56e327b95dca91c14`。状态：正确性与组件资格完成；未做性能回放。

## 问题与当前证据

D177 普通 W0 已封存，不重跑其分析。平均/P95 TTFT 为 3.034756782/
6.955747798 秒，推理前两段共 2.660959498 秒，占平均 TTFT 的 87.68%。
这些阶段不能全部归因于 HOST 盘点。D162 的四个 native core 中 HOST
inventory 分别出现在 128/552、116/531、89/529、98/527 个采样栈中；
这是历史线索，不是当前 CPU 时间占比。D176 已去掉持有 HOST 引用后
重复的全图查询；D174 已去掉路由不消费的 tensor 描述表。本轮不重复它们。

新的调用链证据：`_ieee_protect_selected_source` 对非 native 的 HOST 文件、
NVMe 和 Remote 选择，在获取前重新检查所选引擎是否刚出现更快 native
副本。该分支只使用 owner/epoch/clock、adapter identity、tier/source ID，
不使用 footprint，却调用完整 `ieee_routing_sources`，从而同步构建 HOST
和 GPU pool 存储图。首次路由/成本分类、之后实际加载、完整 planner 和
admission 则确实需要各自的全量信息，不能一并删掉。

## 原始实现与假设

- [vLLM 0.30 worker manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)
  通过注册映射、原生 LRU 和激活路径处理已加载状态，不要求每次只查存在性
  都重建所有 tensor 的物理明细。这是具体接口差异，不证明其他系统没有 CPU 瓶颈。
- [vLLM 0.30 model manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/model_manager.py)
  的 packing/scaling/pinning 会改变对象表示，因此本轮拒绝按 adapter ID
  永久缓存 footprint；不把最近已知状态冒充新确认。
- [官方 CPU 性能分析](https://vllm.ai/blog/2024-09-05-perf-update)
  说明同步 CPU 工作可限制推理服务；这里只借鉴按实际消费者拆分工作，
  不借用其加速数字。[dLoRA](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)
  的请求—adapter 协同为相关背景，不据此推断它的具体观测开销。

可证伪假设：在仍然执行同一次实时 owner 状态验证的前提下，对上述一次
存在性复核不构建无消费者的物理图，可减少同步工作，并可能改善完整回放
的首 token 等待。组件查询次数不是 TTFT 收益，仍需一次普通 Full 验证。

## 唯一候选与约束

新增明确的 `source_identity_snapshot`/`ieee_source_identities` 只读路径。
仍调用原 `owner.source_snapshot()`，保留原线程、poison、CPU/GPU cache、
slot、pin、source object/incarnation、未知和未确认 ID 校验。原生加载、
owner 和预算代码不变。它不返回 footprint，不发零容量或空图冒称全量盘点。

frontend 仍校验完整身份覆盖和 clock/epoch；返回 readiness-only 投影。
仅低层所选副本复核使用它。首次路由、成本估计、selected-copy 原子引用、
加载前全图、冲突重查和所有物理预算路径保持原完整观测。

维护独立的已接收身份版本记录：相同 epoch 的 copy/slot/身份变化仍拒绝；
旧 epoch/旧 capture 的延迟全图不能越过已经收到的新身份版本。身份复核
不覆盖已测 footprint；它也不能直接提供 service class。此记录是收到的
版本下界，不是可被下一请求直接拿来路由的缓存或 TTL。

九个 IEEE 公式、profile、cap4、到达、输出、60 秒准备、SLO、预算和远端
已发布交付缓存均不改。无新 GPU/remote 运行、无新数据/工件或更宽超时。

## 验证与后续

使用 `run-experiment` 的既有入口和 D176 的 3/4 GiB、swap0、CPU2,3,26,27
限定执行器。新增测试覆盖身份/全量版本交错、同 epoch 冲突、坏 owner/
clock、无 footprint 不能估计成本、低层变成更快副本后的重新选择、实际
本地 RPC 传递和原 worker owner 异常。复用 D176 精确父方法组件对照。

按 `academic-plotting` 与计划 §11，资格阶段使用精确状态表，不制造 GPU
性能图。资格后备份，再以相同 D177 配置与 D157 profile 做一次普通
4000 请求 W0。当前磁盘低于 150 GiB 新重型任务门槛；先审计可重建空间，
不放宽门槛、删除原始证据或在推理中清理。

数值正确性、共同 reference/Resident、旧 Prime 新指标对照及 G1/G2 尚未
达标；3B 和外部基线继续暂停。一次查询简化不能关闭这些验收项。

### 实际资格结果

| 检查 | 结果 |
|---|---|
| 定向检查，含新增 11 项 | 66 项通过，1.420 s |
| 完整受影响回归 | 958 项通过，103.589 s；与定向检查有重叠，不相加 |
| 实际本地 TCP worker/proxy 身份查询及取消 | 通过；只读取消不虚构未释放 native 工作 |
| 五种来源、精确 Git 父方法对照 | 10 条组件记录；层级、加载事务、HOST 引用/释放相同 |
| 新普通 4000 请求 W0 | 尚未执行，不能据组件结果宣称更快或已达到 G1/G2 |

| 来源 | 全图查询，父版 → 候选 | 身份查询，父版 → 候选 | 候选实时查询总数 | 残留引用 |
|---|---:|---:|---:|---:|
| gpu | 1 → 1 | 0 → 0 | 1 | 0 |
| host | 1 → 1 | 0 → 0 | 1 | 0 |
| host_file | 3 → 2 | 0 → 1 | 3 | 0 |
| nvme | 3 → 2 | 0 → 1 | 3 | 0 |
| remote | 3 → 2 | 0 → 1 | 3 | 0 |

host 指原生 CPU LoRA 对象，host_file 指受管 HOST 文件，两者不混为一类。
每个样例仍发生 1 次 demand-load/reference 事务；所有 GPU/HOST/request
引用最终释放。总实时观察次数没有减少，只改变其中一次不消费存储信息的
观察内容。不能将全图次数少 1 次解释成 TTFT 少三分之一，不给 n=1 CI。

第一轮完整回归 958 项中 1 项因旧模拟副本未实现新接口而报 AttributeError；
对应的是前置路由测试 fixture，不是实际 worker 丢失接口。保留失败源文件、
日志和 SHA；只为该 fixture 接入已测试的真实新方法并声明只读操作，未改
生产候选、时限、断言或资源条件。第二轮全组通过。定向检查后另将已有
取消用例扩展到新只读方法；它包含在最终完整回归中，初版源码也保留。

测试与组件任务均在已声明 3/4 GiB、swap0 的独立 CPU 资源域运行，最终
memory.events/swap 为零，自动移除已核实。没有 GPU 推理或远端服务操作。
原始证据：`results/ieee_tc/p2_backend_qualification/d178_20261003/`；
curated JSON/CSV 与可复查源码包使用独立 `20261003_d178_*` 名称，不覆盖历史。

Plan SHA `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`；
MetricV1 SHA `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`。
