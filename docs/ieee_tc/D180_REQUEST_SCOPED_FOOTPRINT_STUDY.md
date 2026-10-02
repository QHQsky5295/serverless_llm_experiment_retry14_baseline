# D180：请求范围 footprint 的语义与工作量研究

状态：CPU 依赖研究通过；尚未改变生产路径、未运行新的 GPU 性能回放。
主线仍是 7B，实际正确性、共同参考、旧/新 Prime 的 G1/G2 验收均未关闭。

## 问题、历史与假设

D179 完整回放的平均 TTFT 为 3.117376972 s，其中进入原生引擎前
2.740055572 s，约 87.90%。这是阶段定位，不证明整个阶段都由某次扫描造成。
D162 独立 CPU 采样曾在四个 GPU worker 分别观察到 HOST inventory
128/552、116/531、89/529、98/527 个业务栈；样本数不是 CPU 时间百分比。
D174 已删除路由不消费的 tensor-view 描述，D178 已缩小被选副本的
身份复核范围；D179 未证明 D178 带来整体净收益，不重跑该假设寻找好结果。

当前初始路由仍读取每个副本中全部注册 adapter 的每个 HOST 张量。
但 `confirmed_source_class` 只以当前请求的 adapter 生成 Eq. (2) 的
footprint/representation 类别。全局 identity、slot、owner、epoch 仍需要新鲜。

可证伪假设：在保留全副本、新鲜完整身份视图的前提下，对当前请求只测量
其 adapter 的原生 storage capacity，可以保留 Eq. (2) 输入，同时减少
GPU 串行执行循环中的无关 HOST 张量检查。尚不假设端到端延迟会改善。

## 原始资料核查

- [vLLM 0.30.0 worker_manager.py](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
  `LRUCacheWorkerLoRAManager._apply_adapters` 遍历本批请求的 LoRA；
  `add_adapter` 依当前 cache membership 加载或复用。它不是 Prime 的
  确认状态传播机制，不能照搬其检查范围作为本系统正确性的证明。
- [vLLM CPU 路径优化说明](https://vllm.ai/blog/2024-09-05-perf-update)：
  区分前端与引擎，减少 CPU 工作阻塞 GPU。只借鉴问题分解，不引用其
  其他硬件/版本的加速比作为本实验收益。

源码审计另发现旧 runtime hints 仍由 IEEE scale-down 的
`slot.load_queue_depth` 检查消费。因此不直接删除旧 hint 更新。

## 本次最小研究

复用 D174 的 CPU fixture/已保存观测方法、现有 HOST inventory helper 和
D169 的四个原生 7B snapshot，不生成权重、请求或新池，不访问远端。

1. 小型真实 CPU tensor：shared storage、partial view、packed modules、
   missing target、target mutation、unsupported target 与 pinning 冲突。
2. 对四个已有 snapshot 的每个注册 adapter，比较全图与 target-induced
   子图得到的 footprint、representation，以及 Eq. (2) service class。
3. 保留完整 identity view；明确子图容量既不是全局 HOST union，也不是
   eviction reclaimable bytes，不输入 planner/admission/物理预算。
4. 记录每次观察需检查的 tensor view 数及不同请求共享读的潜在代价。
   不测量/宣称新的 RPC 字节或时延；CPU fixture 不冒称 GPU 性能。

生产实现尚未更改。投影仅用于检验数据依赖，不能绕过现有全覆盖 parser。
若研究支持该方向，后续实现必须显式定义 scoped schema，并测试：

- 全局 fresh identity + 当前请求 footprint 在同一个原生观察中完成；
- owner/epoch/slot/source incarnation 和独立 identity frontier 保持严格；
- 同 epoch 的不同 footprint 范围不能被误判为状态改变，也不能掩盖重叠项冲突；
- in-flight sharing 只能共享覆盖当前 target 的观测；不得沿用别的请求缺失的
  footprint，也不得缓存已完成观测、引入 TTL 或使用未来热点；
- planner、admission、physical capacity、selected-copy reference 继续使用
  各自完整/权威的资源检查；不将子图的 exclusive bytes 当可释放内存；
- 全请求回放要计入可能新增的 RPC 次数，不根据扫描次数推算实际 TTFT。

## 资源与证据

沿用 D174/D178 bounded CPU runner：MemoryHigh 3 GiB、MemoryMax 4 GiB、
swap 0、CPU 2,3,26,27，CUDA 不可见；先确认没有 GPU 作业。
输入小于 64 MiB；输出唯一 `d180_20261003/probe1.*`，禁止覆盖。
完成后先释放并核查 scope，给出结果表与结论，再决定实现；不跳到 3B/baselines。

## 结果与解释

2026-10-03 05:17–05:18 +08，probe1 正常退出。复用输入
`d169_20261002/7b_probe1.json`（38,068,749 bytes），SHA
`ead0c5c81acc52c502874e9d45cfba111665f875220e23467c2e716a0d770602`。

| 已有快照 | 注册 adapter 数 | 全量 HOST tensor 检查 | 单目标检查 | 工作项减少 |
|---|---:|---:|---:|---:|
| 0 | 8 | 2,048 | 256 | 87.500% |
| 1 | 12 | 3,072 | 256 | 91.667% |
| 2 | 20 | 5,120 | 256 | 95.000% |
| 3 | 24 | 6,144 | 256 | 95.833% |

四个快照共 64 个 adapter，逐项真实 footprint/representation 相等；
每项取三个确定输入组合，共 192 次 Eq. (2) 类别计算相等。
完整 source identity 投影保持相等。这些是依赖关系验证，不是 64 次
独立性能重复。GPU slot metadata 在此复用同一快照，不声称减少 GPU pool 检查。

五个 CPU 正例全部通过；目标不支持的 tensor 字段、不完整 A/B、
同 storage pinning 不一致三个负例均被全量/目标两条路径拒绝。
目标缺失不填虚构 footprint；目标 tensor 更换后重新测量并得到新容量。

明确保留两个反例/限制：

1. shared/partial/packed fixture 的目标可访问容量是 512 bytes，
   但全图中的 exclusive reclaimable bytes 是 **0**。隔离子图中形式上的
   exclusive=512 不能用于驱逐/预算；该区别已写入逐例表。
2. 与目标无关的损坏 tensor 会被全量检查拒绝，却不会被目标检查发现。
   因而这不是全局验证等价证明。正式实现必须保留加载、规划和 admission
   的全局/权威验证职责，不能将“本次 routing 有足够输入”写成“整个 cache 已核验”。

D179 的 4,578 次 source-read 请求共享为 2,138 个 collection；按 target
细分可能削弱共享并增加 RPC。当前日志投影不含逐 collection 的 target 成员，
不反推缺失分布，不重解析旧大文件来伪造精确收益。后续测试必须计量实际
collections/RPC、request waiting 和 TPOT，不能从上表推算加速比。

实际 scope InvocationID `204014b8ae83422cb14c3bf6b382a320`，退出码 0；
16.15 s 为整个 Python 研究进程 wall time（包含导入/校验），**不是系统性能**。
最大 RSS 1,154,824 KiB；high/max/OOM/swap 均为 0；scope 已自动移除并核查。
147 项历史保护文件、Plan/V1 SHA 均通过；生产代码未改，无远端服务操作。

## 决定与下一步

接受为一个有实测数据依赖支持的**候选实现方向**，不接受为性能优化成功。
下一步仅实现显式 request-scoped footprint 观察，保持全局 fresh identity、
按 target 覆盖关系共享在途读、独立 freshness frontier 和完整物理预算检查。
先用取消/失效/同 epoch 不同 scope/共享 storage 测试，再以普通 7B Full
完整回放判断净收益。保持现有配置、公式、阈值、远端交付和资源护栏。
不删除旧 runtime hints，不叠加其他候选，不启动 3B 或外部 baseline。
