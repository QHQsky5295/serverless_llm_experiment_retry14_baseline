# D111：同一次状态观测的解析共享

2026-09-28。Prime IEEE Full 开发诊断；CPU 证据，不是论文性能结果。

## 问题、历史与设计依据

D96 已合并并发只读 RPC，却把收到的原始字典返回给每个等待请求，
使每个请求重复运行 `NativeSourceSnapshot.from_native/_footprints`。
D102 仅测到旧快照的纯解析成本，故当时未接受优化。D110 随后完成的
完整剖析中，`_footprints` 占 559,878 个父进程持 GIL 样本中的
130,957 个（23.39%，排除 save_results 祖先后的 nearest-project 分类）。
这不是 wall time 比例，也不能说明全部超时由它造成。

本轮可证伪假设：同一 in-flight collection 的多个等待者重复验证同一份
不可变输入；将验证放入共享 collection 能消除重复解析，同时保留请求级
实时可行性与后续 selected-source 再验证。

2026-09-28 核查的原始依据：

- [Python 3.12 asyncio 文档](https://docs.python.org/3.12/library/asyncio-dev.html)
  说明同步 CPU 工作会延迟同一事件循环中的其他任务。
- [singleflight 官方接口](https://pkg.go.dev/golang.org/x/sync/singleflight)
  支持合并同 key 的正在执行工作；这是通用实现手段，不是论文算法贡献。
- [vLLM 0.30.0 LoRA worker](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  在单线程 core 中执行 LoRA 操作，但不提供整个多 RPC 路径的原子性。
  因而没有移除当前副本状态、原生引用与准入再检查。

## 同输入因果探针

复用 D102 按 D89 来源 SHA 核验的 D88 `sources_after`，一个真实 owner、
8 个注册 adapter、1,792 个 HOST allocation；不复制工件或生成负载。
使用已有 D96 测试 fixture 调用实际 `_ieee_request_snapshot`，仅原生 RPC、
文件 owner 和利用率观测由测试边界提供。历史 clock 在测试边界显式注入，
原 owner/epoch/captured time/footprint 均未改写；这不是实时鲜度测量。
不把该图当成 D110 的 32-adapter cache 或实际并发分布。

| 等待者 | 实现 | 每组 RPC | 每组解析 | 整组处理三次观测（ms） |
|---:|---|---:|---:|---|
| 1 | 修改前 | 1 | 1 | 3.135 / 3.229 / 2.915 |
| 1 | 候选 | 1 | 1 | 3.250 / 2.986 / 2.941 |
| 32 | 修改前 | 1 | 32 | 293.484 / 87.709 / 87.742 |
| 32 | 候选 | 1 | 1 | 6.185 / 6.455 / 6.020 |

32 等待者的重复解析减少 96.875%。整组诊断时间的对应下降为约
97.89% / 92.64% / 93.14%；包含 Python fixture/计数插桩与运行噪声，
不是 GPU 服务加速比、独立 workload 重复或正式 CI。修改前首个 32 请求
观测明显较慢，全部保留，不凭此选择最快或最慢值作为主张。
单等待者未发现可据三次微测声称的优势。

前后每组路由输入结构化 SHA 完全一致；实时 admitted count 仍为 0–31，
文件快照仍逐请求获取。每组结束后的下一请求重新 RPC，不复用完成结果。
原始输入 SHA 未变。首次探针仅在保存摘要时因 frozenset 序列化失败；
失败源码/日志/退出回执保留，修正探针序列化后才取得上表。

## 实现边界与正确性

仅修改现有 runner 的共享 collection：收齐原生响应，核验成员身份、
不同物理 GPU 身份及原生 owner/clock/epoch/footprint，返回不可变快照
与 GPU identity。新增 `parse_invocations` 记录解析调用入口数量。

- 每个等待者在最后一次 await 后重新检查成员和整组状态鲜度，然后整组发布。
- 文件状态、queue/active/pending、可行集、利用率及成本类逐请求重算。
- 同 epoch 异内容、旧观测、owner 变化仍拒绝；解析错误传播，不自动重试。
- 一个等待者取消不取消其他等待者；最后退出者仍收尾本组读取。
- 无 TTL、已完成结果缓存、人工等待、期限调整、未来信息或错误吞噬。
- 原始响应不再交给等待者；接收时钟验证提前到共享 collection，仍使用
  本地接收时刻，不重标 captured time，不放宽时钟判断。
- 九个公式、原生准备/驱逐/引用、配置、profile、工件和生成合同均不变。

275 项相关回归通过（6.047 秒），涵盖请求生命周期、选路、准备成本、
外置回放和物理 GPU 生命周期。新增/强化检查包括每副本只解析一次、
共享解析失败不部分发布、不自动重试、等待者间出现新 epoch 时旧视图
拒绝，以及成员变化先于解析拒绝。并发取消等原检查继续通过。
受限 CPU 资源域为 3/4 GiB、swap=0、CPU 2,3,26,27；无 GPU/远端启动。

## 判定与下一项

CPU 证据支持将候选进入下一次 canonical 3B Full 4000 W0 开发回放，
尚不构成 Full 合格、SLO 达标或领先基线的结论。采用同 D88/D89 初始化、
原 trace/subset、固定生成与期限、真实已发布交付缓存；不叠加另一优化。
下一回放不加详细 py-spy，故不得将它与 D110 的差直接全部归因于解析共享。
运行结束后先清理、校验、表格、阶段解释，再决定 7B 或下一瓶颈。

原始证据：`results/ieee_tc/p2_backend_qualification/d111_20260928/`。
摘要：`paper_results/ieee_tc/p2_backend/20260928_d111_shared_snapshot_parsing.json`。
旧投稿与 D110 结果不覆盖。warm/Resident、baseline、M1/M2、A1–A5、
S1–S13、numerical adapter 资格仍待完成；一次性交付缓存已经完成，不重建。
