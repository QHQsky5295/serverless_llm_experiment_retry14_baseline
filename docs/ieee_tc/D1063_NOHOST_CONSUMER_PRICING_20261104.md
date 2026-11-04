# D1063：NoHOST 来源语义与替换计价实现（尚未启用回放）

2026-11-04。接续 D1062 唯一 LastKnown 回放，不重跑 Full，不改旧结果。
本页与 `D1063_NOHOST_IMPLEMENTATION_BOUNDARY_20261104.md` 配套。它记录
NoHOST 的分步实现，不把 CPU 单元测试当作性能实验或原生执行资格。

## 主线与本次范围

3B A1 已完成 10/11，剩余 NoHOST。检验的问题仍是：在保留必要 GPU backing
和本次请求/计划 staging、GPU/NVMe 预算不变时，独立 HOST 复用能否改善服务
质量和物理 GPU 时间。Full 在现有几个变体前不占优的观测保留，不能据此修改
共同指标或删去反例。采用冻结 V1 与单轮 V2；无新增 CI、重复 seed 或阈值。

用户确认现有 k3s 不属于本项目。D807/D808/D809 已完成的 HydraServe 3B
隔离 rootless Podman 路径继续复用，不修改其他项目集群，不重复部署。
其机制覆盖仍按已有结果的范围报告，不能把单 stage 适配称为跨节点 pipeline
及 consolidation 的完整性能证据。

## 历史与原生依据

- D1049 已否决“HOST 文件预算置零即完成 NoHOST”和“native HOST 预算置零”：
  前者保留隐形 CPU 缓存，后者破坏 GPU 加载所需暂存。
- D41 区分 native 张量、文件层和 allocator 留存；删除对象不等于物理字节归还。
- D1063 边界审查确认：在 `_refresh` 或 native removal 回调中清理会落入
  半完成事务；回收必须放在最外层操作完成、最后引用释放和 plan close。
- 本轮再次核查官方 vLLM v0.30.0 的
  [worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  和 [model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)：
  CPU 加载/注册与 GPU 激活相互依赖，不能取消 GPU backing 后仍称相同后端。
  本机另有已资格化的补丁，不声称与上游源码逐字节一致。
- `ablation-planner` 既有 reviewer 做了同系列 provisional 只读复核：完整物理
  账本、复用资格与真实生命周期须分开；未冒称外部独立评审。

## 已落实的消费与计价语义

1. 既有 registry schema 增加严格布尔 `independent_host_reuse` 解码，旧字段
   缺省保持 Full；不是可在线自动改变的策略参数。NoHOST 值为 false。
2. 完整 native 身份、registered IDs、CPU/GPU footprint、未知身份及确认状态
   不删除。不可复用 CPU backing/staging 仍计量，但不能成为独立 HOST 命中。
   GPU-ready 仍可用；identity、routing 和 request-scoped 投影保留该资格。
3. router 与 planner 校验文件/native owner 的政策一致，禁止同一个 native
   owner 的观察历史中途切换。HOST 文件即使存在也不作为该模式的服务来源。
   原有内容、路径、clock、epoch 校验保持，不因不可复用而跳过身份错误。
4. NoHOST planner 不生成纯 HOST 最终目标；GPU/NVMe 目标和三层真实剩余
   预算原样保留，不把省下的 HOST 容量转给另一层。GPU backing 仍在物理视图。
5. GPU victim 损失按原冻结需求、成本和实际 NVMe/remote 后继来源计算，不再
   用 GPU backing 的 HOST 准备时延。执行计价需要完整、仍持有的文件引用或
   冻结不可变 remote 身份；即使零需求，也不能用缺失来源证据跳过检查。
6. CPU-only 暂存既不是独立服务来源，退役不再声称损失一个 HOST 命中机会。
   GPU-active 对象的损失为同一 `h × D_successor`；不重复加 HOST 损失。
7. 文件替换若声称零增量损失，NoHOST 必须保留真正 GPU 来源，不能仅保留
   CPU backing。目标载荷明确写 GPU fallback，后续执行必须持有 GPU 引用。
8. execution projection 与 hash-bound GPU objective 保留模式。旧 Full wire
   不新增显式 true 字段；Full 的原 HOST fallback 和默认行为保持。

## 不能据此启动 NoHOST

当前尚未接入 NoHOST 的公开运行配置，物理 owner 不会发布 false。
本页实现只完成消费/规划/计价层，不能通过手工给快照加 false 冒充真实消融。
下一块仍须完成：

- 在已有 owner 初始化/预算安装边界冻结并读回策略；文件 owner 禁止独立
  HOST 最终驻留，native owner 保留 GPU backing 与精确绑定 staging。
- 最外层 native transaction、release、plan close、已 join 的 construction
  settlement 后回收无 GPU、无引用/外部 pin、无活跃 plan/build 的 CPU-only
  对象；不在只读 snapshot/refresh 或 removal callback 递归回收。
- native demand 提交不能偷偷复用其他操作的 CPU-only/staging；仅原操作的
  精确 lease/plan 可完成 staging→GPU。保持取消 join 和重复 release 语义。
- NoHOST 的所有 GPU 替换收集实际 NVMe/remote fallback leases，不再只在
  `staged is not None` 时收集；outer finalize 在一致性检查之后回收 victim。
- 文件替换的零损失来源使用真正 GPU lease，并维持到异步 I/O 完成/取消；
  不能把已有 CPU-only file_fallback lease 改标签就算完成。
- 复用现有 pin、staging、取消、替换 fixture 做针对性验证；完成后才注册
  唯一 3B W1 NoHOST 执行键。无额外性能 smoke、整池重建或同键反复试胜负。

## 测试记录与保护

原始限定测试证据目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1063_20261104/`。
首次测试发现 planner 的新模块级导入造成 spawn worker 循环导入，失败保留；
共享解码移入原有 registry schema 依赖层修正。第二次启动在测试前因不符合
既有 UUID scope 命名而拒绝，未降低所有权检查；后续使用正确 scope 身份。
这些不是 GPU 实验失败，也不构成性能结果。最终测试结果另附回执，未完成前
不写通过。所有测试 CUDA 不可见、CPU 2/3/26/27、内存 high/max 3/4 GiB、
swap 0；脚本沿用 D1062 的独占输出和源 SHA 前后核验。

不修改用户 dirty 的 `configs/generated/lora_manifest_1000.json` 或
`docs/ieee_tc/EXECUTION_STATUS.md`，本页承担当前增量交接；不覆盖旧投稿目录。
未生成图，没有远端服务/共享 k3s 修改或新性能回放。

剩余总范围保留：NoHOST 3B；7B A1 和两个累积节点；受控 A2–A5、M2、
S1–S13；CE/价格/Pareto；B1/B2/C1/C2/C3、公式推导及两轮审稿证据；IEEE
图表、最终同版本 canonical 来源统一和实际 FaaSLoRA V2 的非强制集成。

## 13:38 实施检查结果

- `consumer_tests6`：130项全部通过，源文件测试前后 SHA 相同；包括新增
  NoHOST 语义及 Full 路由、request footprint、原生替换计价和全来源规划回归。
- 内存峰值1,193,066,496 bytes，high/max/OOM/kill及swap均0；CUDA不可见。
  首轮真实 spawn 暴露的循环导入已修复。测试启动曾因生产 runner 自动导出
  PYTHONPATH 导致 stdlib tracker 也加载 vLLM；只调整测试启动路径，在模块
  加载后清除该环境变量，不修改生产 runner、不增加内存限额。
- reviewer指出 `find()` 同时用于物理来源缺失证明，不能按复用资格返回None。
  已恢复其物理查询语义，并新增“不可复用但仍存在，不能报 absent”回归；
  服务过滤仍在独立分类路径。此问题与修正均保留，未改测试判据回避它。
- 第五次限定测试仅有一项新fixture缺 `source_view` 的错误；按真实 wrapper
  返回形状修正。所有早期失败/不完整记录保留在同目录，不冒充全部一次通过。
- 本次完成的是**NoHOST 消费、规划与计价块**。3B性能消融仍为10/11，没有
  新GPU回放、没有 NoHOST 完整执行资格，没有新的系统性能领先结论。

备份交付目录为 `paper_results/ieee_tc/ablation/d1063_nohost_consumer_pricing/qualification/`；
脚本在提交前核验147项旧投稿保护文件和两项用户dirty SHA，单独推送 origin
V2及FaaSLoRA备份分支，不强推仍分叉的实际FaaSLoRA V2。完成回执保存在
上述原始证据目录的 `backup.json`；只有该回执成功才算已推送。
