# D1063：NoHOST 文件 owner 政策与当前交接

2026-11-04，接续 `D1063_NOHOST_NATIVE_LIFECYCLE_20261104.md`。
这是增量实施记录，用户已修改的 `EXECUTION_STATUS.md` 不动。3B A1 仍为
**10/11 个性能点完成**；本轮没有 GPU 回放，NoHOST 尚不能启用。

## 已实现的文件层边界

`LocalSourceReferences → FileStateProcess → ProcessLocalSourceReferences`
传递构造时冻结的严格布尔 `independent_host_reuse`。Full 缺省为 True，来源
wire 不增加新字段；NoHOST 为 False，必须从空的受管 HOST 根目录开始。
已有内容不删除、不冒称未占用；根目录、HOST 总预算及 native/workspace
正预算保留，GPU/NVMe 不获得额外容量。

- NoHOST 禁止 HOST movement interest、准备目标、物化、读取引用和发布；
  verified 与 legacy publication 均覆盖，不能绕过验证分支发布 HOST。
- planning 的最终文件目标只保留 NVMe；HOST 物理几何、budget 和 inventory
  不隐藏。identity、source、preparation、replacement 视图显式带 False。
- 独立进程的 acquire_best 只可选择合法 NVMe 文件；真实 publication、内容
  SHA、引用、demand witness、取消/释放和 plan 关闭语义继续复用。
- ResidencyManager 的两条 legacy materializer 同时约束源与目标：不能把
  未验证 HOST 单文件/目录先复制到 NVMe 再使用；合法外部/NVMe 输入仍可用。
  非合作出现的 HOST 文件作为未知状态报告，不静默清除或伪装 remote miss。
- memory config 已能透传给文件 owner；实验层总开关、native 授权和 GPU/file
  fallback 事务仍未全部接入。因此没有创建/启动半实现的 NoHOST 配置。

## 依据、复核与针对性检查

继续沿用 D1049/D1063 对 GPU backing、CPU staging、独立 HOST 服务副本的区分，
不重试零 HOST budget 等已排除方向。本轮再次核查官方 vLLM v0.30.0
[worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
CPU 注册和 GPU 激活是相连但不同的步骤。该事实支持保留必要暂存，不证明消融
收益，也不将本机已资格化补丁声称为与上游逐字节相同。

`ablation-planner` 的既有 reviewer 作 provisional 只读复核，发现 legacy
HOST 源→NVMe 的遗漏，已补限制及对应 fixture。另将 publisher 未被调用的断言
移到预期异常之外，确保断言实际执行。不是独立外部审稿，没有代理 GPU 操作。

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1063_20261104/file_policy/`。

| 检查 | 结果 | 范围 |
|---|---|---|
| file_tests1 | 56 项通过，0 失败/错误 | 初版文件政策、既有 NoHOST、完整 file process 回归 |
| source_guard_tests1 | 12 项通过，0 失败/错误 | 新增 legacy HOST 源封口与两种文件 owner |
| source_guard_tests2 | 15 项通过，0 失败/错误 | 最终断言修正、上述 12 项及 3 项 Full legacy/verified copy 回归 |

三轮各自 source SHA 前后一致；它们是不同修订的 CPU 功能检查，不是性能重复，
不能相加称为独立实验。最终生产代码 SHA 以 source_guard_tests2.json 为准。
复用已有 qualification；CUDA 不可见，CPU 2/3/26/27，memory high/max 3/4 GiB，
swap 0。三次峰值分别 1,206,452,224、620,347,392、899,837,952 bytes，high/max/
OOM/kill/swap 均零。没有启动真实远端、生成新工件/trace 或覆盖历史数据。

## 不能遗漏的下一步：仍是 NoHOST，不回退到基线部署

1. **Native staging 授权/join**：普通 demand 只能使用 GPU-ready 或经 native
   owner 原子确认的精确 staging join；绑定 demand lease、source_id、content
   和活跃构造操作，不以 CPU ID、历史 receipt 或任意 plan 字符串替代。
   复用 CPU movement 与已有 `_staged_host_leases`/`_file_host_preparations`，
   不新建队列；仍由原 demand GPU 事务提交。
2. 不让 demand 等整个 proactive GPU movement：`OwnedMovementQueue.submit`
   合并时只提优先级、不替换原 action，原 action 仍受 soft admission；会形成
   消融额外阻塞。CPU movement 还会在返回前 release HOST lease，旧 receipt
   不是 demand 可直接使用的 live reference。私有 build 未完成时，保留原
   demand 先加载、build settlement 丢弃重复件的正常进展路径。
3. 旧 plan targets 只保护 proactive replacement，不原本限制 ordinary demand
   LRU。已失活且仅因 plan 保留的 CPU backing 不算 HOST hit；有真实 NVMe/file
   witness 且无 live ref/pin 时可明确 retire→reload，不新增整计划等待。实际
   引用冲突才等待其精确 release witness，不能忙循环。
4. 同步修正 controller `_ieee_protect_selected_source`：当前非 native 选择见
   任意 physical native row 就重新选源，NoHOST CPU-only 会导致重复重选。
   必须检查真实 GPU 或 host_reuse_allowed；`prepare_gpu_source` 同样区分。
5. NoHOST 的所有 GPU 替换都收集实际 NVMe/remote fallback，不只 staged 情况；
   文件替换零损失后继须持实际 GPU lease，不给 CPU lease 改标签。保持同样
   victim/slot、内容与取消/join 检查，沿用已完成的 D1063 计价。
6. 全调用链模式一致后，再加实验开关及既定单个 3B W1 NoHOST；复用 D1047
   Full，不加无理由 GPU smoke 或同键性能重复。反例、失败和数值正确性未知
   边界继续保留，不能保证或伪造 Full 胜出。

## 最新集群答复、保护与完整剩余范围

用户确认现有 k3s 不属于本项目，授权合理适配。复核 D807/D808/D809：HydraServe
已通过独立 rootless Podman 路径完成 3B W0/W1/W2，故复用，不修改共享 k3s、
权限或工作负载；pp1 观测不宣称验证了 pipeline/consolidation。这次答复不
触发重复部署、重跑或新的基线性能搜索。

本轮仍以冻结计划与 G1/G2 为准，不生成展示图。除 NoHOST 外，7B A1/累积节点、
受控 A2–A5、M2、S1–S13、CE/价格/Pareto、B1/B2/C1/C2/C3、公式/新颖性及两轮
审稿文档、IEEE 图表、统一 canonical 来源、实际 FaaSLoRA V2 非强制集成均未
取消。两项用户 dirty 和 147 项旧投稿文件逐 SHA 保护。

按 `github-sync` 进行允许路径提交、秘密/差异/校验检查及远端备份核验。
curated：`paper_results/ieee_tc/ablation/d1063_nohost_file_owner/`；实际完成的
commit/ref 见 RAW `backup.json`。独立备份分支不冒称已经整合分叉的 FaaSLoRA V2。

首次备份在 staged whitespace 检查停止，未提交/推送：新测试文件 EOF 多一个
空行。只移除这一个换行，恢复脚本逐字节验证“当前内容＋一个换行”与测试时
SHA 相同；生产代码不变，不为格式修改重跑实验。原归档不覆盖，追加格式及
恢复回执后继续同一检查点。该备份失败不是性能失败，也不改变上述检查结果。
