# D1063：NoHOST 来源复核与精确 CPU 释放等待

2026-11-04，接续 `D1063_NOHOST_STAGING_JOIN_20261104.md`。已完整读取更新后的
目标 `/home/qhq/.codex/attachments/0cf15a55-6d42-4032-b0eb-fc9169a9be81/goal-objective.md`
及原计划。用户新增“允许来源授权与文件层衔接”，本轮据此继续同一消融主线。
前一检查点有已测试、已备份的 staging 实现，属于 progress，不是等待中的 GPU 作业。

**3B A1 性能点仍为 10/11；NoHOST 没有启用，没有新的性能回放。**
本页是增量交接，不覆盖用户已修改的 `EXECUTION_STATUS.md`。

## 为什么需要这些修改

NoHOST 需要消除独立 HOST 缓存服务能力，同时保留 GPU 执行及进行中操作必需的
CPU 对象。计划只是候选意图，不是实际借用引用。若普通请求已经替换 GPU
adapter，而旧计划仍让其 CPU backing 存活，就可能把关闭的缓存能力重新引入。
反过来，实际借用尚未结束时强删对象也不合法。

本轮复核 D1063 生命周期、staging 和文件 owner 实现，以及现有计划注册约定：
注册计划不预留或 pin CPU/GPU，也不限制 ordinary demand。没有重试零 HOST
预算、扩大 GPU/NVMe、把等待整个 GPU plan 当释放证据等方向。
再次在线核对 vLLM 0.30.0 的
[model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
和 [worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)。
注册 CPU 对象和激活 GPU 对象是关联但不同的生命周期；本次是本机 NoHOST
合同实现，不声称上游存在 PrimeLoRA 的计划/引用协议。

## 已实现及验证的边界

1. **NoHOST CPU 退役不再由 plan-only 意图阻挡。** GPU/slot backing、真实
   CPU/GPU pin、GPU/HOST 引用、staging lease、尚未结算 private build 仍保护
   对象。proactive GPU victim 的 all-target 保护不变。替换的原有 victim/slot
   后置检查仍先完成，最外层才退役；快照、失败和软延期不主动回收。
2. **两个请求侧来源入口检查复用资格。** cached-only probe 不对 NoHOST
   CPU-only backing 发起 acquisition；选中文件/remote 来源后的复核不再因
   该 backing 的存在而循环拒绝文件。物理 CPU 行仍保留在原始快照和 identity
   frontier，不伪造缺失。GPU 与默认 Full HOST 服务资格保留。
3. **protected-file 容量冲突绑定本次实际 epoch。** 文件引用本来允许跨越
   无关 epoch 变化；等待使用同 owner、本次原子拒绝回执的 epoch，而非旧
   admission epoch。后续仍须释放证据、重新观察和重新申请，不宣称容量预留。
4. **CPU movement release 单独可等待。** 在 prepare RPC 前登记实际
   `(owner, lease, engine, adapter)` completion；只在对应 HOST release ACK
   后标 `released`。明确没有取得引用标 `no_acquisition`，未知结果标
   `unresolved`。HOST release 回执增加请求 lease ID，首次/幂等路径均覆盖。
   等待者消费精确 HOST blocker 的 ACK，不等待整个 GPU plan。完成的 witness
   保留在本 runner 内以避免先释放、后收到冲突的丢失唤醒；不是长期持久缓存。
5. **capacity witness 重试保持原命令 schema。** 复核发现清除旧 source guard
   会删除 core 必需的 `expected_source_id` 键。现保留该键为 `None`；新增检查
   让第一次和第二次命令都通过实际 core schema/observation 方法。

Full 的路由、规划、GPU victim、安全及预算策略没有改动；新增 ACK 字段和
容量等待缺陷修复也适用于 Full，不能把这类共同正确性修复单独归因给 NoHOST。
论文九个公式、配置、工件、trace、远端交付和已封存性能数据不变。

## 检查结果和失败记录

沿用 `ablation-planner` 的既有 reviewer 做同系列 provisional 复核；没有
把它称作独立外审。其指出的 core schema 问题已在运行前修正。

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1063_20261104/source_authorization/`。

| 项目 | 结果 |
|---|---|
| 首次 scope 启动 | 自定义名称不符合已有 UUID scope 约定，测试前退出；记录保留，未放宽约定 |
| source_tests1 | 64 项，2 个新 fixture 错误：混用完整/紧凑 wire；辅助占用请求缺文件引用 |
| source_tests2 | 65 项，1 个新 fixture 错误：传入的文件 row 缺 owner/epoch 外框 |
| source_tests3 | **67 项通过，0 失败/错误**；source SHA 前后一致 |

失败通过修正测试的真实来源构造解决，没有放宽生产检查。第三轮包含 8 项
新增测试：物理 CPU 行与服务资格分离、真实文件 capacity waiter、实际 core
重试 schema、CPU release 不等待 GPU plan、早到 ACK、等待者取消、丢失 ACK。
还复用 GPU/Full HOST 来源、NoHOST 生命周期/staging、native capacity、CPU
movement 和旧源真正删除后 typed supersession 的现有测试。不是重复性能运行。

最终测试含导入耗时 10.075 秒，测试本体 1.352 秒；内存峰值 608,714,752 bytes，
high/max/OOM/kill/swap 均零。三次均 CUDA 不可见、CPU 2/3/26/27、memory
high/max 3/4 GiB、swap 0；终态 scope 已 inactive/dead。测试为 CPU contract
fixture，不证明 CUDA 数值正确性、真实 GPU 性能或 NoHOST 全链路已合格。
core retry 测试复用实际 schema/observation 方法，但 worker v2 receipt 是明确
标注的测试替身；不能将此解释成完成真实 KV/LoRA 数值验证。

## 下一步：同一 NoHOST 主线，不额外加性能门槛

1. native `hold_host_source`、已注册但失活的 backing、proactive commit 仍须
   完成禁止独立 HOST 复用的授权。真实引用冲突可使用本轮精确 CPU release
   witness；有合法文件来源时执行真实加载，不能只换 tier 标签。private-build
   保护暂留，但不因此授权旧 backing 成为 HOST 命中。
2. planner 内 `prepare_gpu_source` 仍无条件复用 physical native row，须完成
   相同政策衔接。源消失已有 supersession 测试，不再复制测试框架。
3. NoHOST 每次 GPU replacement 都须获取实际 NVMe/remote fallback；文件
   替换的零损失后继须真实 GPU lease。不得把 CPU 引用重命名成 GPU 引用。
4. `experiment_stack`/runner 的模式安装与读回尚未贯通。完成上述受影响路径
   后登记并运行既定**单个 3B W1 NoHOST**，复用 D1047 Full，不添加同键重复。

本轮不启用半完成模式、不额外重跑 Full、不根据输赢改阈值或删除负面消融。
已有 LastKnown、ElasticLocality 等反例保留；尚未证明 Full 优势的事实不改变。
NoHOST 完成之后才依据完整消融证据确定一个有因果依据的优化方向，不凭期望
宣称各机制均有效。

## 保护、备份与完整余项

共享 k3s 不动；HydraServe 已完成隔离适配及 3B W0/W1/W2，不重复部署。
远端 D78/D80 不可变缓存复用，本轮没有远端服务/网络/工件操作。
未生成图、权重、trace、工件池；未覆盖旧结果。开工磁盘可用
162,939,424,768 bytes（约 151.75 GiB），下一个重型任务前重新核验。

依据 `github-sync` 做源码、测试和 curated 证据检查点；结果以 RAW `backup.json`
的 commit/远端 ref 核验为准。实际 FaaSLoRA V2 已分叉，独立备份 ref 不代表
完成集成，不强推。用户两项 dirty 与 147 项旧投稿文件保持原 SHA。

未完成范围完整保留：NoHOST 3B、7B A1 和两个累积点、受控 A2–A5、M2、S1–S13、
CE/价格/Pareto、B1/B2/C1/C2/C3、公式推导/新颖性/两轮审稿文档、IEEE 图表、
同版本 canonical 来源及 FaaSLoRA V2 非强制集成。3B 独立数值正确性仍未知，
不能从 4,000 个成功或 token 数正确推造 `n_correct`。整体目标保持 active。
