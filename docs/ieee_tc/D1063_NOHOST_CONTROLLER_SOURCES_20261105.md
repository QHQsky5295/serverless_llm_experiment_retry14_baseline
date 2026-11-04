# D1063：NoHOST 来源执行、文件后继与精确释放衔接

2026-11-05，接续 `D1063_NOHOST_NATIVE_AUTHORIZATION_20261104.md`，父提交
`7a96e20166b0d65e3b284e331a3169f039720941`。本轮完整读取最新目标、AGENTS、
原计划及前序实现证据。本页作为增量交接，不覆盖用户修改的 EXECUTION_STATUS。

**3B A1 性能点仍为 10/11。NoHOST 的 controller/file 衔接与模式安装已完成
CPU 检查；本轮尚未配置或启动新的 GPU 回放。**

共享 k3s 不属于本项目，继续保持不动。HydraServe 的独立 rootless Podman
适配和 3B W0/W1/W2（D807–D809）已有结果，直接复用，不重新部署。

## 问题、来源与本轮边界

NoHOST 必须取消独立 HOST 缓存服务能力，但 GPU backing、进行中构造和真实
借用引用仍有物理生命周期。D1063 已实现原生授权，本轮解决消费者对该授权的
衔接，避免旧 CPU backing 被继续当缓存命中，也避免真实引用没有释放等待路径。

复核本地修改历史和 D1063 前序日志，在线核对 vLLM 0.30.0 的
[worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
及 [model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)。
文件构造、CPU 注册与 GPU 激活是不同阶段；本机控制器必须尊重其实际对象生命周期。
没有重试零 HOST 预算、扩大 GPU/NVMe、凭计划意图保留独立缓存或把整个计划结束
当作某一引用的释放 ACK。

沿用 ablation-planner 的同系列 reviewer 只读复核和 vLLM 技能检查，补齐其指出
的 staging 统计兼容及 GPU fallback 释放证据。此为 provisional 实现审查，
不是独立外审，不证明性能收益。

## 已完成的衔接

1. `prepare_gpu_source` 不再无条件复用物理 CPU 行。NoHOST 的失活 backing
   保留在快照中，但实际从规划选定的 NVMe/remote 来源构造。NoHOST 正收益
   GPU recipe 只允许文件来源；不虚构 native absence，不以旧 GPU 来源路径
   猜测当前仍有文件。正常 Full 路径保留。
2. GPU executor 核验 native/objective 模式，使用同 name/content 的实际 staged
   path；证据同时保留 frozen/consumed 路径，冻结 objective/hash 不改写。
   NoHOST GPU 替换取得真实 NVMe/remote fallback，文件引用持续到原生操作
   结算；不引入 HOST 文件后继。
3. 新 `hold_gpu_source` 在同一个原生 owner 调用中验证身份、content、epoch，
   再调用现有 `acquire`，真实 pin CPU/GPU。它不激活失活 copy，也不从文件
   偷加载。文件 owner 接受这样的 GPU 后继证明；Full 仍使用原 CPU 后继。
   文件工作完成即释放对应 GPU lease，不等整个 GPU plan。
4. GPU fallback 在 hold RPC 前登记精确 owner/lease/engine/adapter completion。
   只有实际 release ACK 才置 released；明确负 ACK 为 no_acquisition，未知结果
   为 unresolved。需求容量等待复用这些证据，不把真实 GPU pin 排除出保护集。
5. CPU preparation 的精确 backing 冲突使用现有容量 waiter。新增可选
   `MovementOutcome.retry_ready`：物理 action、文件引用和 pressure 已退出后，
   共享 job 进入 waiting_release 并释放执行名额；只等精确 ACK。普通 wake
   不绕过它，异常正常失败，取消仍由最后一个 subscriber 的既有 task 生命周期
   管理。默认 None 不改变 Full 的队列语义，不新建重试/监控框架。
6. preparation cost 观测识别带明确 NoHOST 同-turn 标记的 staging；仍要求
   CPU 构造回执、内容、source incarnation、时序相符，不改名为 HOST 命中。
   共用构造或来源已变仍不更新不属于本操作的加载样本。
7. scenario `preloading.independent_host_reuse: false` 接到 file owner 和 native
   model config；首次 native budget 安装及复用成员均读回核验模式。保持正数
   HOST 安全预算、同 GPU/NVMe 预算和默认 Full；未添加启用该选项的运行配置。

论文九公式、控制策略、D1047 Full 冻结配置、trace、工件和远端 D78/D80 缓存
均未改动。CPU 临时对象和 page cache 不假装消失。

## 有限检查及完整失败记录

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1063_20261104/controller_sources/`。

| 尝试 | 结果 |
|---|---|
| controller_tests1 | 59 项，3 个新 fixture 问题：目标集不等于选择集、设置了已冻结预算、未复用既有 inline planning fixture 导致超时 |
| controller_tests2 | 59 项，1 个 fixture 初始化容量问题：未留下载归档的临时空间 |
| controller_tests3 | 79 项，1 个旧测试未 await 已变为异步的 retirement，未实际执行预期检查 |
| controller_tests4 | **80 项通过，0 失败/错误**；源码 SHA 前后一致 |

修正测试接入、初始 fixture 预算和缺失 await，没有放宽生产协议。所有尝试保留。
最后一轮含导入 14.297 秒，测试本体 5.196 秒，内存峰值 626,913,280 bytes；
high/max/OOM/kill/swap 均零。CPU 2/3/26/27、memory high/max 3/4 GiB、swap 0，
CUDA 不可见。使用 run-experiment 的既有入口规范和当前项目的受限 CPU 检查器，
没有用旧环境或无约束进程运行 GPU。

新增组合测试覆盖实际文件 owner/native owner/runner：失活 CPU 从文件重构、
实际 staged 路径、零损失文件替换的真实 GPU 后继及释放、GPU fallback 解除后
需求重查、原生模式安装、staging 成本证据。实际 CPU 冲突测试证明等待前文件
lease 和 pressure 已归零，精确 CPU ACK 后以新 epoch 重试。队列另覆盖并发一
时其他工作仍可完成、普通 wake 不绕过 ACK、共享 subscriber 存活及未知 ACK
终结失败；已有 CPU release 早到/取消检查复用。

这些测试使用真实控制和文件 owner、小文件与 CPU/native cache doubles，
不是 CUDA 数值正确性或性能资格。3B `n_correct` 仍未知，不能由 HTTP、长度
或引用链合格推造正确请求数。

## 下一主线及剩余范围

先按 github-sync 备份此 tested checkpoint，再从 D1047 生成唯一 NoHOST 运行
配置（只改该 feature 与本轮独占路径），核验真实后端的模式安装及受影响路径。
随后既定单次 3B W1 4,000 请求，复用 D1047 Full，不加同键性能重复，不重跑
其他变体。当前没有后台 GPU 作业。是否需要短原生资格取决于已有证据能否覆盖
新 staging/lease 路径，不能把本 CPU 结果称为已覆盖 GPU。

此前 LastKnown/ElasticLocality 等反例全部保留；Full 尚未在完整消融中领先，
暂不画图。NoHOST 结果完成后再综合机制证据决定一个公式保持的优化方向。

完整余项：NoHOST 3B、7B A1 和两个累积点、受控 A2–A5、M2、S1–S13、CE/
价格/Pareto、B1/B2/C1/C2/C3、公式推导/新颖性/两轮审稿文档、IEEE 图表、
最终同版本 canonical 来源以及实际 FaaSLoRA V2 非强制集成。整体目标 active。

## 保护与备份

用户两项 dirty SHA 保持不变。旧投稿 147 项 seal、secrets、显式 stage 清单和
远端 commit 核验由本 RAW `checkpoint.py` 执行，实际完成以 `backup.json` 为准。
origin V2 与 faaslora_origin 专用 backup ref 保存，不 force，不把备份冒称已
集成到分叉的 FaaSLoRA V2。原始日志不覆盖，curated 闭合文本无损 gzip 并验证
恢复 SHA。下次重型任务前重新核验磁盘 150/100 GiB 与内存保护。
