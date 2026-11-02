# D1024：共享预取关联补齐与剩余准备时延

2026-11-03。状态：离线分析完成；没有新增 GPU 性能运行，没有改变生产
策略、配置、远端服务或共享 k3s。此文接续 D1023；用户 dirty 的
EXECUTION_STATUS.md 不覆盖。完整研究目标保持 active。

## 当前主线及本次范围

完整重读目标文件、PLAN、冻结 V1／单轮 V2、存储规范和 D1023。使用
analyze-results 技能区分原始观察、解释和待验证的因果关系；只扩展既有
audit_d703 分析入口，不创建新实验框架，不把分析器测试算作性能实验。

D1022 的原始结果不变：4000 请求原生合同完成，3770 条联合达标（94.25%），
8062.415716 物理 GPU-s，P95 TTFT 1997.601528 ms。距 3800 门槛仍差 30。
独立数值正确性仍未知，n_correct=null；没有因此取得 G1/G2 最优资格。
当前应继续 3B 主线，不切换模型或重做已完成的外部基线。

最新集群澄清已经落实到既有 HydraServe 隔离适配：共享 k3s 不是本项目的，
不修改集群、权限或凭据；既有 rootless Podman 回放和机制覆盖限制保留。
这条迟到的授权不要求重新部署 HydraServe。

## 1. 补齐的是关联，不是补造传输

D1023 的旧入口只读取 preparation_interval 中的直接 transfer ID，因而
493 条 dispatch-Remote 请求只有 13 条进入分解。D1000 的合法到达预取使用
route_confirmed_arrival_file_preparation；此次显式 opt-in 读取该记录，并核验：

1. request/adapter、时钟、实际到达时间，以及没有 GPU reservation/admission 权限；
2. 完整 transfer 回执与本轮 503 条全局物理记录逐字段相同；
3. 发布 owner、内容 SHA、目标路径与所选来源一致；
4. 实际原生加载的 file_source_witness 与发布 owner/adapter/path/content/tier 一致；
5. 发布先于实际原生加载，逐请求分段与已封存 admission→acquired 相加一致。

不使用“相同 adapter 的最近一次传输”猜关联，不改变 profile_eligible、D 成本
模型或 EWMA。共享消费者的等待不变成它拥有的 I/O，字节只按物理 transfer 计一次。

| 覆盖项 | 数量 |
|---|---:|
| 全部请求，保留原分母 | 4000 |
| route-confirmed 到达记录 | 4000 |
| 预取 completed / skipped | 485 / 3515 |
| skipped：已有非 Remote 来源 / 无可行确认路由 | 3466 / 49 |
| 预取实际 I/O owner / 共享订阅者 | 482 / 3 |
| 预取观察到的不同物理 transfer | 482 |
| dispatch-Remote 请求 | 493 |
| 直接关联 / 新增共享关联 | 13 / 479 |
| 可完整分解的请求 / 不同关联 transfer | 492 / 491 |
| 全局物理 transfer，完整保留 | 503 |

剩余 req_03012 有显式预取/文件证据，但 native_load_invoked=false，不能构造
不存在的最终加载区间。保留其完整请求、原 SLO 和预取记录，仅不纳入该项
加载分解；不是删失败，也不是零等待。新分解误差最大为 0 ms。

全局线上字节仍为 1167457984。482 个预取 owner 关联 1118673067 字节；这个
子集不是全局传输总量。485 个预取任务仅 3 个在实际 admission 前完全结束；
它们与请求路径并行，其完整任务时长不能再加到 TTFT/E2E 上。

## 2. 新的完整条件分解

下表均为真实 admission→acquired 窗口内裁剪后的互斥区间，单位 ms。
最后一列沿 D1023 已定义的序号 ≥500 分组，不改变任何评价分母。
TTFT 违约组含同时 TPOT 违约者，不等于 TTFT-only 组。

| 区间均值 | 关联全部492条 | 其中TTFT违约101条 | 序号≥500且TTFT违约91条 |
|---|---:|---:|---:|
| admission 后至 fetch 开始 | 59.463 | 123.017 | 124.938 |
| headers | 1.776 | 2.080 | 2.017 |
| 空间预留、接收及写入 | 137.036 | 266.273 | 291.830 |
| 开始解包前间隔 | 0.171 | 0.350 | 0.380 |
| 解包与流中检查 | 183.762 | 210.557 | 214.702 |
| 解包完成至确认发布 | 321.052 | 455.613 | 486.857 |
| 发布至原生加载开始 | 280.567 | 412.422 | 427.332 |
| 原生加载 | 39.200 | 39.270 | 39.100 |
| 合计 | 1023.026 | 1509.582 | 1587.156 |

后两段发布/交接包络占全部关联请求准备时间的 58.81%，在 TTFT 违约组为
57.50%。后 3500 请求的条件组仍为 57.60%，说明该现象并非仅首轮启动。
101 条违约中，39 条最大分项为发布包络，28 条为发布后交接，19 条为
预留/接收/写入，10 条为 fetch 前等待，5 条为解包；完整分组保留在 CSV。

485 个 completed 预取任务的平均自身时长 1327.394 ms；从到达到启动平均
290.812 ms、中位数24.000 ms，包含初始排队长尾；发布至预取任务结束平均
409.257 ms。这些与上表不是互斥项，不相加，也不等同用户关键路径上的
可消除耗时。单轮条件观察不提供置信区间或优化后的反事实 SLO。

## 3. 解释、已关闭方向与下一限定问题

1. **观察：**共享预取确实触发，且 479 条缺失关联已恢复。**含义：**不能把旧
   arrival 字段缺失写成机制未运行，也不能只以 13 条代表全部远端请求。
2. **观察：**发布和交接包络大于最终原生加载，且后段仍存在。**解释边界：**
   publication 包含容量清点、目标内容验证、IPC/排队及发布，不等于已测锁等待；
   receive 包含预留和落盘，不等于纯网络。不能把整个远端包络从论文延迟中扣除。
3. **历史裁决：**D1001–D1010/D1012 的等待/重路由、D1014 文件优先查询、
   D1005–D1007/D1013 容量/回收候选已无整体收益，不重开。D1011 是画像绑定
   拒绝、没有性能证据，不能误写成实测四路并发失败。
4. **防止换名重复：**D628/D631 已撤回 SHA 移锁；D850 证明单纯提前通知或
   预发租约不解除消费者依赖；D851–D855 全 owner 线程方案负向；D887 之后
   已采用独立文件状态进程。现在看到发布包络大，本身不构成重开这些方向的新前提。

当前源码中 FileStateProcess 使用单 worker；_execute 只有操作顺序/身份回执，
没有 submit→worker-start→worker-end→parent-receive 的分段时间。publish_transfer
在 owner 内完成库存检查及目标字节 SHA；不是 HTTP 流中又做一遍相同内容 SHA：
当前 owner_verifies 下流中内容验证已关闭，仍保留归档 SHA 与结构检查。
因此不能直接声称“重复 hash 已找出，删除即可”。

**下一限定问题：**先沿既有 CPU file-state fixture/执行路径，区分发布自身的
清点/校验工作与串行 owner 的等待、回执恢复；复用已有组件，不生成整池或
再跑完整 GPU 诊断。只有确定新的可删除重复工作或可安全改变的依赖后，才
登记一个候选并进行下一唯一 Full W2。若必须补观测，只补上述具体缺失边界，
不扩大成泛化 profiling。D1023 的可达 capture 集合分析保留为另一已知问题，
不能粗删 dummy=33，也不同时堆叠两个候选。

2026-11-03 重新核查的外部依据：
[Python ProcessPoolExecutor](https://docs.python.org/3.12/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor)
区分提交、执行和 Future 完成；这不量化本机等待。
[vLLM 官方 CPU 路径分析](https://vllm.ai/blog/2024-09-05-perf-update)
说明隔离解释器和减少同步依赖是不同问题。其收益比例不能外推本机，也不
构成已经隔离进程后再做一次相同隔离的理由。

## 4. 可复现入口、验证与保护

目录：`paper_results/ieee_tc/p2_backend/d1024_shared_prefetch/`。
remote_preparation 下有 diagnosis.json、完整4000行 route_prefetch.csv、492行
remote_requests.csv、503行 transfers.csv，以及旧字段的空表（明确未记录）。
所有输入和输出 SHA 在 diagnosis.json，未覆盖 D1023 或 D1022。

```bash
python scripts/ieee_tc/audit_d703_3b_full.py \
  --raw /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1022_20261102/d1022_3b_active_lora_w2_full \
  --diagnose-remote-preparation <新的输出目录> \
  --stage-audit paper_results/ieee_tc/p2_backend/d1022_3b_active_lora_w2/completion/stages \
  --include-route-prefetch
```

旧模式默认不变。首次离线调用使用绝对 stage 目录，与旧 seal 记录的相对路径
字符串不等而退出；实际 CSV SHA 未变，没有输出。改为 seal 的原相对路径后
通过，没有放松检查或重跑 GPU。分析源 SHA：
3cefdc50f1c2f2ee4a6c9d5ad5395ad924d9e369376cbd3608ba3d06848f88b8。
新测试 SHA：780311d9c3e3c51f75d0d6dfbebc69f8dbde35b63cfa523897fe63a34d3a5cb1。

- 49项相关测试通过，19项既有依赖警告；使用已有 vLLM 环境，不安装依赖。
- 9项新引用 SHA 核验通过；原13条直接关联行逐字段不变，503条物理传输表
  与 D1023 字节相同；原请求/全局 outcome SHA 与 D1023 引用相同。
- 147项旧投稿保护通过；用户 manifest 与 EXECUTION_STATUS 两个 SHA 不变，
  不编辑、不暂存。D1023 后置备份说明随本次正常检查点提交。
- 原始JSON分析/测试串行处于3/4GiB、swap0、CPU2/3/26/27、CUDA隐藏的辅助域。
  未保留全部退出前资源峰值，不冒称完整零事件证明。无GPU任务或远端管理操作。
- 可用磁盘最后读数161399005184字节（约150.314GiB）；只新增小型汇总，
  本轮不重复全盘扫描、不删除数据。下一重任务仍即时核查空间和内存。
- PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
  V1/V2 SHA 仍为 D1023 所列冻结值，不改阈值、分母或模型配置。

完整未完范围：同版本两模型W0/W1/W2、G1/G2与外部比较、独立数值正确性范围、
A1–A5、S1–S13、CE/Pareto、B1/B2/C1/C2/C3与两轮审稿证据文档、后期图表、
最终V2集成。已完成基线适配与观测直接复用，不能将上述“外部比较”解释为
全部基线重跑。当前仍不作图、不改论文、不新增权重或负载。

后置备份回执：10文件检查点7489cc9e1a241da7010528c809e33b439da21d23
已推送并ls-remote核验origin/retry14_continuous_queue_v2与faaslora_origin的
backup/d1024-shared-prefetch-evidence-20261103。后者只是可回退备份，尚非
历史分歧V2的最终集成；没有强推。暂存路径/差异/凭据模式检查通过。
收尾时无GPU compute进程、无D1024辅助scope；可用磁盘161397600256字节。
两个用户dirty文件仍未触碰。此后置说明随下一正常检查点提交，不重复发布。
