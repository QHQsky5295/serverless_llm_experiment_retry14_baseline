# D155：7B 排队瓶颈与并发配置的容量依据

日期：2026-10-02。阶段：P2/P3，Prime 7B 开发诊断；不是主比较、SLO 标定或正式性能结果。
服务代码仍为 `39cc3c1147f1699a4e7cddb54996be2564a895c3`，本次仅扩展既有离线分析器。
权威计划 SHA：`0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`；
冻结指标 V1 SHA：`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`。

## 1. 为什么转向这个问题

D154 原完整 4000 请求均满足原生生成计数合同，但数值 adapter 正确性、共同 SLO
尚未合格，不能称为 G1/G2 达标。平均 TTFT 103.898 s，其中 dispatch 等待
102.436 s；此前去除重复文件扫描没有证明整轮收益。

已封存的 D154 阶段分析显示：平均 native dispatch→末 token 为 4.3545 s，
gate→终态为 7.6242 s；二者平均并发分别为 4.3554 和 7.6256。
后者是上包络，不是准确 permit-release 时间。不能把差值全称为 CPU 计算，
也不能把首尾时段内“有请求”称为 GPU kernel 始终繁忙。

现有每卡 `runtime_concurrency_cap=2`、`max_num_seqs=2`，最多四卡：
外层 gate 上限为 8。名额覆盖准备、生成和确认释放全过程。
`slot.active_requests` 同样要等 pending/native GPU/HOST 引用确认后才释放。
因此该配置不是“GPU 始终并发生成八个请求”的保证。

历史 gate 可追溯至 `a96ab89`（2026-04-03）；`cb53f04`（2026-04-22）
增加了控制线程争用的解释注释。这段历史不是 vLLM 0.30.0/max_loras=4
条件下 cap=2 最优或 cap=4 不安全的测量证明。此前历史容量警告仍保留，
不直接抬高上限、更不删除物理准入或提前释放引用。

## 2. 已有全量日志的 KV 容量表

复用 D154 完整 metadata 的已验证紧凑副本，SHA：
`2c5d17d718f27a491e8f6ad655ea198c6917543727cfc837769589681b7c7c0f`。
没有重投影/重读原始巨型请求结果，没有 GPU 运行、远端启动或工件再生成。

扫描全部保留的 admission snapshot，816 次出现按
`(replica_id, epoch, captured_at)` 去重为 408 个；408 个重复完全一致。
同 identity 内容冲突会拒绝统计。下表 replica 使用唯一前缀，完整身份见 CSV。

| Replica | 去重观测数 | 空闲 block 最小值 | Type-1 中位数 | 最大值 | admitted 最大值 |
|---|---:|---:|---:|---:|---:|
| 1d702759 | 97 | 178 | 216 | 304 | 2 |
| 291ce1ce | 93 | 180 | 233 | 304 | 2 |
| 814dc692 | 109 | 178 | 241 | 304 | 2 |
| b1eb1672 | 109 | 178 | 245 | 304 | 2 |

全部观测 block=16 tokens、8,388,608 bytes，iteration budget=1024 tokens。
这些是加载/准入事件选择的观测，不是连续采样；admitted 包含 controller pending，
不能当作当时 GPU 正在运行的序列数。原保留投影没有整个时间线的 preemption
计数或 KV pool 总容量，因此保持未知，不补零。

结论仅为：**现有 cap=2 的准入观测中没有出现 KV 空闲 block 用尽，且有测试更高
并发的物理依据。不能据此断言整轮从不发生 preemption，或 cap=4 已经安全。**

本次按计划 §11 和 academic-plotting 选择精确状态表，而非连续曲线或冠军图。
机器可读表：`paper_results/ieee_tc/p2_backend/20261002_d155_admission_capacity/`。

## 3. 官方实现核查与单一候选

vLLM 的 `max_num_seqs` 定义为单次 iteration 的序列上限，不是涵盖应用准备与
引用释放的请求生命周期上限；其配置也参与编译图身份。
[v0.30.0 scheduler 配置源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/scheduler.py)

批处理增加可能改善设备利用，也可能增加 TPOT 或触发 KV preemption。
因此必须同时观察完成、TTFT、TPOT、实际 KV 和资源释放，不能仅看吞吐。
[vLLM 优化与 preemption 文档](https://docs.vllm.ai/en/v0.30.0/configuration/optimization/)

已核对安装环境对应定义：
`/home/qhq/.venvs/primelora_vllm0300_tc_20260925/lib/python3.12/site-packages/vllm/config/scheduler.py`，
SHA `822179da62be4bca0619c8590082ae09123f5c4415f9cc2037be673748239edb`。
这里只核对所用定义，不声称安装文件与在线 tag 逐字相同。技能示例中的通用吞吐
数值、500 ms 目标和大 GPU 参数不作为本实验的门槛或优化依据。

**下一单一假设：**旧每卡 2 请求上限把非生成阶段也占用的名额限制得过紧；
在真实容量允许时扩大 native batch 与对应应用名额，能减少 W0 排队，且不使
TPOT、正确性或物理预算不可接受。若资格或完整回放否定该假设，保留失败结果，
不通过再增 deadline、放宽 SLO 或删除释放确认来维持它。

候选为每卡 4，而非无依据倍增：

\[
c_{candidate}=\min\left(K_{LoRA},
\left\lfloor\frac{B^{observed,max}_{free}}{\lceil L_{max}/b\rceil}\right\rfloor\right)
=\min(4,\lfloor304/64\rfloor)=4.
\]

这是已有配置下的**资格候选筛选**，不是论文新公式、在线控制律或新配置容量证明。
使用已声明最大上下文 1024，不用未来实际输出长度。候选启动后重新读取真实
KV/allocator，不能假定旧 304 blocks 不变；不满足则判失败，不覆盖旧配置。

仅改变 `max_num_seqs` 和对应 effective/requested `runtime_concurrency_cap`
这一个执行容量因素至 4。保留 TP=1、FP16、GPU memory=.70、max_loras=4、
rank≤64、max_model_len=1024、max_num_batched_tokens=1024、prefix=false，
保持九个公式、排队/路由/引用/物理 admission 及远端交付语义。
不是扩大同一后台仍只能执行 2 请求的隐蔽 RPC 队列。

## 4. 下一步验收与不允许的捷径

1. 复用现有 `backend-model-check/native_source_matrix`，只建立已有请求/工件的
   四并发索引，不新建负载或权重。验证四条 native 生成实际重叠、正确 token、
   唯一 adapter/source 身份、真实 KV 与 retirement；不能把“提交四条”当成
   “真正执行四条”。四条最长合法请求所需 KV 上界为 256 blocks。
2. 新容量影响模型配置身份及服务类覆盖；D89 的 cap=2 时间 profile 不得改标签
   用于 cap=4。先收集新配置原生证据，再按现有初始化流程绑定完成长度和服务/
   准备 profile，覆盖新的 admitted 类域。不削弱旧 profile 精确身份校验。
3. 用三轮测量覆盖原有 source/rank/content 类；只复用未受影响的工件 SHA、
   输入、环境、安全资格。发生 OOM/生成/所有权错误即停止，无盲重试。
4. 合格后才做同 W0/source42/4000 的 ordinary Full；不能只展示小波次优势。
   全部阶段、失败、物理 GPU-s、TTFT/TPOT 和副作用与 D154 一起保留。
5. 本候选不是正式 G1/G2 结论；共同 warm/reference、数值 adapter 证据仍待完成。
   即使较 D154 明显改善，也必须继续核查旧 Prime 与新目标差距。
6. 当前不另做同步 CPU objective 优化、不拆分生命周期 gate、不改 timeout，
   不推进 3B、外部 baseline、消融或敏感性。

## 5. 验证回执

- 分析器新增 8 项测试，与原 timeline 测试合计 32 项通过；0.141 s 测试体。
- 包含重复去重、冲突 identity、缺字段/空集、错误计数/时间、重复请求、
  geometry 变化、显式新目录与来源 SHA 检验。
- 同一 3/4 GiB、swap=0、CPU 2,3,26,27 域内执行；实际 InvocationID：
  `a9946ef3c54f4866973bed192b9b1f75`。
- 测试命令 8.86 s / RSS 1,082,172 KiB；分析 10.88 s / RSS 1,445,280 KiB。
- 最终 memory high/max/oom/oom_kill 均 0；scope 已自动移除，GPU 空闲。
- D155 summary SHA：`e179c24144e84bc9b2c4e70106c6809ea9e2d06acfd7e03c6d8d2e85a87fffe9`。
- 此时没有新 GPU 实验运行，cap=4 尚未资格验证或用于服务；仅选定下一假设。
