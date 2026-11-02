# D1023：启动代价与剩余 TTFT 违约的分离

2026-11-03。状态：离线审计完成；没有新 GPU 回放，没有修改生产策略、后端
安装或远端服务。当前交接以本文和 D1022 完成节为准；不覆盖用户 dirty 的
EXECUTION_STATUS.md。完整目标仍 active。

## 主线与不变边界

本回合先完整阅读目标文件、PLAN、冻结指标 V1/单轮 V2，以及最新完成记录。
沿用 analyze-results 与 vLLM 技能：从实际日志、实际执行路径和官方源码建立
证据，不把通用技能的吞吐示例当作本项目的评价要求。

- PLAN SHA：d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
- V1 SHA：5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22。
- V2 SHA：193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。
- D1022：4000 原生合同完成，3770 联合达标（94.25%），8062.415716 GPU-s，
  P95 TTFT 1997.601528 ms。仍比 3800 门槛少 30；不因资源低于预算而判合格。
- 独立数值正确性仍未知，n_correct=null、ranking_eligible=false；不修改原始值。
- 继续 3B 主线。D810 的全部外部基线观测及适配局限保留；HydraServe 的
  隔离适配已完成，不接管非本项目的共享 k3s，不重开基础适配。

## 1. 找到实际运行日志，而非用捕获集合推造实测

复用保留在 /tmp 的原生 worker.log，分别以封存的物理 allocation journal
中的 native_workers PID 绑定。D1020 全部 4 个 worker、D1022 全部 3 个 worker
均覆盖，包含后续释放的副本，不只挑初始或最快 worker。输入日志 SHA、精确
路径、引用链和关键原始日志行保存在：

`paper_results/ieee_tc/p2_backend/d1023_native_startup/{d1020,d1022}.json`。

| 原生引擎初始化记录 | D1020 初始 | D1022 初始 | D1020 后续三个 | D1022 后续两个 |
|---|---:|---:|---|---|
| init engine（秒） | 62.08 | 82.56 | 17.75 / 15.85 / 14.89 | 34.33 / 35.05 |
| 其中编译（秒） | 31.46 | 31.00 | 0.26 / 0.33 / 0.27 | 0.29 / 0.30 |
| 内存估计用捕获（整数秒） | 11 | 20 | 4 / 4 / 4 | 12 / 12 |
| 实际服务用捕获（整数秒） | 5 | 17 | 4 / 4 / 4 | 14 / 15 |
| 实际捕获 PIECEWISE 描述数 | 22 | 88 | 均 22 | 均 88 |
| 实际捕获 FULL 描述数 | 8 | 32 | 均 8 | 均 32 |

两个初始 worker 均发生约 31 秒编译，不能把差异归结为“一个编译冷、另一个
编译热”。增加 specialization 后，日志中的捕获准备合计增加约 21 秒，方向
与 init 增加 20.48 秒相符；这是同配置变更下的阶段证据，不是受控 kernel
因果测量。不得把 init、其中 compilation、其中 capture 再相加成生命周期。
编译缓存没有清除或预热，原有不同 cache key 和实际冷/热行为完整保留。

内存估计时 FULL 只采样 2 个大描述；PIECEWISE 完整捕获。估计后原生实现
释放临时图，并在真实 KV 缓存建立后重新捕获。两次不是同一进度条的重复打印，
也不能安全地直接复用临时图。分析器对 tqdm 完成行去重，不把 88/88 的重复
显示累计为 176；描述数也不是底层 CUDA 节点数或逐算子执行次数。

### 必须修正的源码归属

所有实际日志明确 `Using V2 Model Runner`。本轮实证应引用
`vllm/v1/worker/gpu/{model_runner,lora_utils,cudagraph_utils}.py`，而不是仅凭
旧 `v1/cudagraph_dispatcher.py` 推断当前实现。D1021 的原生开关与元数据
说明继续有效，但旧 dispatcher 引用不足以证明当前 V2 的实际分派路径。

实际安装文件 SHA：

- lora_utils.py：1e90eddd9ddb4ca55e574bb28e87ae3a6f6750175a62d373102426543f4ccd66。
- cudagraph_utils.py：6e9c042890603535e300a40df8ee159dbed1058a64a83ae50ff0329e332e05ff。
- model_runner.py：174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce。
- core/sched/scheduler.py：c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884。

2026-11-03 核对的官方 v0.30.0 来源：
[V2 LoRA 路径](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu/lora_utils.py)、
[V2 执行图管理](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/worker/gpu/cudagraph_utils.py)。
公开源码用于解释；实际安装身份以以上 SHA 为准，不宣称逐字完全相同。

## 2. 可达批形状：不能粗暴删除所有大于并发上限的图

真实请求分派计算本次 scheduled request 集合中的不同 LoRA 数；每条请求最多
一个 LoRA，scheduler 的 running 上限为 max_num_seqs=8。因此真实批的活跃数
不超过 8。32 个驻留槽与此不同，不能为了减少计算而把缓存容量降回 8。

本轮确认原生集合为 [0,1,2,4,8,16,32,33]，token 捕获大小仍为既有 11 个值；
88=11×8，FULL 32=4×8，与实际日志相符。16/32 是真实批不可达的计数分支。
但是 33 是 max_loras+1 的原生 dummy/profile 分支，不能按真实请求界限直接
删除：V2 的 _dummy_run、初始化 sampler warmup 等路径仍使用该约定。
“裁成 [0,1,2,4,8] 必然安全并节省 37.5%”尚未证明，不实施这种兜底改法。

若后续选择此候选，最低风险的语义范围应是：保持驻留容量、token 轴、所有
真实活跃计数及 dummy 哨兵，按并发可达上界的原生向上取整规则排除中间
不可达计数；当前静态形状推算为 66/24，而非实测加速。不得重用临时内存
估计图，不改全局安装文件，不引入失配后静默 eager/fallback。
改变执行图集合也可能改变图内存和 KV 池，必须纳入新身份与画像，不能只
承诺启动收益。没有在本回合登记或启动这个性能候选。

这与 D724 已关闭的 token 形状覆盖不同：新增前提是 D1021 才打开 active-LoRA
专门化。不过，仅有可达性结论还不足以优先运行一个只改善启动的小优化。

## 3. 全部请求保留后的违约分布

| 同一输入序号区间 | D1020 联合通过/总数 | D1022 联合通过/总数 |
|---|---:|---:|
| 0–24 | 0/25 | 0/25 |
| 25–99 | 73/75 | 65/75 |
| 100–499 | 387/400 | 394/400 |
| 500–3999 | 3286/3500 | 3311/3500 |

首次实际 admission 之前已到达的请求为 18→27，均 TTFT 违约。此边界由
同主机单调时钟的原生请求字段得到，不用 wall-clock 日志与 monotonic 时间
直接相减；first admission 不是 engine-ready 的替代字段。该条件集合不同，
不能作为跨运行配对群组或把它移出 4000 分母。

D1022 后 3500 请求仍有 125 条 TTFT 违约，其中主导段为 admission→acquired
73 条、arrival→admission 51 条、executable→engine 1 条。因此启动不是全部
问题；27 也不是“可恢复请求数”的因果上界，因为更早 ready 会改变后续队列。
不通过离线减去启动耗时伪造达标结果。

全部 155 条 TTFT-only 中，dispatch 前 tier 为 GPU 28、HOST 9、NVMe 19、
remote 99。remote 这 99 条的平均 admission→acquired 为 1508.816 ms，
其中 before-final-load 为 1465.842 ms，最终 native load 为 39.244 ms；
该条件统计提示先检查文件准备和等待，而不是直接再次改 GPU 加载或 decoder。

## 4. 当前远端阶段分析的覆盖边界

复用现有 --diagnose-remote-preparation，没有修改其统计语义。产物位于
`paper_results/ieee_tc/p2_backend/d1023_native_startup/remote_preparation/`。

- 全部 503 个物理 transfer 保留，平均完整本地观察包络 864.340 ms。
- 其中 reserve/receive/write 平均 233.179 ms、extract/verify 186.683 ms、
  file-publication 包络 318.542 ms、publication 后返回 122.297 ms。
- 上述包络分别可能包含资源预留、写入、校验、锁等待和事件调度，不能把
  864 ms 或 receive 包络称为“纯网络”，也不能把 publication 包络称为已测锁等待。
- dispatch-remote 请求共 493，旧 direct-transfer 关联只覆盖 13 条；479 条
  明确为 shared_file_preparation_reused，另 1 条为 native source 已复用/改变。
  这些不是丢请求、零传输或失败；不把 13 条的时延分解推广到 493 条。
- 旧 arrival_file_preparation 字段在本轮未记录。D1000 的 route-confirmed
  预取使用不同证据字段，不能将 unrecorded=4000 写成“预取没有运行”。

**下一项优先工作：**沿现有 route_confirmed_file_preparation / 已到达预取的
request-transfer 关联，复用 D1022 日志补足这 479 条共享文件等待的归因；同时
核对 D1000–D1014 已关闭的预取等待/重路由/文件优先反例。只对真实剩余缺口
作一个有证据的候选，不重跑 decoder、控制延后或扩缩容参数扫描。若现有
日志已经充分，不再增加 GPU 诊断回放；只有具体必要字段缺失才提出有界观测。
执行图可达性结论保存，不因其容易修改而抢占主线。

## 5. 维护、验证与交付

- 复用既有无损压缩脚本处理已关闭 D1017 main_outcome；两次解压 SHA 一致，
  回收 525111296 B（约 0.489 GiB）。回执：
  `paper_results/ieee_tc/storage/d1023_closed_outcome_20261103.json`。
  之后可用 161412001792 B（约 150.326 GiB），新重任务仍须即时复查。
- 不删除唯一证据；D1020/D1022 未压缩主结果继续保留，不进行无边界全盘扫描。
- 仅小幅扩展既有 audit_d703 的离线 startup 模式，原分析命令不变；生产
  策略、配置、vLLM 安装、IEEE 公式、权重和输入均未改。17 项针对性测试通过，
  19 项既有警告保留。最初误用无 pytest 的 LLM 环境，测试加载失败；随后
  在已有 vLLM 环境运行通过，没有安装依赖或重复 GPU 运行。
- 所有分析在辅助 CPU 2/3/26/27、3/4 GiB、swap0、隐藏 CUDA 的 scope 中串行
  完成，未与 GPU 测量重叠。未保存全部 scope 退出前峰值，不宣称完整零事件。
- 147 项旧结果保护核验通过，回执 protected.json。两个用户 dirty 文件
  不编辑、不暂存。使用 github-sync 的检查流程备份当前可回退检查点。
- 32 项分析来源/输出 SHA 已核对；7 份 /tmp 原生日志另存无损归档，逐项解压
  SHA 与分析引用一致。原文件未删除。归档位于 raw
  `p2_backend_qualification/d1023_20261103/native_startup_logs.tar.gz`，14374 B，
  SHA 91a6b6855d9bfe5ebd9d400229761ecf89ebc12b5bcb4b86c7413fb4425c3f01。
  归档不进入 Git。临时日志消失后可从此恢复，不需要重跑实验。
- 本轮不作图、不改论文、不启动服务、不改共享 k3s 或健康远端。

未完成范围保持：同版本两模型 W0/W1/W2 与 G1/G2、独立数值正确性范围、
A1–A5、S1–S13、CE/Pareto、B1/B2/C1/C2/C3 与两轮审稿证据文档、后期图表、
最终 V2 集成。现有多项 baseline 工作已完成，不因这份待办范围又重开基础适配。

后置备份回执：12 文件检查点 8558abb7457e02fb96639c9569c88ecfa7832458 已推送，
ls-remote 核验 origin/retry14_continuous_queue_v2 与 faaslora_origin 的
backup/d1023-native-startup-evidence-20261103 均为该 SHA。后者是回退备份，
不是分歧 V2 的最终集成；未强推。staged diff/凭据模式检查通过；两个用户
dirty 文件未暂存、SHA 不变。当前 GPU compute 为空，无新回放或分析运行；
可用磁盘 161410715648 B。此后置回执随下一正常检查点提交，不重开任何已闭合实验。
