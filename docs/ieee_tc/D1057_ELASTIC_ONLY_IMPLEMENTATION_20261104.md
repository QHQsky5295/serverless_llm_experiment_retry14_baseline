# D1057：ElasticOnly 入口与统计支持；完整回放已关闭

## 当前交接：2026-11-04，结果已审计

唯一一次 D1057 已完成 4,000/4,000 请求，GPU 已实际释放；不要重复启动。
完整结果、机制核对与下一步见
[D1057 观测分析](D1057_ELASTIC_ONLY_OUTCOME_20261104.md)。联合 SLO 为
98.20%，物理 GPU 时间 6032.623108 GPU-s，P95 TTFT 695.139877 ms。
与 D1047 Full 逐请求输入核对通过；本轮不支持 Full 在该复合消融中领先。
数值 LoRA 正确性证明、已知开发 trace、单轮及远端监测限制全部保留。

收尾已将原启动前 `remote_health.json` 按原字节保留到
`remote_health_prelaunch.json`，完成后另建 health；未覆盖证据、未改远端。
collector 与 comparison 已通过，结果在本项 curated 的 completion/ 和
comparison/。两个用户 dirty 文件继续不改不暂存，本页与结果文档承担交接。
以下运行中、待执行和“尚无结果”段落全部是历史记录，不代表当前状态。

## 2026-11-04 10:21 CST：实际运行交接（尚未完成）

10:28:31 CST 续查：原有 launcher/replay/service/watchdog 四个 PID 均仍存活，
无需重启。当前 823 arrived、819 done/ok、0 failed；已观察到三次自然
扩容和一次缩容。第一个 500-request phase 已完成，正在第二阶段。
新鲜 watchdog sample 534 无 high/max/OOM/swap、外部 compute、逃逸 worker
或保护中止；磁盘余 159860355072 B，低于下一任务 150 GiB 启动线但远高于
本轮 100 GiB 停止线，不因此中止有效测量。收尾后按既有规则清理本次可重建
缓存，再判断下一任务的空间。完整回执在 raw `live_checkpoint_001.json`。

收尾复用提示：原 D1056 collector 会新建 raw parent `remote_health.json`，
但 D1057 该名称已用于新鲜启动前 health。须先将这份启动前证据按原字节
保留为 `remote_health_prelaunch.json` 并记录移动，之后再新建完成后 health；
不能覆盖旧内容或直接复跑 collector。四个 analyzer 调用均显式使用
`scenario='tc_ieee_elastic_only'`，逐请求 key 同样使用原始 ElasticOnly 身份。
这些只是后续分析入口提示，本轮未执行重型分析、源码修改或远端操作。

此前检查点 `ef2fc5d7` 已备份；后续仅指标解释文档更新到
`432435be36b9d43c46075512c4f6eb65acd80251`。启动前逐项核验 D1057
已测源文件、配置、Full 对照、协议及用户 dirty 文件 SHA，复用 45 项通过
测试，不重复 CPU 测试、500-ID 资格、smoke 或 Full。147 项旧投稿保护清单
未变；真实远端 health 通过，复用 D80 只读 gzip，未重启或调整远端。

本次已重新完整读取当前目标（附件 `26a2d598-f71c-455b-90ae-41236c64b8cf`）、
完整 PLAN、冻结 V1/单轮 V2 和执行规范。最新用户要求强调消融优势的论文
说服力：它是后续机制改进目标，不能用来筛除负结果或改评价口径。已完成的
四格目前并不证明 Full 全面领先；D1057 先补齐弹性与主动 adapter 管理的
整体差异，尚不关闭 A1，更不替代 confirmed-state 受控证据。

10:19 CST 经原有磁盘/内存/进程检查后实际启动一次完整 3B W1：

- tmux：`tc-d1057-elastic-only-3b-w1`，启动 PID `3003757`。
- 原始路径：`d1057_20261104/d1057_3b_elastic_only_w1/`。
- service scope：`primelora-tc-svc-400b284d24a64dd0bd1e4544dcdcff5b.scope`。
- aux scope：`primelora-tc-aux-8cc2ee0186604ef9914541ed526d7610.scope`。
- 实际 scenario `tc_ieee_elastic_only`，W1 source/view SHA 与 Full 一致。
- 10:21:31 CST 轻量监测：55 arrived、51 done/ok、0 failed；尚未结束。
  同时监测到 2 张实际持有 GPU（包含启动中的副本，不等同于 live 摘要中
  已就绪实例数）。服务内存约 9.71 GiB、swap 0、high/max/OOM 事件 0；
  主机可用约 94.56 GiB，磁盘约 150.53 GiB，未出现外部 compute 或保护中止。

新鲜核查见 raw parent 的 `fresh_launch_check.py/json` 和
`remote_health.json`；当前监测证据在 `launch.launch/`。这些是启动/进度
证据，不是最终性能、数值正确性或正式排名证明。日志的 legacy 5000ms
达成率不能代替完成后冻结 D648 阈值下的联合 SLO。

**下一动作：只监测这个已运行执行键，禁止重复启动。** 结果关闭后，复用
D1056 collector 并显式传入 ElasticOnly scenario，检查完整输出、物理释放、
实际机制触发并与同一 D1047 Full 比较。测量期间不构建、测试、压缩、推送
或修改远端。新进度文档待测量结束再备份，避免备份影响测量。

两个既有用户 dirty 文件仍保持原字节，故不改旧 `EXECUTION_STATUS.md`；
本页承担当前交接。总体目标 active；ElasticLocality、LastKnown、NoHOST、
7B 对应消融、受控机制、敏感性及论文交付范围不删减。

以下保留此前实现记录和原始待执行说明；启动事实以上述时间戳记录为准。

2026-11-04。本次继续完整 PLAN 的 A1/A5，不新增外部 baseline、不改变
G1/G2、不重新运行 D1047 Full。上一次完成的是 D1058 指标取舍分析；
本次恢复机制消融主线。完整目标仍 active。

## 研究问题与本次进展

ElasticOnly 检验“相同弹性控制下，主动 adapter 管理整体是否有额外收益”。
这是复合消融，不能独自证明 confirmed-state propagation。它沿用 D1047
的 3B W1、1–3 实例、vLLM 0.30.0、输入和生成、D1031 profiles、D648 SLO、
D657 预算、物理资源持有计量及共同资源包络。

- 复用已有 owned `load_only`；不切换到另一套 legacy 资源管理路径。
- 明确关闭 activation handoff、主动协调/分层规划/软 admission、主动
  forwarding/HOST promotion；文件层采用普通 demand LRU。
- 保留物理容量检查、原子 reservation、引用与取消、原生按需缓存以及
  已到达请求驱动的来源准备。没有按 phase 清缓存或人为削弱按需加载。
- 新身份为 `elastic_only_v1` / `tc_ieee_elastic_only`，不能冒用 Full 或
  ElasticLocality。Full 的默认行为、九公式及已有 ElasticLocality 合同不变。

依据沿用入口审计，并再次读取固定版本的
[vLLM 0.30.0 按需 LRU 实现](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)。
关闭主动管理不代表取消正常缓存；该源码是设计边界参考，不是性能证据。
历史 D453 仅复用实现经验，旧 7B W0 数值不混入本轮 3B W1。

## 实现和验证范围

1. 现有 runner 扩展组合身份及实际 router/file-owner 检查；旧 Full 默认不变。
2. 现有 D1000 launcher 增加显式 `FAASLORA_D1000_SCENARIO`，默认仍为
   `tc_ieee_full`，不复制启动框架。配置三个工作路径与实际 launcher 一致。
3. 现有 D703 的 SLO、阶段、profile 和收尾分析函数增加 keyword-only
   `scenario`，默认 Full。新分析记录保存原始 scenario，并核对原始 metadata、
   detailed key/name、配置及分析身份。旧无 scenario 的 D714 记录仅允许
   按其历史 Full 身份读取，不回写原始结果。
4. 复用原 CPU fixtures：20 项通过，覆盖 Full/ElasticLocality 回归、
   ElasticOnly 组合、真实控制入口扩容且无主动准备、按需引用/释放。
   同时修正既有测试漏 `await`，不是生产算法补丁。
5. 统计入口的 25 项针对性测试通过，shell 语法通过；19 条现有依赖
   deprecation 警告保留在日志。两个测试资源域无 high/max/OOM/swap。
   没有把 CPU fixture 当作 GPU 性能或机制实际触发证据。

配置：`configs/ieee_tc/20261104_d1057_3b_elastic_only_w1.yaml`。
证据：`paper_results/ieee_tc/ablation/d1057_3b_elastic_only_w1/prelaunch/`。
原始测试记录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1057_20261104/`。

## 用户本次集群答复的处理

现有 k3s 不是本项目集群。复核 `HYDRASERVE_ISOLATION_DECISION_20261012.md`
以及 D807/D808/D809：独立 rootless Podman 的必要适配和三种 3B 完整回放
已有证据。本次不申请该共享集群 kubeconfig，不接管、重启或改权限/网络，
不重复部署 HydraServe。保留官方专用后端以及适配范围；这些运行都是 pp1，
不据此声称验证了 pipeline/consolidation。既有异常与适配限制继续披露。

## 下一动作和运行身份

先提交并非 force 备份本检查点；随后在原入口新鲜核查磁盘/内存/GPU 和
远端 health，再启动唯一一次完整 3B W1 ElasticOnly。没有要求重新进行
500-ID 资格、无理由 smoke 或 Full 重跑。执行命令为：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
tmux new-session -d -s tc-d1057-elastic-only-3b-w1 \
  env FAASLORA_D1000_RUN_ID=d1057_20261104 \
      FAASLORA_D1000_RUN_TAG=d1057_3b_elastic_only_w1 \
      FAASLORA_D1000_CONFIG=configs/ieee_tc/20261104_d1057_3b_elastic_only_w1.yaml \
      FAASLORA_D1000_SCENARIO=tc_ieee_elastic_only \
      FAASLORA_D1000_WORKLOAD=W1 \
      FAASLORA_D1000_HOST_DESCRIPTOR=fresh_d747_v1 \
  bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

本页是待执行命令，不是已启动回执。运行期间只作轻量监测，不测试、构建、
压缩、推送或改远端。收尾复用 D1056 collector，四个新参数均显式传入
`scenario='tc_ieee_elastic_only'`；请求/机制分析使用相同原始 key，不能改名
JSON。继续记录 actual routing、handoff、tier/admission 触发与缺失项。

当前无新性能数值。source42 已知开发负载、单轮 V2、全量非零 LoRA 数值
正确性证明缺口及 `n_correct=null` 均保留。负结果、错误和局限不得隐藏。

ElasticLocality、LastKnown、NoHOST、7B A1/两个累积节点、受控 A2–A5、
S1–S13、成本/CE/Pareto、B1/B2/C1/C2/C3/推导与两轮审稿文档、最终 IEEE
图表及 FaaSLoRA V2 集成全部仍在范围内。检查点备份不等于最终 V2 集成。
两个既有用户 dirty 文件不编辑、不 stage；本页承担当前交接，保留旧状态
文件的原字节。
