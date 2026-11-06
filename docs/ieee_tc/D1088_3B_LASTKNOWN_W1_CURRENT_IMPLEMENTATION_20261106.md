# D1088：3B LastKnown W1 当前实现重测登记

## 科学问题

D1062 已经证明了 LastKnown 设计边界，但其性能结果来自旧实现，不能直接作为
D1082 Full/D1085/D1087 当前实现的单因素对照。D1088 复用已经通过 CPU 边界测试
的 `last_known_source_sample_hold_v1`（0.5 s），只限制 router 的来源排名视图；
owner、residency、admission、selected-source revalidation、EWMA、请求/远端
工件和扩缩容均保持当前 Full 语义。

本轮只回答“confirmed-state 信息传播缺口是否在当前同版本 W1 轨迹中造成可见
端到端影响”。它不关闭 tiered cache，不移除物理确认，不把 shadow 选择变成
反事实时延，也不把 0.5 s 声称为最优同步周期。

## 冻结执行键

- `RUN_ID=d1088_20261106`
- `RUN_TAG=d1088_3b_last_known_w1`
- runner：`scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh`
- config：`configs/ieee_tc/20261104_d1062_3b_last_known_w1.yaml`
- config SHA：`9b9273a8d254ac394af34ea09f4820e95832bdbf0d45c945db347a967f1792b7`
- scenario：`tc_ieee_full`（配置内 LastKnown 视图明确记录变体身份）
- workload：W1，source seed42，4,000 requests，500 logical adapters，固定
  `fixed_length_greedy_v1`，相同 prompt/arrival/target map
- model/backend：Llama-3.2-3B-Instruct，vLLM 0.30.0，FP16/TP1，1–3 dedicated replicas
- current implementation：`462f9987944d68370be7e411abb9252980a3e2b1`
- source trace SHA：`4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`
- remote delivery：启动前健康确认的只读 `prepublished_gzip_v1`
- bandwidth：`local_sim_no_delay`

CPU 资格测试已在当前 vLLM 0.30.0 环境通过：service-routing LastKnown 8 项、
request-lifecycle LastKnown 2 项，共 10 项定向测试；先前 D1062 的 26 项历史
资格记录继续作为实现背景，但不把 CPU 测试当作 GPU 性能证据。

## 单因素不变量

配置只引入 `ieee_routing_information_view=last_known_source_sample_hold_v1` 和
`ieee_routing_source_hold_s=0.5`；其余 Full 的 handoff、hierarchical residency、
coordination 和 admission 逻辑不改。正式请求仍把选中副本的原始当前来源行交给
reservation/protection/load，旧排名 tier 不得授权 GPU 引用或改写 dispatch tier。
必须分别保存 controller hold age 与底层 source observation age；候选视图分歧
不是物理 false-ready 真值。

## 资源、远端服务与数据保护

启动前必须确认磁盘可用至少 150 GiB、空服务主机 `MemAvailable` 至少 102 GiB、
GPU compute 进程为空、无遗留 tmux 回放。远端必须由
`primelora-artifact-tc-3b.service` 提供 active/running 状态，并用带认证的
`/health` 回执确认 `ok=true`、clock、`prepublished_gzip_v1` 和
`artifact_timing_v2`；健康 clock 与 post-run journal 一起保存。若服务未 active，
不得启动回放。

旧结果、旧图、用户 dirty 文件、D1086 无效 raw 全部保留；新 raw 使用独立路径：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1088_20261106/d1088_3b_last_known_w1`。
curated 输出为 `paper_results/ieee_tc/ablation/d1088_3b_last_known_w1/`。

## 唯一启动命令

```bash
ssh primelora-artifact-174 \
  'systemctl --user start primelora-artifact-tc-3b.service && \
   systemctl --user is-active primelora-artifact-tc-3b.service'

cd /home/qhq/serverless_llm_experiment_retry14_baseline
tmux new-session -d -s tc-d1088-last-known-3b-w1 \
  env FAASLORA_D1000_RUN_ID=d1088_20261106 \
      FAASLORA_D1000_RUN_TAG=d1088_3b_last_known_w1 \
      FAASLORA_D1000_CONFIG=configs/ieee_tc/20261104_d1062_3b_last_known_w1.yaml \
      FAASLORA_D1000_SCENARIO=tc_ieee_full \
      FAASLORA_D1000_WORKLOAD=W1 \
      FAASLORA_D1000_HOST_DESCRIPTOR=fresh_d747_v1 \
  bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

这是唯一 D1088 执行键；不得覆盖 D1062 或 D1086/D1087。远端只读服务应在本轮
完成并收集 transfer journal 后停止。

## 有效性和分析门槛

只有 4,000 请求终态完整、服务/replay/watchdog 返回 0、物理 allocation-union
闭合、132 条远端传输绑定健康 clock 且全部有效、profile/tier 绑定通过时，才
进入 D648 离线核算。报告联合 TTFT/TPOT SLO、P95 TTFT、TPOT、E2E、物理 GPU-s/
request、LastKnown view disagreement、hold/observation age、selected-source
revalidation、scale-up/down 和 handoff/preparation 触发计数。

`n_correct=null` 或 `ranking_eligible=false` 必须原样保留；本轮用于 confirmed-state
机制诊断，不进入正式跨系统数值正确性排名。结果若与 Full 方向相反或差异很小，
均按 trade-off/未分辨解释，不筛选 seed、不调整 hold 周期、不修改公式。
