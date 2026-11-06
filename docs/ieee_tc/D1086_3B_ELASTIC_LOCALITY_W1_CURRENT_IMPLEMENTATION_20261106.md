# D1086：3B ElasticLocality W1 当前实现重测登记

## 科学问题与必要性

D1061 的 ElasticLocality 结果来自旧的生产实现，不能直接与 D1082/D1085 当前
同版本结果构成因果消融。D1086 只把 D1061 的既定 `gpu_locality` 复合消融
迁移到当前已用于 D1082、D1083、D1084、D1085 的实现，保持 W1 trace、模型、
请求生成合同、后端和资源口径不变。它不搜索 `locality_load_slack`，不依据
D1061 的胜负调参，也不改变 ElasticLocality 的语义。

论文问题是：在同一当前实现下，普通 GPU-locality 加负载回退能否改善
ElasticOnly 的资源—尾延迟取舍，或仍然不能解释 Full 与 ElasticOnly 的差异？
本轮是单 seed 描述性 A1 证据；若未达到数值 adapter correctness 资格，仍不
进入正式跨系统排名。

## 冻结执行键

- `RUN_ID=d1086_20261106`
- `RUN_TAG=d1086_3b_elastic_locality_w1`
- runner：`scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh`
- config：`configs/ieee_tc/20261104_d1061_3b_elastic_locality_w1.yaml`
- config SHA：`6c5cbc48cb6e03da94010a429bb40dc46c45cee965e6410d6b0f24c3553557cc`
- scenario：`tc_ieee_elastic_locality`
- workload：W1，source seed42，4,000 requests，500 logical adapters，固定
  `fixed_length_greedy_v1`，相同 prompt/arrival/target map
- model：Llama-3.2-3B-Instruct，vLLM 0.30.0，FP16，TP1，1–3 dedicated replicas
- current production implementation：`462f9987944d68370be7e411abb9252980a3e2b1`
- HOST descriptor：`fresh_d747_v1`，SHA
  `b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044`
- source trace SHA：`4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`
- remote delivery：既有只读 `prepublished_gzip_v1`；不重打包、不重建工件
- bandwidth：`local_sim_no_delay`；不改变历史带宽语义

配置中的机制边界固定为：`ieee_system_variant=elastic_locality_v1`、
`ieee_selection_strategy=gpu_locality`、`locality_load_slack=1`、
`ieee_handoff_policy=no_handoff`、`hierarchical_residency_enabled=false`、
`coordination_enabled=false`、`effective_capacity_admission_enabled=false`。
该 slack=1 沿 D448 预声明复用，不是由本轮结果选择的最优点。

## 资源与数据保护门槛

启动前必须重新确认：磁盘可用至少 150 GiB，服务/辅助尚未启动时
`MemAvailable` 至少 102 GiB，无 GPU compute 进程，无遗留 tmux 回放，远端服务
处于 inactive/dead。运行中磁盘低于 100 GiB、主机可用内存低于 16 GiB、服务
cgroup 触发 72/80 GiB 或 OOM/kill/swap 保护时，只停止本轮新增写入，不改评价
口径。旧 `final_v2/`、历史 `figs/paper/`、投稿数据、用户 dirty 文件均不覆盖。

执行前后保持两项保护 SHA：

- `configs/generated/lora_manifest_1000.json`：
  `3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df`
- `docs/ieee_tc/EXECUTION_STATUS.md`：
  `41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374`

## 唯一启动命令

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
tmux new-session -d -s tc-d1086-elastic-locality-3b-w1 \
  env FAASLORA_D1000_RUN_ID=d1086_20261106 \
      FAASLORA_D1000_RUN_TAG=d1086_3b_elastic_locality_w1 \
      FAASLORA_D1000_CONFIG=configs/ieee_tc/20261104_d1061_3b_elastic_locality_w1.yaml \
      FAASLORA_D1000_SCENARIO=tc_ieee_elastic_locality \
      FAASLORA_D1000_WORKLOAD=W1 \
      FAASLORA_D1000_HOST_DESCRIPTOR=fresh_d747_v1 \
  bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

这是唯一 D1086 执行键；不得重复启动或复用旧 D1061 raw 目录。raw 目录为
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1086_20261106/d1086_3b_elastic_locality_w1`，
curated 目录为 `paper_results/ieee_tc/ablation/d1086_3b_elastic_locality_w1/`。

## 完成与分析规则

服务、回放、watchdog 和 native GPU context 必须正常释放后，沿 D1084/D1085
收尾脚本做 D648 重分析、dispatch tier/delivery/profile 绑定审计和物理
allocation-union GPU-s 核算。必须保留失败、错误输出、机制未触发和
`n_correct=null` 事实；不把 `infra_gpu_seconds_total` 当作主资源指标。

将 D1086 与同一当前实现的 D1085 ElasticOnly W1、D1082 Full W1 做单轮描述性
对照，报告联合 TTFT/TPOT SLO、P95 TTFT、平均 TPOT、平均 E2E、物理 GPU-s/request、
scale-up/down 次数、GPU/HOST/NVMe/Remote tier 计数、locality 选择审计和
handoff/主动计划计数。若 locality 与 ElasticOnly 或 Full 方向不一致，报告
trade-off 或未分辨，而不是筛选结果、修改 slack 或人工削弱对照。

本轮不生成论文图，先保存可复核表格、JSON、请求级证据和 SHA manifest；完成后
再提交当前仓库并推送一个带日期的 FaaSLoRA backup 分支。
