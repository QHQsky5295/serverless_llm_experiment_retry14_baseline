# D1087：3B ElasticLocality W1 远端服务已确认重测

D1086 已被记录为无效执行：回放虽然完成，但远端 artifact service 未启动，132
条传输均为 `not_published`、0 字节。D1087 是不覆盖 D1086 的替代执行键，只在
真实发布服务处于 active 且健康回执确认后运行；D1086 的失败证据保留不进入
任何结果表。

## 冻结键与不变量

- `RUN_ID=d1087_20261106`
- `RUN_TAG=d1087_3b_elastic_locality_w1`
- scenario：`tc_ieee_elastic_locality`
- config：`configs/ieee_tc/20261104_d1061_3b_elastic_locality_w1.yaml`
- config SHA：`6c5cbc48cb6e03da94010a429bb40dc46c45cee965e6410d6b0f24c3553557cc`
- current implementation：`462f9987944d68370be7e411abb9252980a3e2b1`
- W1 source trace SHA：`4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`
- model/backend：Llama-3.2-3B-Instruct，vLLM 0.30.0，FP16/TP1，1–3 dedicated replicas
- 4,000 requests、500 logical adapters、`fixed_length_greedy_v1`、相同 prompt/arrival/target map
- `gpu_locality`、slack=1、无 handoff、无 hierarchy、无 effective-capacity admission
- 只读 `prepublished_gzip_v1` delivery，local-sim-no-delay bandwidth semantics

启动前已通过：本机 GPU compute 进程为空，磁盘可用高于 150 GiB，主机可用内存
高于 102 GiB；远端 `primelora-artifact-tc-3b.service` 为 active/running，健康
回执为 `ok=true`、`delivery_mode=prepublished_gzip_v1`、
`timing_contract=artifact_timing_v2`。健康回执中的 clock ID 必须写入本轮 raw，
并在结束前收集远端 transfer journal。

## 唯一启动命令

先由启动控制步骤保持远端服务 active，再启动本地回放：

```bash
ssh primelora-artifact-174 \
  'systemctl --user start primelora-artifact-tc-3b.service && \
   systemctl --user is-active primelora-artifact-tc-3b.service'

cd /home/qhq/serverless_llm_experiment_retry14_baseline
tmux new-session -d -s tc-d1087-elastic-locality-3b-w1 \
  env FAASLORA_D1000_RUN_ID=d1087_20261106 \
      FAASLORA_D1000_RUN_TAG=d1087_3b_elastic_locality_w1 \
      FAASLORA_D1000_CONFIG=configs/ieee_tc/20261104_d1061_3b_elastic_locality_w1.yaml \
      FAASLORA_D1000_SCENARIO=tc_ieee_elastic_locality \
      FAASLORA_D1000_WORKLOAD=W1 \
      FAASLORA_D1000_HOST_DESCRIPTOR=fresh_d747_v1 \
  bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

raw 目录固定为
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1087_20261106/d1087_3b_elastic_locality_w1`；
curated 目录固定为 `paper_results/ieee_tc/ablation/d1087_3b_elastic_locality_w1/`。
不得复用、截断或覆盖 D1086 raw。

## 有效性门槛与收尾

只有以下条件全部满足时，D1087 才能进入 D648 离线核算：4,000 请求终态完整、
远端传输全部绑定到健康 clock 且使用 `published`/有效交付状态、service/replay/
watchdog 返回 0、native GPU context 释放确认、物理 allocation-union 完整、
mechanism contract qualified=true。若任一条件失败，保留失败记录并另开新 key，
不得从部分运行抽取优势数字。

有效收尾后复用 D1085 collector 的路径，改为
`scenario='tc_ieee_elastic_locality'`、当前 run 路径和 `ElasticLocality` 身份，
再与 D1085 ElasticOnly、D1082 Full 做同版本单轮描述性对照。主资源指标为
物理 GPU allocation-union GPU-s/request；旧 `infra_gpu_seconds_total` 只作辅证。
不生成目标优势图，先保存请求级表、机制审计、远端 journal、健康回执、closure
和 manifest；结果方向若是 trade-off 或未分辨则如实保留。

## 完成结果

D1087 已完成并通过执行闭合：4,000/4,000 请求到达终态，成功 4,000、失败 0；
launcher、回放、watchdog 返回 0，native GPU context 释放确认，物理
allocation-union 测量完整。D648 联合 TTFT/TPOT SLO 为 3,943/4,000（98.575%），
平均 TTFT 316.413 ms，P95 TTFT 691.947 ms，平均 TPOT 14.843 ms，平均 E2E
1,917.281 ms，P95 E2E 4,093.919 ms；主资源为 5,861.7327 GPU-s，即
1.465433 GPU-s/offered-request。

远端证据与 D1086 明确不同：健康 clock
`remote-process-monotonic:a92b2475079443ef9d8d90113e211837` 下收集 132/132
条 `outcome=sent`、`prepublished_gzip_v1` transfer，总读取字节 306,360,162，
`pack_performed=false` 且 `temporary_created=false`。服务在收集 post-run health
和 journal 后停止。3B 数值 adapter correctness 仍为 `n_correct=null`，因此
`ranking_eligible=false`；该结果可用于时延、SLO、资源和机制诊断，不用于正式
跨系统数值正确性排名。

与同当前实现的 D1085 ElasticOnly W1 相比，ElasticLocality 的物理 GPU-s 低
3.463%，但 P95 TTFT 高 3.814%，联合 SLO 低 0.075 个百分点；与 D1082 Full W1
相比，GPU-s 高 8.684%，但 P95 TTFT 低 33.764%，联合 SLO 高 1.950 个百分点。
因此当前证据仍是资源—尾延迟 trade-off：普通 locality 能改善资源侧，但没有
证明 Full 的全部管理机制带来统一收益，也不能宣称 slack=1 最优。完整对照和
机制触发数据见
`paper_results/ieee_tc/ablation/d1087_3b_elastic_locality_w1/completion/versus_d1085_d1082_w1.json`。

机制审计显示 8 次 scale-up、7 次 scale-down、0 次 handoff plan、0 个主动
preparation/residency epoch；路由观察记录 8,005 次 cache-stat 请求、6,542 次
confirmed-snapshot hit。该证据支持“本轮确实执行了 locality 选择、但没有混入
handoff/hierarchy”的身份检查；它不构成多 seed 稳健性或全局最优性证明。
