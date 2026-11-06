# D1090：7B 当前实现 current-control W2 重验证登记

日期：2026-11-07  
执行键：`d1090_20261107 / 7b_current_control_w2`  
阶段：PrimeLoRA 7B 主线闭合前的同版本重验证

## 科学问题

D931 是当前 7B W2 的历史 G1 候选，但它产生于较早的代码提交。D1082–D1089
期间，PrimeLoRA 的物理 allocation-union、HOST 预算投影和请求路径审计发生了
共享实现更新；这些更新不能仅凭 3B 结果外推到 7B。因此 D1090 只回答一个
可证伪问题：在不改变 IEEE 版本论文的九个公式、W2 负载、后端和控制参数的
条件下，D931 的 current-control 7B 结果是否仍由当前实现重现。

本轮不是参数搜索、不是新的机制候选，也不把历史 D931 的较好数值预先写成
结论。若 D1090 与 D931 方向不同，保留两者的执行身份，并根据阶段分解判断
差异来自共享实现、资源计量还是随机/环境因素；不筛选有利运行。

## 冻结合同

- 基座：Llama-2-7B；vLLM 0.30.0，FP16，TP=1，单 GPU dedicated replicas。
- 代码：当前仓库 `aa6ffb4c413ce818da5c0b08c510734c0f1f7ce5`；不修改 serving
  algorithm 或论文公式。
- 配置：`configs/ieee_tc/20261029_d931_7b_current_guard_off_w2.yaml`，SHA
  `fce02375d9dd9cd67cc08e016a83b650e2880aa37fc0be91c2959d68404b0568`。
- 输入：既有 seed 42、4,000-request W2 trace、500-adapter subset、固定
  `fixed_length_greedy_v1`；不生成新的权重、工件、trace 或 adapter map。
- adapter map：`paper_results/ieee_tc/inputs/20261015_7b_w2_adapter_map_v1.json`，
  SHA `48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300`。
- 远端交付：既有只读 `prepublished_gzip_v1` 服务；请求触发的读取、传输、
  本地物化保留在时间线中，远端管理打包不进入请求 E2E。
- 主指标：联合 TTFT/TPOT SLO、Type-1 P95 TTFT、物理 allocation-union
  GPU-s/request。`n_correct` 和 `ranking_eligible` 原样保留，不能由固定长度
  或 token 绑定推断。

## 结果隔离

- raw：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1090_20261107/7b_current_control_w2`
- curated（完成闭合后才创建）：`paper_results/ieee_tc/p2_backend/d1090_7b_current_control_w2/`
- 不覆盖 D931、D989、D1082 或任何旧 `figs/paper`、`final_v2` 和用户 dirty 文件。
- 本轮不生成论文图；先完成请求、资源、远端交付和阶段审计。图表延期规则仍
  由当前 PLAN 统一约束。

## 启动前事实与停止规则

启动前必须重新确认：本机磁盘至少 150 GiB、空服务主机 `MemAvailable` 至少
102 GiB、无 GPU compute 进程、无遗留回放；远端 7B artifact service active，
健康回执含 `ok=true`、`prepublished_gzip_v1` 和 `artifact_timing_v2`。运行中仍
使用推理机 100 GiB 停止线、72/80 GiB 服务 high/max、2 GiB service swap 和
独立辅助资源域；远端 artifact 节点使用独立并发打包峰值门槛，不套用推理机
150 GiB 固定线。

任何门槛触发都保留失败 raw，不从截断运行提取性能数字，也不重用本执行键。
服务、回放、watchdog 和物理 allocation journal 必须闭合后，才可进行 D648
离线重算。

## 唯一启动入口

本轮复用 `scripts/run_all_experiments_user_scope.sh` 和 D931 配置，仅通过
环境变量指定新的输出、NVMe、HOST 和结果标签；这避免复制或修改旧 runner，
同时保证新的执行键不可覆盖 D931。

```bash
FAASLORA_OUTPUT_DIR=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1090_20261107/7b_current_control_w2 \
FAASLORA_RESULTS_TAG=d1090_7b_current_control_w2 \
FAASLORA_NVME_CACHE_DIR=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1090_20261107/7b_current_control_w2_nvme \
FAASLORA_HOST_CACHE_DIR=/dev/shm/tc-d1090-7b-current-control-w2 \
bash scripts/run_all_experiments_user_scope.sh \
  --config configs/ieee_tc/20261029_d931_7b_current_guard_off_w2.yaml \
  --scenario tc_ieee_full --full-stack
```

正式启动时必须通过受限 external replay 环境传入既有 7B W2 adapter map、trace、
remote endpoint、launch receipt 和当前 native NVML binding；不得把 shell 命令
中的 token 写入文档、日志或 Git。

## 预注册裁决

1. 4,000/4,000 完整终态、物理 allocation-union 闭合、远端交付有效且所有
   恒等式通过：进入 D648 审计。
2. 与 D931 的差异按联合 SLO、物理 GPU-s/request 和 P95 TTFT 同时报告；不以
   单一 TTFT 或 CE 取代 G1/G2。
3. 若 D1090 仍达到共同联合 SLO 参考并保持较低物理 GPU-s，则它成为当前 7B
   G1 的同版本候选，随后才进入外部 baseline 的正式比较。
4. 若仅尾延迟改善而资源变差，记录 G1/G2 trade-off；若联合 SLO 不足或执行
   合同不完整，保留为诊断/失败，不继续无依据调参。

无论结果方向如何，D931 和 D1090 都永久保留；本轮不根据结果改变评价阈值、
生成合同、远端带宽语义或对手配置。
