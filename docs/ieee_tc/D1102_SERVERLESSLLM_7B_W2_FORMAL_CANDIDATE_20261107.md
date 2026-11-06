# D1102：ServerlessLLM 7B 同合同 W2 正式对照候选

2026-11-07。D1102 是 ServerlessLLM 的一个新的、单次的 7B W2 对照执行键。
它用于补齐 D1101 历史审计指出的缺口：D941 是 W0，D555 是旧身份的 W2，
二者都不能直接充当当前 W2 主比较。

## 科学问题和不变条件

本轮只回答一个问题：在与当前 PrimeLoRA D1100 完全相同的 7B W2 请求、适配器
集合、固定长度贪心生成和远端预发布工件交付下，ServerlessLLM 的原生扩缩容、
路由、缓存和加载策略产生怎样的联合服务质量、尾部 TTFT 和生命周期资源观测。
不搜索 ServerlessLLM 性能参数，不改变其官方调度策略；只做输入视图和现有
固定合同的必要衔接。

| 项目 | 冻结值 |
|---|---|
| 模型 | Llama-2-7B |
| 请求 | 4,000，seed 42，原 W0 到达/提示词/目标长度 |
| adapter view | W2：每 100 请求轮换，48-ID 热点窗口，stride 12，邻窗重叠 75% |
| adapter universe | 既有 500 个，不生成权重；实际出现 ID 由冻结 map 决定 |
| 生成 | `fixed_length_greedy_v1`、temperature 0、top_p 1、native vLLM token IDs |
| 后端 | ServerlessLLM 官方复现路径，官方指定 vLLM 0.10.2 环境 |
| 交付 | 远端 `prepublished_gzip_v1` 服务；请求期间不打包 |
| 结果命名 | 对外图表写 `ServerlessLLM`，内部身份保留 `serverlessllm_new` |

## W2 输入投影

源 trace 不被改写：

`/home/qhq/serverless_llm_baselines/results/paper_experiments/11_remote_fair_main_local_sim_v4/20260512_123619_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1_trace.json`

源 SHA-256：`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`。

冻结 W2 map：

`paper_results/ieee_tc/inputs/20261015_7b_w2_adapter_map_v1.json`

map SHA-256：`48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300`；
已通过的 W2 view SHA 为 `5734bd88587c75dacc0d5c7534ad5b4d5d2a67aa0980f9b203dd87135d3adce7`。

baseline 仓库的新适配脚本
`serverless_llm_baselines/scripts/derive_frozen_w2_trace.py` 逐请求核验 request
ID、原 adapter、source SHA 和局部 rank，只更新 `adapter_id`、请求体中的
`lora_adapter_name` 及对应 domain；arrival、prompt、目标 token、采样 seed 和
500 个工件不变。该仓库提交为 `50968b116b2dafbe3ca9e17f2c5c2c297e225ff8`。
投影后的 JSON 只属于本轮输入视图，不是重新生成的 workload。

## 执行入口和资源

入口：

`serverless_llm_baselines/scripts/run_serverlessllm_new_remote_w2_formal.sh`

它调用原有 `run_serverlessllm_fair_experiment.sh`，只显式传入投影后的 trace、
原有 adapter subset 和固定生成合同。结果放入新的
`results/paper_experiments/16_serverlessllm_new_remote_w2_v1/` 命名空间，拒绝
复用已有结果目录。

推理服务通过 user scope 运行，资源命令固定为：

```text
systemd-run --user --scope --unit=tc-d1102-serverlessllm-w2 \
  --property=MemoryHigh=72G --property=MemoryMax=80G \
  --property=MemorySwapMax=2G --property=AllowedCPUs=4-23,28-47
```

回放器和监控沿用现有独立资源域及磁盘/内存检查；一次只运行一个重型任务。
远端服务只在本轮开始前启动并在终态后停止，健康检查、启动/停止回执和本地
完整输入 SHA 一并保存。D1102 不修改共享 k3s 集群。

## 资格和裁决

必须满足 4,000/4,000 请求终态、fixed-length native token 合同、trace/subset/map
身份完整、无请求期打包、服务进程和远端服务终止、物理 GPU 资源字段可审计。
若运行失败或联合 SLO 未达 95%，保留为真实观测，不用重跑筛选赢家；先判断是
适配/输入错误还是 ServerlessLLM 机制在该工作点的实际结果。D1102 不能把
ServerlessLLM 的部署生命周期估计冒充 PrimeLoRA 同格式物理 GPU-s；若字段不
匹配，G1 资源列保持 unavailable，并在比较表中说明。

本轮完成后才进入 D1102 结果审计和同合同 7B 比较；不修改 PrimeLoRA D1100、
D931、D1095 或 D941/D555 历史结果。
