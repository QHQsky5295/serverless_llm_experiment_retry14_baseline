# S-LoRA 3B 历史结果入口核对（非性能运行、非独立审计）

2026-10-23，D766 ServerlessLLM W2运行期间仅做低负载只读记录核对。
本记录不改变当前执行配置，不启动另一个服务，不生成新权重或负载。
正式复用判定还应沿现有D425来源检查入口核对配套配置/日志；这里不声称
全盘穷尽搜索或独立完整审计通过，也不把字段存在性当作数值正确性证明。

## 找到的四份既有完整3B回放

根目录：`/home/qhq/serverless_llm_baselines/results/paper_experiments/`。
下表路径均相对该目录，文件保持原位，没有复制、改写或删除。

| 历史组 | 文件字节 | 原reported success / requests | completion_tokens == guard_max_tokens |
|---|---:|---:|---:|
| 20260509主点mainv1 | 9327552 | 4000/4000 | 2209/4000 |
| 20260509主点mainv2_seq8 | 9327256 | 4000/4000 | 2211/4000 |
| 20260512 local-sim | 9327337 | 4000/4000 | 2211/4000 |
| 20260513 real-remote | 9327526 | 4000/4000 | 2213/4000 |

四个文件均：`min_output_tokens=0`、`routing_policy=round_robin`；全部请求的
`completion_token_source=local_generated_text`，非空`server_metrics`数量0。
请求中`native_token_ids`、`native_first_token_monotonic_s`或
`native_last_token_monotonic_s`三个检查字段的存在数均0。字段名检查并非
对所有替代计量位置的穷尽证明；但明确的文本重分词来源已不足以支撑当前
原生整数token主计量合同。表末列只是旧字段关系，不是原生fixed-length资格。

精确回放路径：

1. `03_main_comparison/20260509_134259_llama32_3b_main_s8/raw/replay/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_slora_dp4_tp1_replay.json`
2. `03_main_comparison/20260509_201205_llama32_3b_main_s8_v2/raw/replay/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv2_seq8_slora_dp4_tp1_replay.json`
3. `11_remote_fair_main_local_sim_v4_patch/20260512_223735_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1/raw/replay/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1_slora_dp4_tp1_replay.json`
4. `12_remote_fair_main_real_remote_v1/20260513_160342_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1/raw/replay/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_slora_dp4_tp1_replay.json`

真实远程回放的`metric_schema_version=e2e_v3`；其TPOT定义使用service
completion而非原生末token时刻。因此不能仅因该组曾使用真实远程，就把旧
数字当作当前共同输出长度、首末token和物理GPU生命周期口径的主结果。
本轮没有从旧时延中减估计开销，也没有用HTTP成功填`n_correct`。

## 下一项复用边界

- D753已核验vLLM D653/D654/D656三个完整3B观测与D657参考，继续复用；
  不为基线另跑相同执行键。
- S-LoRA复用现存原生环境、模型/LoRA集合、既有加载兼容修改以及D591的
  当前测量/启动路径；D432–D435/D542/D591属于7B证据，不能升级为3B功能
  或性能证据。先核对3B配置、原生模块/形状与已有3B适配，再只补受影响路径。
- 四份旧完整回放作为历史及来源诊断保存，不直接进入新主比较。是否存在
  其他更晚的3B fixed-output证据，沿已有来源索引继续核查后再决定重跑范围；
  本记录不声称上述四份涵盖服务器全部3B历史。
- 不因ServerlessLLM或S-LoRA某项SLO较低而开展额外基线参数搜索。
  D766自然终态、释放/分析/封存和必要空间回收先完成，才启动后续重型任务。
- 本次未做文件哈希扫描、GPU测试、独立语义审查或展示图；来源SHA核验可
  在当前测量结束后的正常来源/提交检查中复用现有工具完成。

此记录仅用于减少下一阶段重复查找与无必要重跑，不改变冻结指标、所有
失败记录、完整实验范围或PrimeLoRA最终优化目标。
