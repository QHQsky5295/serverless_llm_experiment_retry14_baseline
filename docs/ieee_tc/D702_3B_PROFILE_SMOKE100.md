# D702：3B D701-profile 短资格回放

执行键：`D702_20261019`
状态：资格通过，性能与排名不合格（不是性能排名运行）。

## 科学目的

D701 已证明 3B 的 native service/preparation profile 在 160/160 reachable
service classes 上完整且可由 frozen loaders 读取。D702 只检验这份 profile 是否
能被 PrimeLoRA 的完整运行时在固定输入合同下无 fallback 地接受，并保留原生
请求、adapter 绑定、远端交付和物理资源账目。它不尝试通过短前缀证明 G1/G2，
也不替代 4,000-request Full W0。

## 冻结合同

- 模型：Llama-3.2-3B-Instruct，TP=1，RTX 3090；不改变论文公式或控制策略。
- 负载：既有 W0 trace 的前 100 条请求、既有 500-adapter 子集和固定生成合同；
  不生成新 trace、LoRA 或权重。
- `fixed_length_greedy_v1`、prompt 上限 759、输出上限 256、并发 8。
- 远端：已有 `prepublished_gzip_v1` 只读工件；请求期读取、传输、本地物化和
  必要解包计入请求路径，远端部署期归档/运维不计入；请求期打包若出现则运行
  违规，不做事后扣除。
- 唯一机制变化相对于历史 D696/D697 是 profile provenance：service/preparation
  profile 改为 D701 严格导出，输出和缓存目录使用新的独占路径。

配置：`configs/ieee_tc/20261019_d702_3b_profile_smoke100.yaml`
入口：`scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh d702`
原始输出：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d702_20261019/3b_profile_smoke100`

## 预先判定

短资格必须保留完整计划分母，且满足：

1. 100/100 请求终态、目标/实际 token 一致、adapter identity 绑定完整；
2. runtime profile loader 无 nearest-class、插值、零值或 fallback；
3. 远端 transfer 内容核验通过且 `remote_pack_performed=false`；
4. physical GPU allocation、worker、HOST/NVMe workspace 和远端 owned service
   全部释放；
5. watchdog 的 memory high/max/OOM/swap 事件为零。

若未达到上述合同，保留失败证据并停止在原因定位；若通过，也只授权进入下一
个完整 3B PrimeLoRA 资格回放，不产生性能图或 baseline 比较结论。

## 实际结果与严格裁决

D702 已在独立的 100-request W0 前缀上完成。严格回执为
`paper_results/ieee_tc/p2_backend/d702_3b_profile_smoke100/qualification.json`，
机器可读 manifest 同目录保存；原始结果仍保留在配置指定的 raw 目录，未覆盖
D645、D697 或历史图表。

| 检查项 | 观测 | 裁决 |
|---|---:|---|
| 终态请求 | 100/100 | 通过 |
| 固定生成契约 | 100/100，`completion_tokens=min(source_expected,256)` | 通过 |
| 原生 token 来源 | 100/100，vLLM `token_ids`，native terminal | 通过 |
| adapter/content/lease 绑定 | 100/100，29 个实际 adapter identity，全部 request release 完成 | 通过 |
| service profile | D701 SHA，160/160 类别，100/100 请求命中 | 通过 |
| preparation profile | D701 SHA，96 类别，55 个非 GPU 请求命中 | 通过 |
| profile fallback | 0；无 nearest/interpolation/default | 通过 |
| 远端交付 | 29 次，全部 `prepublished_gzip_v1`，请求期打包 0 | 通过 |
| 资源安全 | memory high/max/OOM/oom_kill/swap 全为 0 | 通过 |
| 生命周期释放 | native contract 100/100，allocation journal closed，远端服务已停止 | 通过 |

该结果只证明 D701 的精确测量 profile 能被完整运行时加载并用于一个短前缀，
不证明数值 adapter correctness、4,000-request Full W0、G1/G2、共同 SLO 或
任何系统排名。短前缀的观测值（平均 TTFT 529.451 ms、P95 TTFT
1191.936 ms、232.160 physical GPU-s）只作为资格诊断，不生成图，不与旧的
3B Full 结果拼接，也不外推到完整工作负载。下一步仍需使用同一 D701 profile
执行一次新的完整 3B PrimeLoRA W0；完整运行未通过前不能进入 3B baseline 对比。
