# D940：ServerlessLLM-new fixed-length smoke 验收

日期：2026-10-30  
状态：8 请求协议 smoke 通过；不是正式论文结果。

## 运行对象

- 模型：Llama-2-7B；500-adapter subset；沿用既有 shared trace 和 subset；
- 服务：ServerlessLLM-new，源码路径为 `vendor_new_baselines/ServerlessLLM_new_main_20260518`；
- 后端：vLLM 0.10.2，探测最终选择 V1；
- 远端工件：已有预发布 gzip 服务 `192.168.4.174:18081`；
- replay：`fixed_length_greedy_v1`，目标输出上限 256，prompt 上限 759；
- smoke queue：`20261030_smoke_fixed7b_repo`，`SLLM_REPLAY_MAX_REQUESTS=8`。

## 发现与修正

初次运行得到 HTTP 401，原因是 ServerlessLLM runner 没有从本机受保护配置
继承工件服务 bearer token。随后在 fair runner 中加入只读 token 加载，不打印、
不提交 token。

第二次运行到达 vLLM 后端，但 ServerlessLLM-new 的 vLLM 0.10.2 路径把
`return_token_ids` 误传给 `SamplingParams`，导致 8/8 请求服务失败。该字段
是 fairness audit 的协议字段，不改变采样；后端现在在构造 `SamplingParams`
前消费该字段，并在请求要求时把 vLLM 原生 `result.token_ids` 放入响应 choice。
同时修正 stack launcher，使 `SLLM_REPO_ROOT` 传入 tmux head/worker/serve，
避免同步过的源码被默认旧仓库路径覆盖。

baseline runner 分支及提交：

```text
codex/ieee-tc-serverless-contract-20261030
824a53d61b348d2b0b576f1518e52b69bc73187a  # token + fixed runner
a6c5a4c54f99fbe3269d0de9fb170fd60df5766d  # propagate selected source path
```

本机 vendor 源文件中的最小 vLLM 兼容改动为：

```python
return_token_ids = bool(request_data.pop("return_token_ids", False))
response = process_output(final_output, model_name,
                          include_token_ids=return_token_ids)
```

`process_output` 仅在该协议标志为真时增加
`choice["token_ids"] = list(result.token_ids)`；没有修改调度、扩缩容、LoRA
加载、采样参数或模型权重。该 vendor 目录本来被仓库忽略，因此改动由本审计
文档和 runner 分支共同记录；后续正式运行前必须验证源文件 hash。

## Smoke 结果

从 replay JSON 逐请求检查：

| 检查 | 结果 |
|---|---:|
| 成功请求 | 8/8 |
| `generation_contract=fixed_length_greedy_v1` | 8/8 |
| `output_contract_match=true` | 8/8 |
| `completion_token_source=vllm_token_ids` | 8/8 |
| `completion_tokens == requested_completion_tokens` | 8/8 |
| prompt canonical hash 存在 | 8/8 |
| native invalid token IDs | 0 |

Smoke 的 dispatch wait 约 53–70 秒，伴随 ServerlessLLM-new 首次启动多个
vLLM 实例；它只证明协议和服务链路可运行，不进入 G1/G2、成本或尾延迟表，
也不替代 4,000 请求正式运行。

## 下一步门槛

在保持此合同、同一 trace/subset、同一远端交付和资源配置的前提下，先核对
vendor 源文件 SHA、GPU/磁盘 gate 和旧结果 hash，再启动 7B 4,000 请求
ServerlessLLM-new。若正式运行出现合同失败，保留失败证据并停止汇总，不用
文本重分词或 trace expected 值兜底。
