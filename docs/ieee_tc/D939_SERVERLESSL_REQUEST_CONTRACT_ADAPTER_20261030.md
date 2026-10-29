# D939：ServerlessLLM-new 公平 runner 的生成合同适配

日期：2026-10-30  
状态：runner 适配完成；尚未产生新的 ServerlessLLM 4000 请求结果。

## 发现

ServerlessLLM-new 的原公平入口
`/home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_new_remote_formal_queue.sh`
调用 `run_serverlessllm_fair_experiment.sh` 时没有传递
`--generation-contract`。`replay_openai_trace.py` 的默认合同因此是 `legacy`，
即便 trace 本身包含 fixed-length 所需的源输出长度，也不会强制
`ignore_eos=true`、无 stop sequence 或原生 token-ID 审计。

这不是 ServerlessLLM 调度算法的问题，而是跨系统请求合同没有落到 runner 的
协议适配问题。历史 ServerlessLLM 结果继续保留为旧合同证据，不被事后改写。

## 适配内容

baseline 仓库单独分支：

```text
codex/ieee-tc-serverless-contract-20261030
eaa46e8c98f230fbe91bcf66bbea7e5cfba4b73a
```

修改文件：

- `scripts/run_serverlessllm_fair_experiment.sh`
- `scripts/run_serverlessllm_new_remote_formal_queue.sh`

入口现在显式设置并传递：

- `generation_contract=fixed_length_greedy_v1`；
- `fixed_output_max_tokens=256`；
- `fixed_prompt_max_tokens=759`；
- `--vllm-native-generate`，使 vLLM 返回的 token IDs 成为主计量来源；
- `--force-stream` 与 `--include-stream-usage`，保持 TTFT/TPOT 的流式观测。

当 fixed-length 合同下后端不是 vLLM、或固定上限不是正整数时，runner 现在
直接失败；不再静默降级到 Transformers 或 legacy 生成。legacy 仍可通过环境
变量显式选择，以保留历史运行兼容性。

远端适配器仍通过已有预发布 gzip 对象在请求路径读取；本次没有改变
ServerlessLLM 的 router、autoscaler、模型加载或实例生命周期。没有新增工件、
没有重生成 trace/adapter subset，也没有把请求路径打包移到计时范围外。

## 验证边界

已完成 shell 语法检查和 diff 空白检查。下一步必须先做 bounded vLLM/ServerlessLLM
fixed-length smoke，验收每个请求的 target map、prompt hash、native token IDs、
`actual_tokens == target_tokens` 和阶段恒等式；smoke 未通过前不得启动 4000 请求。

该适配只修复共同实验协议，不构成 ServerlessLLM 性能优化，也不预先承诺其与
PrimeLoRA 的最终排名。正式结果仍必须以同一 trace、adapter subset、远端交付、
资源限制和 G1/G2 指标重新判断。
