# Session Handoff: 2026-05-18

This is the current restart point for PrimeLoRA/FaaSLoRA after the 3B/7B
closed-loop paper experiments, backend portability extension, and true-remote
artifact mirror were completed.

## 1. Repositories And Branches

Primary PrimeLoRA/FaaSLoRA repository:

```text
path:   /home/qhq/serverless_llm_experiment_retry14_baseline
branch: retry14_continuous_queue_v2
remote: faaslora_origin -> https://github.com/QHQsky5295/FaaSLoRA.git
latest pushed commit at handoff: 098604e Add true-remote full figure artifact set
```

Baseline/fair-comparison repository:

```text
path:   /home/qhq/serverless_llm_baselines
branch: main
remote: origin -> https://github.com/QHQsky5295/serverless_llm_experiment_retry14_baseline.git
```

Important dirty-file note:

```text
/home/qhq/serverless_llm_experiment_retry14_baseline/configs/generated/lora_manifest_1000.json
```

This file is currently modified but intentionally not staged or committed in
the final paper sync. Do not include it in unrelated commits unless the user
explicitly asks to update the generated manifest.

## 2. Current Terminal And Machine State

As of this handoff:

```text
tmux: no experiment session is running
GPU: all four RTX 3090 GPUs are idle
true-remote full-figures queue: completed
```

Quick verification commands:

```bash
tmux ls || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
cd /home/qhq/serverless_llm_experiment_retry14_baseline
git status --short --branch
git log --oneline -5
```

Expected status is clean except for the pre-existing
`configs/generated/lora_manifest_1000.json` modification.

## 3. Final Paper Data State

Default paper result set:

```text
figures:      /home/qhq/serverless_llm_experiment_retry14_baseline/figs/
curated data: /home/qhq/serverless_llm_experiment_retry14_baseline/paper_results/final_v2/
```

This is the primary closed-loop dataset for the current paper draft. It uses
the completed Llama-family main comparison with:

- Llama-2 7B
- Llama-3.2 3B
- 4,000 requests
- 500-adapter pool
- Zipf exponent 1.0
- hot-set rotation every 500 requests
- time scale `s8`

The final table/figure intent is to present the Llama-2 7B and Llama-3.2 3B
main comparison together. Llama-2 13B is historical exploration and should not
replace the 3B/7B paper snapshot unless the user explicitly changes the paper
plan.

True-remote mirror result set:

```text
figures:      /home/qhq/serverless_llm_experiment_retry14_baseline/figs_remote_full_real_remote_v1/
curated data: /home/qhq/serverless_llm_experiment_retry14_baseline/paper_results/final_remote_full_real_remote_v1/
```

This is a non-overwriting mirror generated after true HTTP remote artifact
transfer was enabled for the new real-remote experiments. It should be treated
as remote-realism evidence or appendix/supplementary material unless the user
decides to replace the default paper result set.

## 4. Completed Experiment Families

The following experiment families are closed under the current scope:

- Main comparison on Llama-2 7B and Llama-3.2 3B.
- TTFT decomposition tables.
- Lifecycle cost figure.
- Service-readiness audit and table.
- Mechanism ablation.
- Control-path overhead audit.
- Backend portability / sensitivity with:
  - vLLM
  - PrimeLoRA-vLLM
  - SGLang
  - PrimeLoRA-SGLang
- Load sensitivity and adapter-pool sensitivity.
- True-remote mirror for the above figure families, generated without
  overwriting the original `figs/` snapshot.

The completed true-remote queue id is:

```text
20260514_real_remote_fullfigs_v1
```

Queue status markers:

```text
/home/qhq/serverless_llm_experiment_retry14_baseline/results/remote_full_figs_queues/20260514_real_remote_fullfigs_v1/state/
```

Expected completed markers:

```text
00_remote_health.done
10_load_queue.done
20_adapter_pool_queue.done
30_ablation_queue.done
40_build_figures.done
```

## 4.1 New Serverless Baseline Candidate After Handoff

ServerlessLLM-new was reproduced after the 2026-05-18 handoff as a separate
true-remote candidate. It does not overwrite the old ServerlessLLM baseline or
the default paper data.

```text
baseline project: /home/qhq/serverless_llm_baselines/ServerlessLLM_new_project
upstream commit:  9f50241baa5386e06a9321c51f19a9ef5f964c2b
result section:   /home/qhq/serverless_llm_baselines/results/paper_experiments/15_new_serverless_baselines_remote_v1/
curated bundle:   /home/qhq/serverless_llm_experiment_retry14_baseline/paper_results/new_serverless_baselines_remote_v1/
```

Validated outputs:

- Llama-2 7B clean queue `20260518_serverlessllm_new_remote_v1_clean7b`:
  `4000/4000`, no `trace_expected` fallback, TTFT avg `237136.08 ms`, service
  TTFT avg `408.59 ms`, CE `1.5581`.
- Llama-3.2 3B queue `20260518_serverlessllm_new_remote_v1`:
  `4000/4000`, no `trace_expected` fallback, TTFT avg `237811.23 ms`, service
  TTFT avg `498.50 ms`, CE `1.8560`.

Treat it as a `ServerlessLLM-new` candidate row. Do not silently replace the
old `ServerlessLLM` row or regenerate default `figs/` unless the paper data
policy is explicitly changed.

## 5. Important Experiment Decisions

### a500 is the default main workload

The 500-adapter point is the default/main experiment, not an extra sensitivity
point. Do not rerun `a500` inside adapter-pool sensitivity unless a data
integrity audit proves the canonical main point is invalid.

Adapter-pool sensitivity should use:

```text
a100, a200, a300, a400 from the sensitivity queue
a500 from the canonical main round
```

The true-remote sensitivity builder uses an explicit PrimeLoRA summary override
for the canonical `a500` main round because that round's baseline compare JSON
does not contain a `faaslora` row. This behavior is implemented in:

```text
scripts/plot_paper_sensitivity.py
scripts/run_true_remote_full_figures_queue.sh
```

### Do not overwrite closed-loop figures

The original closed-loop paper figures live under `figs/`. When producing
remote variants or new diagnostic variants, write to a separate mirror
directory such as `figs_remote_full_real_remote_v1/`. If a script accidentally
updates old `figs/` files while generating a mirror, restore those old files
before committing unless the user explicitly asks to replace the paper
baseline.

### True-remote endpoints

The true-remote artifact HTTP services used during the mirror run were:

```text
Llama-2 7B:    http://192.168.4.174:18081
Llama-2 13B:   http://192.168.4.174:18082
Llama-3.2 3B:  http://192.168.4.174:18080
```

Do not commit credentials. If remote service restart or SSH access is needed,
ask the user to provide credentials in the live session.

### Remote artifact interpretation

The original closed-loop dataset remains valid as the primary paper snapshot.
The true-remote mirror showed broadly consistent trends and therefore mainly
serves as realism/sanity evidence. Do not replace the primary paper snapshot
automatically.

## 6. Baseline And System-Specific Notes

ServerlessLLM:

- Current result is intentionally kept as a general serverless LLM baseline,
  not a multi-LoRA adapter-readiness baseline.
- Its very high end-to-end TTFT is dominated by dispatch/admission/scale-out
  waiting, while service TTFT and TPOT stay in ordinary backend ranges.
- If adding newer serverless LLM baselines, do not delete ServerlessLLM without
  a paper-level decision. Add candidates through a reproducibility gate first.

S-LoRA:

- Llama-2 7B path uses DP4/TP1 and the normal packed-BGMV path:
  `bmm=0 (requested=auto, reason=packed_bgmv)`.
- The previous 13B BMM work was diagnostic and should not be copied into 7B or
  3B unless a concrete compatibility issue requires it.

PrimeLoRA-SGLang:

- Implemented as a backend portability extension, not a replacement for the
  main vLLM-based PrimeLoRA evaluation.
- Current paper positioning: PrimeLoRA is a backend-portable serverless control
  plane. The main evaluation uses vLLM, and backend sensitivity shows that the
  control-plane gain persists with SGLang.

## 7. If The Next Task Is New Baseline Survey

For new Serverless+LLM inference systems, use this gate before any code work:

1. Confirm the paper is actually serverless + LLM inference, not only
   autoscaling, serving, or generic GPU scheduling.
2. Confirm source code or artifact is public and buildable.
3. Confirm it can run or be fairly adapted on the local testbed:
   four RTX 3090 GPUs, 125 GB host memory, Python/vLLM/SGLang environments.
4. Confirm it can consume the shared trace and adapter pool, or explain exactly
   what comparable workload will be used.
5. Confirm metrics can be mapped to `e2e_v3` without inventing fields.
6. Prefer adding it as an additional diagnostic/appendix baseline before
   changing the main paper table.

Do not start long experiments for a new candidate until this gate is written
down and the user approves the candidate.

## 8. Safe Startup Prompt For The Next Session

Use the prompt below in a fresh Codex session.

```text
你是 Codex，在 /home/qhq 上继续 PrimeLoRA/FaaSLoRA 项目。请先阅读：

1. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/SESSION_HANDOFF_2026-05-18.md
2. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/CODEX_INTERACTION_RULES.md
3. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/DOCUMENTATION_INDEX.md
4. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/PROJECT_PROGRESS.md
5. /home/qhq/serverless_llm_experiment_retry14_baseline/docs/对比实验日志.md 的末尾

然后检查：

cd /home/qhq/serverless_llm_experiment_retry14_baseline
git status --short --branch
git log --oneline -5
tmux ls || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits

当前默认结论：
- 主实验和论文图表已闭环。
- 默认论文数据是 figs/ 和 paper_results/final_v2/。
- 真实 remote 镜像是 figs_remote_full_real_remote_v1/ 和 paper_results/final_remote_full_real_remote_v1/。
- a500 是默认主实验点，不要在 adapter-pool sensitivity 中重跑；a100-a400 才是额外 sensitivity 点。
- 不要覆盖 figs/，除非我明确要求替换主论文图。
- 不要提交 configs/generated/lora_manifest_1000.json，除非我明确要求。
- 不要提交任何远端密码或凭据。

如果我要调研新的 Serverless+LLM inference baseline，请先做候选系统复现可行性表：
论文/系统名、年份/会议、是否开源、代码地址、是否支持 LLM inference、是否支持 LoRA/adapter、是否能映射 e2e_v3、适配代价、是否推荐作为主表/附录/不采用。
不要直接跑长实验，先给候选排序和理由。
```
