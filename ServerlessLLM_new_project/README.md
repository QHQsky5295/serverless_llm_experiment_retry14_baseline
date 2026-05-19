# ServerlessLLM-New Baseline Project

This directory is the project entry point for the 2026-05-18 reproduction of
the current upstream ServerlessLLM main branch. It is intentionally separate
from `ServerlessLLM_project/` so the historical ServerlessLLM baseline, code,
and data are not overwritten.

## Current Status

- Upstream source: `/home/qhq/serverless_llm_baselines/vendor_new_baselines/ServerlessLLM_new_main_20260518`
- Project symlink: `repo`
- Upstream commit: `9f50241baa5386e06a9321c51f19a9ef5f964c2b`
- Runtime env: `sllm_vllm0102_newserverless_20260518`
- Backend: ServerlessLLM with vLLM 0.10.2 worker path
- Formal scope: true-remote Llama-2 7B and Llama-3.2 3B, 4,000 requests,
  500 LoRA adapters, shared trace/subset from the closed true-remote rounds.

## Entry Points

Run the isolated formal queue:

```bash
/home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_new_remote_formal_queue.sh
```

Run only one backbone without touching the other output directory:

```bash
SLLM_NEW_ONLY=llama2_7b \
SLLM_NEW_QUEUE_ID=20260518_serverlessllm_new_remote_v1_clean7b \
bash /home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_new_remote_formal_queue.sh
```

## Output Boundary

The formal outputs live under:

```text
results/paper_experiments/15_new_serverless_baselines_remote_v1/
results/logs/new_serverless_baselines_remote_v1/serverlessllm_new/formal/
```

These directories are deliberately outside the old ServerlessLLM result paths.
The curated, Git-tracked paper bundle for this closed experiment is maintained
in the FaaSLoRA repository under:

```text
paper_results/new_serverless_baselines_remote_v1/
```

Do not replace `figs/` or `paper_results/final_v2/` with these results unless
the paper data policy is explicitly changed.
