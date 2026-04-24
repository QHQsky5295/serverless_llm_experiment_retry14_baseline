# ServerlessLLM Baseline Project

This directory is the project entry point for the official ServerlessLLM
baseline reproduction.

## Current Status

- Upstream source: `/home/qhq/serverless_llm_baselines/repos/ServerlessLLM`
- Project symlink: `repo`
- Head env: `sllm_head_official`
- Worker env: `sllm_worker_official`
- Current formal backend: ServerlessLLM with vLLM backend when correctness
  probing succeeds.
- Local reproducibility patch:
  `/home/qhq/serverless_llm_baselines/patches/ServerlessLLM_local_changes.patch`

## Current Rules

1. Use shared trace and shared adapter subset from the active round.
2. Generate deploy config from the authoritative FaaSLoRA profile.
3. Preserve ServerlessLLM's serverless design; do not add PrimeLoRA mechanisms.
4. Runtime fixes are allowed only for serving correctness, environment
   isolation, metric instrumentation, and fair profile propagation.
5. vLLM runtime env must be explicit in deploy config and propagated into tmux
   launched processes.
6. No silent transformers fallback in formal runs.

## Entry Points

Formal full-round entry:

```bash
/home/qhq/serverless_llm_baselines/scripts/run_full_fair_round.sh
```

ServerlessLLM-only debug wrapper:

```bash
/home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_fair_experiment.sh
```

Stack utilities:

```bash
/home/qhq/serverless_llm_baselines/scripts/start_serverlessllm_stack.sh
/home/qhq/serverless_llm_baselines/scripts/stop_serverlessllm_stack.sh
```
