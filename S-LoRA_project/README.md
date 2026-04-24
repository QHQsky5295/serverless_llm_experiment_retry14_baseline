# S-LoRA Baseline Project

This directory is the project entry point for the official S-LoRA baseline.
S-LoRA is treated as a serverful multi-LoRA paper baseline, not a serverless
baseline.

## Current Status

- Upstream source: `/home/qhq/serverless_llm_baselines/repos/S-LoRA`
- Project symlink: `repo`
- Runtime env: `slora_official_cu118`
- Formal wrapper:
  `/home/qhq/serverless_llm_baselines/S-LoRA_project/scripts/run_slora_fair_experiment.sh`
- Integration layer: shared trace, shared adapter subset, native
  `/generate_stream`, `e2e_v3` replay/summary.

## Current Reproduction Boundary

Allowed:

- Environment isolation.
- Prompt budget alignment with S-LoRA server tokenizer semantics.
- Adapter path materialization from the shared subset.
- Replay and summary instrumentation.

Not allowed:

- Rewriting S-LoRA scheduling or paging mechanisms.
- Adding PrimeLoRA adapter-aware scale-out/preloading logic.
- Filling missing mechanism metrics with estimates.

## Formal Round

S-LoRA normally runs as one stage inside:

```bash
/home/qhq/serverless_llm_baselines/scripts/run_full_fair_round.sh
```

For interrupted rounds, use:

```bash
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh
```

## Notes

If S-LoRA fails, analyze the native server log and replay error samples. Do not
silently drop failed requests or generate an all-zero summary.
