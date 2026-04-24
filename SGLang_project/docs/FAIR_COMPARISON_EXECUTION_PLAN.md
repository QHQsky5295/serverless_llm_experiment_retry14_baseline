# SGLang Fair-Comparison Notes

SGLang is an active serverful many-LoRA serving baseline in the current
PrimeLoRA/FaaSLoRA comparison harness.

## Current Rules

- Use the shared trace and shared adapter subset from the active round.
- Use `metric_schema_version=e2e_v3`.
- Main fair topology: `DP=4, TP=1`.
- `TP=4` is reserved for an optional serverful upper-bound appendix, not the
  main fair table.
- Use native `/generate` with `input_ids`.
- Do not modify SGLang core serving logic.

## Current Entry

Use the full round runner:

```bash
/home/qhq/serverless_llm_baselines/scripts/run_full_fair_round.sh
```

Or system wrapper when debugging only SGLang:

```bash
/home/qhq/serverless_llm_baselines/SGLang_project/scripts/run_sglang_fair_experiment.sh
```

## Output

SGLang summaries must provide the same headline fields as all active systems:

- `TTFT_e2e avg/p95`
- `E2E_e2e avg/p95`
- `TPOT`
- `Throughput_tok_s`
- `Cost/req`
- `CE`

If a result lacks `e2e_v3`, it is historical and cannot enter the paper table.
