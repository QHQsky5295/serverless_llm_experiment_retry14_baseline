# ServerlessLLM-New True-Remote Formal V1

## Inputs

- Source true-remote 7B round:
  `results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_012813_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1`
- Source true-remote 3B round:
  `results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_160342_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1`
- Shared workload: 4,000 requests, 500 adapters, hot48, rot500, s8.
- Remote adapter endpoints:
  - 7B: `http://192.168.4.174:18081`
  - 3B: `http://192.168.4.174:18080`

## Outputs

- 7B clean queue id: `20260518_serverlessllm_new_remote_v1_clean7b`
- 3B clean queue id: `20260518_serverlessllm_new_remote_v1`
- Result root:
  `results/paper_experiments/15_new_serverless_baselines_remote_v1/`
- Log root:
  `results/logs/new_serverless_baselines_remote_v1/serverlessllm_new/formal/`

## Validation

Both clean replays passed:

```bash
python scripts/validate_replay_results.py --system ServerlessLLM-new \
  --replay <replay.json> --expected-total 4000
```

The validator reported `ok=4000 total=4000` and no `trace_expected` fallback
for both backbones.

## Inclusion Boundary

This experiment is a formal candidate row named `ServerlessLLM-new`. It should
not overwrite the old `ServerlessLLM` code path, result directories, figures,
or curated paper data. The FaaSLoRA repository owns the Git-tracked compact
bundle and any future table/figure integration decision.
