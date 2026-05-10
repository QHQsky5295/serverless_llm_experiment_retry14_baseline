# PrimeLoRA Final Paper Data Snapshot V2

Snapshot date: 2026-05-10, Asia/Shanghai.

This directory contains the final paper-facing experiment data that should be
kept with the `retry14_continuous_queue_v2` branch. It is intentionally small
and curated: old debug rounds, failed rounds, 13B exploratory results, 1B smoke
results, and Qwen bring-up data are not included.

## Scope

Included data:

- Main comparison on Llama-2 7B and Llama-3.2 3B.
- Diagnostic TTFT decomposition for the same main comparison.
- Measured backend-sensitivity comparison for vLLM, PrimeLoRA-vLLM, SGLang, and
  PrimeLoRA-SGLang.
- Lifecycle-cost figure data for the main comparison and backend-sensitivity
  audit.
- Compressed raw JSON summaries for the final source runs.

All final rows use 4000 requests, a 500-adapter pool, Zipf exponent 1.0, hot-set
rotation every 500 requests, and the s8 replay scale.

## Files

- `source_manifest.json`: source paths, compressed snapshot paths, sizes, and
  SHA256 hashes.
- `source_manifest.csv`: the same source manifest in CSV form.
- `raw_json_gz/`: compressed copies of the final raw JSON summaries.
- `tables/`: final LaTeX tables, CSV tables, and table manifests.
- `figure_data/`: CSV/manifests for final lifecycle figures.

The publication figures themselves remain under `figs/paper/...`, while this
directory is the recoverable data snapshot.

## Restore Example

```bash
gzip -dk paper_results/final_v2/raw_json_gz/main_llama2_7b_primelora_vllm.json.gz
```

Use `source_manifest.json` to map each compressed file back to its original
source run.
