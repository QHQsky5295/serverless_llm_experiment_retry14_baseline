# D944: ServerlessLLM formal-cache archive

The completed 7B ServerlessLLM formal run remains the authoritative result at:

`results/paper_experiments/15_new_serverless_baselines_remote_v1/20261030_formal_serverlessllm_new_7b_llama2_7b_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_serverlessllm_new/`

The unpacked `remote_cache/` directory was archived after the run and verified
before removal of the duplicate unpacked bytes.  The archive is:

`remote_cache.tar.zst`

- SHA-256: `616493489ef56959041c76b1b2f16f3d738ef5f515031d579a9661becf996c82`
- Archive size: 129 MiB
- Tar listing entries: 1454
- The raw replay JSONL and summary JSON were not modified.
- The archive preserves the exact unpacked cache and can be restored without
  changing the reported request-level metrics.

This was a storage-preserving operation only.  The formal result remains
unqualified for cross-system superiority because its HTTP fetcher recorded
`http_unthrottled` rather than the requested aggregate MiB/s limit.
