# True-Remote Full Figure Snapshot

- queue_id: 20260514_real_remote_fullfigs_v1
- figs_root: /home/qhq/serverless_llm_experiment_retry14_baseline/figs_remote_full_real_remote_v1
- paper_results: /home/qhq/serverless_llm_experiment_retry14_baseline/paper_results/final_remote_full_real_remote_v1
- generated_at: 2026-05-18 17:31:49 CST

This snapshot is non-overwriting and uses real HTTP remote artifact endpoints for every system in the newly run figure experiments.

Post-close alignment on 2026-05-22 regenerated and mirrored the paper-facing
single-round `fig1_intro_teaser.*` and `fig5_main_normalized.*` artifacts under
`figs/paper/main/`. The checksum manifests were regenerated without including
the `SHA256SUMS` files themselves, so `sha256sum -c SHA256SUMS` validates the
snapshot directly.
