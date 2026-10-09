# D371 7B W0 identity-witness audit

This is an offline audit of the closed development replay; it is not a new
performance run and does not promote the result to G1/G2.

| Field | Value |
|---|---|
| Raw outcome | `results/ieee_tc/p2_backend_qualification/d371_20261010/7b_full_w0/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d371_7b_full_w0.json` |
| Raw outcome SHA-256 | `279ab57f704d4e680dfec5da22b64d1cf017d13809b4c6e60b15da0f8c9343ea` |
| Requests audited | 4,000 |
| Plan/protocol | `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7` / `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22` |
| Source code | `3ba847af857a07e378f16fd3ae44927a4dd8571f` |

For every request, the nested native witness was checked without inferring
semantic output correctness:

- request adapter ID equals the native intent `lora_name` (4,000/4,000);
- the intent's expected content SHA equals the selected source content SHA
  (4,000/4,000);
- native acquisition succeeded (4,000/4,000);
- the selected dispatch snapshot was confirmed (4,000/4,000);
- the output contract and requested token count matched (4,000/4,000).

The trace contains 132 logical adapter IDs and 132 adapter/content pairs, but
only six distinct content SHAs. Thus this is an identity and content-witness
audit, not evidence that the 500 logical IDs are 500 independently trained
weights. The selected source tiers were GPU 2,520, NVMe 856, HOST 491 and
remote 133.

Two requests (`req_00621`, `req_01662`) encountered a legitimate
`required_source_changed` transition. Each recorded exactly one stale
recheck and then completed with the refreshed source identity; no repeated
stale loop was observed in the nested request evidence. This supports the
narrow source-identity repair claim, but the analyzer does not expose a
standalone semantic `n_correct` field. Numerical adapter correctness and the
common warm/Resident references therefore remain open, and this audit cannot
be used for lifecycle ranking or an SLO qualification claim.
