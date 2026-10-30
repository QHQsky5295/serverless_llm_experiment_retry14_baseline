# Re-creatable cache cleanup after D968 disk guard

The inference-node free-space gate stopped D968 at approximately 148.7 GiB.
After preserving D968's launch audit and partial logs, its re-creatable NVMe
adapter copies were removed. To restore the margin needed for the next PrimeLoRA
full replay, the following historical NVMe-only caches are also eligible for
removal. These directories contain materialized cache copies, not the source
trace, adapter manifest, result JSON, or mechanism logs; the corresponding
historical result directories remain in place.

| historical run | cache | reason |
|---|---|---|
| `d115_20260929` | `3b_nvme_full10` | old development run; cache is reproducible |
| `d211_20261004_resident_ref1` | `7b_nvme_resident_ref1.attempt2_semantic_invalid` | explicitly semantic-invalid attempt |
| `d920_20261027` | `7b_nvme_adaptive_guard_w2` | superseded development cache |
| `d937_20261030` | `3b_full_w2_nvme_extent` | cache copy only; result/log evidence retained |
| `d634_20261017` | `7b_nvme_min2_w0` | superseded development cache |

The cleanup is limited to these re-creatable cache directories. It does not
alter tracked code, paper sources, result JSON, logs, or the frozen artifact
pool. The exact paths and pre-cleanup sizes are recorded here before deletion.

