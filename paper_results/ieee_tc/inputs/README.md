# Existing-pool content indices — D62, 2026-09-27

These small metadata files index existing assets; no weights or traces were
created, copied or rewritten. They are NOT latency profiles or serving results.

## Candidate contracts for the observed remote ordinary-file layout

| Model | Use this index | IDs / files | Logical bytes | Exact file-tree classes |
|---|---|---:|---:|---:|
| 3B | `20260927_3b_remote_content_index.json` | 500 / 4000 | 19482573296 | 24 |
| 7B | `20260927_7b_materialized_content_index.json` | 500 / 5000 | 12985984450 | 6 |

Both use `regular_files_materialized_support_v1`: hash existing explicitly
allowed local support targets under the relative names that are ordinary files
in the remote pool. Original local symlinks and both artifact pools are unchanged.
The existing HTTP consumer validates every local PEFT metadata/routing identity.
Remote file counts/types/bytes were inspected read-only, but full remote content
and actual HTTP downloads remain unqualified. Each JSON records
`remote_content_verified=false` and `serving_qualified=false`.

File SHA256:

- 3B: `bd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275`
- 7B: `e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c`

Canonical HTTP contract SHA256 (distinct from the whole JSON SHA):

- 3B: `ac8e9b36c328376a9e1ecec6a4d928dd684f4d94e3fa4d2e08f531b166da145e`
- 7B: `684c7ab113b6a51b694066753b340fce4eb0b26f565e1f9b7ebbd97d3b8ea050`

Common indexer SHA256:
`99ccb362cbb5640c2dfb27b5746f96ab08bfb578f4eb9d018ecfb1da105be700`.
Original completed tensor-audit SHA256:
`224b9337ecaba2590ac888e9d0e3b9eb41a17074ce77b952e98e935259adf60f`.

## Preserved local-pack diagnostics — do not use as remote contracts

| File | Excluded links / bytes | SHA256 |
|---|---:|---|
| `20260927_3b_content_index.json` | 2500 / 4570774000 | `ba57a73918851669c28af0e383df9cacb38d2217fbbbaa9d8dd16662c00a6d16` |
| `20260927_7b_remote_content_index.json` | 2 / 797 | `f53c8e4e71e189ff064aa76cb4d133900d62cee43f71b820cd28d27c90fc8bc3` |

The earlier 7B filename is misleading: its provenance says skip mode and it
contains only4998 files. Preserve the original rather than overwrite/relabel it.
These indices match local server link selection, not the remote materialized
directory layout. Never choose an input by wildcard/latest filename alone.

See `docs/ieee_tc/ARTIFACT_CONTENT_AUDIT.md` for the failed first attempt,
symlink discoveries, tests, resource receipts, limits and interpretation.
All-zero weight and independent-correctness limitations from the prior audit
remain unchanged. Content classes are not counts of independently trained LoRAs.

2026-09-27 D76: the unchanged 3B index now has separate actual remote coverage
evidence: `../remote_qualification/20260927_d76_3b_coverage.json`,500/500 IDs and
4000/4000 files verified. Do not mutate the original index's historical
`remote_content_verified=false` field and invalidate its frozen SHA. Join the
new evidence by that SHA.7B full coverage and inference qualification remain open.

2026-09-27 D77: the correct unchanged 7B materialized index also has separate
remote evidence: `../remote_qualification/20260927_d77_7b_coverage.json`,500/500
IDs and5000/5000 files verified. Join by index SHA e85cce3c...; do not rewrite
historical index fields. Both full-pool content gates pass. Inference, numerical
LoRA application and representative performance profiles are separate gates.
