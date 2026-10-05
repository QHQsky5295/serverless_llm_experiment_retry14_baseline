# D225 offline output-difference audit

This directory contains no new GPU run and no copied raw result.  The audit pairs
the immutable D210 and D219 request records by `request_id` and is summarized in
`docs/ieee_tc/D225_OUTPUT_DIFF_AND_NUMERICAL_GATE.md`.

| field | result |
|---|---:|
| paired requests | 4,000 |
| adapter/prompt/input/target mismatches | 0 |
| native token-contract failures | 0 |
| completion token-ID SHA mismatches | 136 (3.40%) |
| numerical ground truth available | no |
| formal `N_correct` upgrade | no |

The source JSON SHA values and the pre-registered next diagnostic are recorded in
the linked document. This is a correctness status table, not a performance figure.
