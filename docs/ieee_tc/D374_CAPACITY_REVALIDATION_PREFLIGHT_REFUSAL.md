# D374 7B capacity-revalidation diagnostic: preflight refusal

Date: 2026-10-10

## Scope

D374 was a bounded 100-request seed-42 development diagnostic for the
capacity-derived `C=8` controller envelope. It was intended to test whether the
post-D371 source-identity fixes could retain the lower GPU-retention envelope.
It was not a formal G1/G2 run and was not eligible for comparison or figure
generation.

The configuration and launcher were committed at `c29a637`. The requested
output identity was
`results/ieee_tc/p2_backend_qualification/d374_20261010/7b_capacity_revalidation_100/`.

## Gate result

The launcher stopped at the unchanged physical-startup-memory gate before
loading a model or starting any service:

| quantity | value |
|---|---:|
| required available memory | 109,521,666,048 bytes |
| observed available memory | 108,639,059,968 bytes |
| shortfall | 882,606,080 bytes |
| classification | `insufficient_physical_startup_headroom` |

The `launch.json` receipt is retained at the requested output path. Its SHA-256
is `667b59f7be88ba7b76f924629570ba3b45016a66ea2d4875719964308ddb0001`.

No local GPU context, model server, remote artifact service, request replay,
or request-time artifact operation was started. The owned auxiliary scope used
for the preflight command was stopped after the refusal and reached zero tasks;
the remote units remained inactive and ports 18080/18081 remained absent.

## Interpretation and follow-up

This is a safety/protocol refusal, not a performance failure and not evidence
against the `C=8` envelope. A new run identity is required after the unchanged
memory gate passes; D374 must not be resumed in place or entered into a result
table.

During post-refusal memory inspection, two stale broad `rg` searches started by
the audit itself were found to consume substantial memory. They were terminated
by exact PID, and available memory subsequently rose above the 102-GiB startup
requirement. These searches were external to D374 and are not part of any
experiment or performance accounting.

The inference-host memory and disk policies were not relaxed. The next launch
must repeat the complete preflight, preserve the same trace/subset/generation
contract, and use a fresh campaign identity.
