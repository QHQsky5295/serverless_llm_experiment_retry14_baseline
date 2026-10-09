# D383 — artifact-node gate recheck and 7B correctness decision boundary

Date: 2026-10-10. This is a read-only gate and evidence interpretation; no
GPU process, inference service, remote artifact service, pack, cleanup, or
experiment was started.

## Artifact-node gate

The approved artifact-only rule is

\[
F_{required}=S+\lceil 1.5(P_{pack}+G)\rceil,
\]

with the already qualified archive bounds, `G=67,108,864` bytes of remaining
log growth, and `S=17,179,869,184` bytes. The required free space is
`97,407,250,432` bytes. A fresh read-only SSH check found:

| observation | value |
|---|---:|
| filesystem free bytes | 139,184,943,104 |
| free inodes | 30,883,806 |
| required free bytes | 97,407,250,432 |
| margin over requirement | 41,777,692,672 |
| 7B published cache | mode 555, manifest mode 444 |
| listeners on 18080/18081 | none |
| filesystem | ext4 |

The system has no `quota` command; read-only mount inspection identifies ext4,
and `xfs_quota` correctly refuses it as a foreign filesystem. This is recorded
as quota not configured/applicable, not silently as a successful quota audit.
The bounded published-cache protocol therefore passes its measured disk gate;
request-time packing or cache-publication changes require a new bound and
filesystem/quota evidence.

The inference host remains subject to its unchanged 150-GiB start floor,
100-GiB running stop floor, memory equation, swap/GPU/OOM protection, and
unique-artifact preservation. The remote service is not left running between
runs.

## Correctness decision boundary

The 7B materialized content index contains 500 logical IDs but only four
distinct `(adapter_model.safetensors, adapter_config.json)` weight/config
classes with counts 348, 150, 1 and 1. D166/D167 independently verify native
slot contents for a bounded 12-ID/4-content-class sample; D236 verifies first
position PEFT agreement for five controls; D247 retry6 matches 19/22 complete
sequences and records three first divergences as FP16 near ties. These are
diagnostic evidence, not a full-workload `N_correct` proof. D379 likewise has
complete native token-count/timing contracts but `n_correct=null`.

The next admissible Prime-first action is not a repeated 22-request probe and
not a 3B/baseline run. First audit whether a pre-registered content-class plus
request-binding witness can satisfy the frozen correctness contract without
changing the strict exact-token rule. If it cannot, register one bounded
diagnostic that captures native token IDs for existing 7B trace requests
spanning all four content classes and relevant input-length groups, then
compare against the unchanged independent reference. Near ties remain warnings,
not silently accepted matches. No performance result or G1/G2 denominator is
created by this diagnostic.

No figure is generated from D383. Raw historical evidence, source traces,
adapter pools, published caches, old figures and the user-owned manifest remain
unchanged.
