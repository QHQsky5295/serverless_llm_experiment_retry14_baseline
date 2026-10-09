# D377 7B capacity-revalidation diagnostic: remote-service refusal

Date: 2026-10-10

D377 passed profile qualification and began the 100-request replay. The
artifact node was not active at launch, so the first remote fetch failed with
`Connection refused` for `finance_lora`. The queue then surfaced the recorded
failure as `failed movement requires explicit owner recovery, not an automatic
retry`, and the run stopped after two terminal failures. The retained request
records contain `RemoteArtifactError` and `RuntimeError`; no successful request
set exists.

The launcher classified D377 as `protocol_or_launcher_error`. GPU context,
service scope and remote state were cleaned up. D377 is not a PrimeLoRA
performance result and is excluded from all metrics, rankings and figures.

This is an experiment-environment failure, not evidence of a controller or
capacity defect. The remote 7B service was subsequently started through its
existing user service, its `/health` endpoint returned `ok=true`, and the
current 1000-Mb/s link was unchanged. A new run identity is required; the
remote service must remain active for the run and be stopped only after its
completion receipt is captured.
