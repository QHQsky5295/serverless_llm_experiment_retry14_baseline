# D376 7B capacity-revalidation diagnostic

D376 is a fresh-identity, bounded 100-request seed-42 diagnostic that tests
the C=8 controller envelope using the already-qualified C=8 D321/D298 service
and preparation profiles. The current source-identity code remains in the
working tree; the two later opt-in policy fields are intentionally absent so
the requested model identity exactly matches the measured C=8 profiles.

This is the scientifically valid capacity-isolation path identified after the
D375 qualification failure. It does not bypass profile checks and does not
claim to evaluate the unqualified combination of C=8 with both later policy
fields. It is not a formal validation selection, G1/G2 result, baseline
comparison, or figure source. The same frozen trace, adapter subset, generation
contract, D371 source code, controller values, and read-only remote cache are
used; only campaign/output/cache identity and profile-aligned configuration
differ.

Before launch, repeat all plan gates. After closure, require 100/100 terminal
records, native fixed-length contract evidence, GPU-context release, empty GPU
census, zero owned scopes and inactive remote units. A failed qualification or
incomplete replay is retained and excluded; no threshold or profile hash may
be relaxed.
