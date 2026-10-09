# D378 7B capacity-revalidation diagnostic

D378 is a fresh-identity retry of D377 after the remote 7B artifact service
was verified healthy. It keeps the corrected D321/D298 C=8 profile bindings,
the current source-identity code, the same seed-42 trace and adapter subset,
the fixed-length generation contract, and the unchanged C=8 controller
envelope. It is a bounded 100-request development diagnostic only.

The remote endpoint is required to be authenticated and healthy before launch
and must remain active until the completion receipt is captured. A remote
health check is setup evidence, not request data. Any endpoint refusal or
movement failure invalidates D378 and requires a new identity; no automatic
retry or fallback is allowed.
