# D418: declared native toolchain propagation

2026-10-11. D417 failed before ready: all four workers report missing `ninja`
in FlashInfer sampling warmup. The executable already exists in the declared
vLLM0.30.0 venv. D214retry6/7 in the main execution history had the same cause
and explicitly supplied PATH. D416 omitted that requirement in its new service
entry. No inference result, throughput limitation or remote-transfer failure
can be inferred from this attempt. Full failure evidence remains in D417.

The existing runner now derives bin from the **declared** interpreter path,
without resolving its symlink into a different base environment. It prepends
that bin to the worker PATH, removes inherited PYTHONPATH/PYTHONHOME as before,
and probes the installed ninja identity before any GPU allocation. Missing,
non-executable or foreign tools fail explicitly. No installation or backend
fallback occurs. Each worker reads back the actual PATH hash, tool SHA and
MAX_JOBS before importing the backend; ready receipts retain this identity.

The configuration explicitly freezes `build_max_jobs=2`, implementing the
plan's existing build safeguard through native MAX_JOBS. This is a new execution
identity, not a hidden performance setting. FlashInfer's per-module file lock
serializes simultaneous requests to build the same sampling module; MAX_JOBS
bounds one native ninja invocation, not an asserted global cross-framework
compiler count. Runtime initialization remains inside service/GPU lifecycle
measurement. No cache prebuild or engine preloading is introduced.

Evidence and references:

- D417 executed baseline5c0cfd8, whose worker environment lacked venv/bin.
- Installed flashinfer-python0.6.18.post1 `jit/cpp_ext.py` invokes `ninja` by
  name and uses MAX_JOBS for `-j`; `jit/core.py` holds a FileLock around builds.
- [Official FlashInfer v0.6.18 invocation](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/jit/cpp_ext.py)
  and [build locking](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/jit/core.py)
  confirm this contract; this tag is not claimed byte-identical to post1.
- [Python venv documentation](https://docs.python.org/3.12/library/venv.html)
  distinguishes explicitly invoking its interpreter from PATH activation.

CPU regression: 125 tests passed in7.756s under4GiB/zero-swap/CPUs2,3,26,27.
Three added checks cover symlinked venv Python, missing/foreign tools and invalid
build limits; all prior native measurement, routing and HTTP tests remain.
These are not GPU or model-correctness results. Follow with one new-key affected
100-request qualification, then reuse prior valid evidence toward full W0.
No changes to RR, caching, loading, sampling, public engine limits, PrimeLoRA's
nine equations, shared SLO or workload. D417 is not restarted or overwritten.

Actual declared venv CPU probe passed with GPUs hidden: ninja version
`1.13.2.git.kitware.jobserver-pipe-1`, binary SHA
`08639e194fffa7f08b259fc4abfa4803aff66b64de52549cee42ec527d55cea6`.
The installed FlashInfer `_get_num_workers`/`run_ninja` AST (source SHA
`c2ebe90fa1ddc3896553c51a4799e43ee8e2cd506740d3e3b1493280bfd7262b`)
produced `ninja ... -j 2` under the qualified environment. This command was
mocked; only `ninja --version` actually executed. No compilation/model/GPU load.
Recipe/receipt are in the exclusive D419 preparation root, not a performance run.
