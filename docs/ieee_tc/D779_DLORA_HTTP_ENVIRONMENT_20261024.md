# D779: dLoRA common HTTP environment correction

The first actual shared-service qualification used the retained 3B DP2/TP1,
migration_type=3 configuration and D778 native measurement source view. It
prepared all 500 existing remote artifacts, then the native Python process
failed at `from aiohttp import web` before Ray/model initialization:
`ModuleNotFoundError: No module named 'aiohttp'`.

This is an integration environment failure, not evidence of dLoRA algorithmic
inferiority. No model-ready event or native inference measurement exists. The
failed attempt is retained; it is not a valid TTFT/TPOT/GPU-s comparison point.
The acquisition record has no qualified worker lifetime, so its incomplete
physical summary is not converted into a fabricated zero-cost result.

## Necessary correction, not backend optimization

Reused D441's existing eight HTTP-only CPython3.9 wheel pins, now referenced by
`configs/ieee_tc/dlora_http_py39_additions.txt`. The direct lock SHA is
`8f13659abb2f95a10e2ff57f148dd43f06edef97a8588f7d0f8612149cc515e8`.
The native environment is
`/home/qhq/anaconda3/envs/dlora_medusa_clone_20260519`.

The common outer environment already uses aiohttp3.13.3. Its [official version
metadata](https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/pyproject.toml)
supports Python>=3.9. A dry run preceded installation with `--no-deps`,
`--only-binary=:all:`, `--require-hashes`, and `--no-compile`.

Only previously absent distributions were added:

| Distribution | Version |
|---|---|
| aiohttp | 3.13.3 |
| async-timeout | 5.0.1 |
| multidict | 6.7.1 |
| yarl | 1.22.0 |
| aiohappyeyeballs | 2.6.1 |
| aiosignal | 1.4.0 |
| frozenlist | 1.8.0 |
| propcache | 0.4.1 |

All previously installed distribution versions are unchanged. In particular,
PyTorch2.1.2+cu121, Ray2.51.2, Transformers, native vLLM and Triton were not
upgraded. No native scheduling, placement, migration, LoRA arithmetic or
performance configuration was changed.

The actual Python3.9.25 environment imported the D778 EngineManager and shared
handler, served an aiohttp loopback health request, and closed the endpoint.
CUDA and Ray remained uninitialized. This validates the repaired dependency
path, not real model execution or independent numerical correctness.

## Evidence and continuation

Raw evidence:
`/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d779_20261024`.
It includes before/after environment inventories, hash-checked pip resolution
and install reports, actual import/HTTP output, failed worker log, frozen inputs,
remote transfer receipts, physical incompleteness and actual cleanup receipts.

Before inventory SHA:
`5f0628c3b770cc05edb7cc9ae2134ea01f09e0c9ae21f22e6270cb15ec10719d`.
After inventory SHA:
`08982afdd11915586b15a1ed0e04d7a9c7aa89ae8f87011723cfb20ec7ff8b93`.

Service peak was 20,533,035,008 bytes, with high/max/OOM/swap events zero.
The failed service and auxiliary domains were closed; the owned remote services
were stopped. After comparing all 4,000 cached files against the retained pool
and preserving resolver evidence, only duplicate objects were removed:
19,482,573,296 payload bytes; observed free-space increase 19,497,287,680 bytes.
Original models, adapters, traces, failed evidence and user changes remain.

Next: bind this environment identity to a new actual-load qualification key,
reuse D778 source view and unchanged DP2/TP1 parameters, then the planned 3B
W1/W2 comparison. Do not rerun the failed key or call this CPU repair a successful
performance replication. The unrelated shared k3s remains untouched.
