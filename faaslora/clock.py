"""Dependency-free identity for local monotonic experiment timestamps."""
from functools import lru_cache
import os
from pathlib import Path
import time


@lru_cache(maxsize=1)
def local_monotonic_clock_id() -> str:
    mono, perf = time.get_clock_info('monotonic'), time.get_clock_info('perf_counter')
    if not mono.monotonic or not perf.monotonic or mono.implementation != perf.implementation:
        raise RuntimeError('IEEE timing requires one verified monotonic/perf clock')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    namespace = os.readlink('/proc/self/ns/time')
    if not boot or not namespace:
        raise RuntimeError('cannot establish local clock identity')
    return f'linux-monotonic:{boot}:{namespace}'
