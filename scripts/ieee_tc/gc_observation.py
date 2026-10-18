"""Bounded GC timing observer for isolated qualification, not serving policy.

No collection, freeze/unfreeze, threshold changes or GC disable. Unlike the
native verbose debugger this does not enumerate objects during a collection.
"""
import gc
import threading
import time


class GCObservation:
    def __init__(self, limit=10000):
        self.limit = limit
        self.events = []
        self.errors = []
        self.pending = None
        self.label = ('setup', -1, 'neither')
        self.callback = self._callback
        self.active = False

    @staticmethod
    def state():
        return dict(enabled=gc.isenabled(), threshold=list(gc.get_threshold()),
                    count=list(gc.get_count()), freeze_count=gc.get_freeze_count(),
                    stats=gc.get_stats())

    def mark(self, stage, index, variant):
        self.label = (stage, index, variant)

    def _callback(self, phase, info):
        stamp, cpu = time.perf_counter_ns(), time.thread_time_ns()
        try:
            if phase == 'start':
                if self.pending is not None:
                    raise RuntimeError('nested GC start')
                self.pending = (stamp, cpu, info['generation'], self.label, threading.get_ident())
            elif phase == 'stop':
                if self.pending is None:
                    raise RuntimeError('unpaired GC stop')
                start, start_cpu, generation, label, thread = self.pending
                self.pending = None
                if info['generation'] != generation or threading.get_ident() != thread:
                    raise RuntimeError('GC generation/thread changed within collection')
                if len(self.events) >= self.limit:
                    raise RuntimeError('GC observation truncated')
                self.events.append(dict(begin_ns=start, end_ns=stamp, wall_ns=stamp-start,
                    thread_cpu_ns=cpu-start_cpu, generation=generation,
                    stage=label[0], index=label[1], variant=label[2], thread=thread,
                    collected=info['collected'], uncollectable=info['uncollectable']))
            else:
                raise RuntimeError('unknown GC phase')
        except Exception as error:
            # Exceptions escaping gc.callbacks are otherwise printed and ignored.
            self.errors.append(str(error))

    def __enter__(self):
        if self.active:
            raise RuntimeError('GC observation already active')
        self.before = self.state()
        gc.callbacks.append(self.callback)
        self.active = True
        return self

    def __exit__(self, *exc):
        gc.callbacks.remove(self.callback)
        self.active = False
        self.after = self.state()

    def report(self):
        if self.active or self.pending is not None or self.errors:
            raise RuntimeError(f'incomplete GC observation: {self.errors}')
        for key in ('enabled', 'threshold', 'freeze_count'):
            if self.before[key] != self.after[key]:
                raise RuntimeError(f'GC policy changed during isolated observation: {key}')
        return dict(before=self.before, after=self.after, events=self.events,
                    gc_policy_unchanged=True, observer_can_shift_collection_timing=True,
                    historical_d685_pause_cause_not_retroactively_proven=True)
