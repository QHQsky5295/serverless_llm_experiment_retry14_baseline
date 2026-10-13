"""Physical union of actual native container reservations, including co-location.

This observer never chooses a GPU or constrains HydraServe's placement. Reuse
PhysicalGPULedger's union integration, but do not use an exclusive runtime lock
which would silently forbid the original half-memory/consolidation placement.
Native worker process/cgroup termination and fresh NVML census are mandatory.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import select
import time
import uuid


class NativeWorkerLedger:
    def __init__(self, *, root, context, census, service_path, process_identity):
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.metrics.metrics_collector import PhysicalGPULedger
        if context['clock_id'] != local_monotonic_clock_id():
            raise ValueError('native lifecycle clock differs')
        self.root=Path(root);self.root.mkdir()
        self.path=self.root/'container_allocations.jsonl'
        self.log=self.path.open('x')
        self.context=context
        self.census,self.service_path,self.identity=census,Path(service_path),process_identity
        self.ledger=PhysicalGPULedger(clock_id=context['clock_id'],
            deployment_notice_s=context['deployment_notice_s'])
        self.owners={}
        self.sequence=0

    def emit(self, event, worker, **fields):
        self.sequence+=1
        row=dict(event=event,worker_name=worker.name,container_name=worker.container_name,
            gpu_uuid=worker.uuid,at=time.perf_counter(),clock_id=self.context['clock_id'],**fields)
        self.log.write(json.dumps(row,sort_keys=True)+'\n');self.log.flush()
        os.fsync(self.log.fileno())
        return row,str(self.path)+':'+str(self.sequence)

    def before_create(self, worker):
        if worker.name in self.owners or not worker.uuid:
            raise ValueError('duplicate or CPU-only native GPU reservation')
        lease=uuid.uuid4().hex
        row,evidence=self.emit('acquire',worker,lease_id=lease)
        self.ledger.acquire(lease_id=lease,owner_id=lease,at=row['at'],
            gpu_uuids=[worker.uuid],clock_id=row['clock_id'],evidence_id=evidence)
        self.owners[worker.name]=dict(lease=lease,fd=None,birth=None,confirmed=False,released=False)

    def after_create(self, worker):
        from faaslora.metrics.metrics_collector import _open_pidfd
        record=self.owners[worker.name]
        if record['fd'] is not None or not worker.owner:
            raise ValueError('missing/duplicate native container owner')
        fd=_open_pidfd(worker.owner['pid'])
        if self.identity(worker.owner['pid']) != worker.owner:
            os.close(fd)
            raise ValueError('native worker changed during pidfd acquisition')
        record.update(fd=fd,birth=worker.owner)
        self.emit('worker_spawn',worker,lease_id=record['lease'],owner=worker.owner,container_id=worker.cid)

    def confirm(self, worker):
        record=self.owners[worker.name]
        if record['confirmed']:
            return
        sample=self.census.sample(self.service_path)
        found=[p for d in sample['devices'] if d['gpu_uuid']==worker.uuid
            for p in d['processes'] if p['kind']=='compute' and p['service_member']
            and p['pid']==worker.owner['pid'] and p['identity']==worker.owner]
        if not found:
            raise ValueError('native container has no confirmed GPU context')
        self.emit('native_context',worker,lease_id=record['lease'],census=sample)
        record['confirmed']=True

    def before_delete(self, worker):
        # Measurement failure must not block exact-owned safety cleanup. Retain
        # missing proof; final qualification then fails instead of fabricating it.
        try:
            if worker.owner and self.owners[worker.name]['fd'] is None:
                self.after_create(worker)
            self.confirm(worker)
        except Exception as exc:
            self.emit('context_unqualified',worker,error=type(exc).__name__+': '+str(exc))

    def after_delete(self, worker):
        record=self.owners[worker.name]
        if record['released']:
            raise ValueError('duplicate physical return')
        birth=record['birth']
        if birth and (record['fd'] is None or not select.select([record['fd']],[],[],0)[0]):
            raise ValueError('native process not kernel-confirmed exited')
        sample=self.census.sample(self.service_path)
        # Same-card processes belonging to OTHER native leases remain charged
        # by the union; do not require them to exit or bill the card twice.
        for d in sample['devices']:
            if d['gpu_uuid'] != worker.uuid:
                continue
            for p in d['processes']:
                if p['kind']=='compute' and (p['identity'] is None or
                        (birth and p['pid']==birth['pid'] and p['identity']['start_ticks']==birth['start_ticks'])):
                    raise ValueError('owned/unknown GPU context remains at return')
        row,evidence=self.emit('release',worker,lease_id=record['lease'],census=sample,
            native_context_qualified=record['confirmed'])
        self.ledger.release(lease_id=record['lease'],owner_id=record['lease'],at=row['at'],
            clock_id=row['clock_id'],evidence_id=evidence)
        record['released']=True
        if record['fd'] is not None:
            os.close(record['fd']);record['fd']=None

    def finalize(self, *, arrival_end_s, last_terminal_s, n_plan, n_terminal):
        report=self.ledger.summarize(observed_until_s=time.perf_counter(),
            arrival_start_s=self.context['replay_t0_s'],arrival_end_s=arrival_end_s,
            last_terminal_s=last_terminal_s,n_plan=n_plan,n_terminal=n_terminal,n_correct=0)
        self.log.close()
        report.update(contract='hydraserve_container_union_v1',
            n_correct=None,gpu_seconds_per_correct_request=None,
            numerical_correctness_qualified=False,
            native_contexts_qualified=bool(self.owners) and all(r['confirmed'] for r in self.owners.values()),
            allocation_journal=str(self.path),allocation_journal_sha256=sha256(self.path.read_bytes()).hexdigest())
        with (self.root/'summary.json').open('x') as handle:json.dump(report,handle,indent=2)
        self.census.close()
        return report
