"""Actual rootless Podman implementation of the D480 lifecycle-only port.

Reuses the D477/D481/D491 qualified containment/IPC/address recipe. The caller
must first establish the bounded parent, adopt its owned Podman pause, verify
the image/source/dependencies and run the external resource watchdog. No model
readiness, placement decision, resource share or Kubernetes object is invented.
"""
import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from .loopback_transport import address_environment


@dataclass(frozen=True)
class ContainerRecipe:
    """Resource and process recipe, not a service-lifetime policy.

    The native controller owns worker retirement; its caller's bounded campaign
    and independent watchdog own emergency cleanup. A request/init deadline must
    never become Podman's --timeout, which kills the entire long-lived container.
    Infrastructure CLI calls remain independently bounded in _call.
    """
    image_id: str
    native_images: Tuple[str, ...]
    entrypoint: Tuple[str, ...]
    readonly_mounts: Tuple[Tuple[str, str], ...]
    environment: Mapping[str, str]
    memory_bytes: int
    swap_bytes: int
    pids_max: int = 512
    tmp_bytes: int = 256 * 1024**2
    native_nas_path: str = '/mnt'


@dataclass
class OwnedWorker:
    name: str
    container_name: str
    address: str
    uuid: Optional[str]
    output: Path
    cid: Optional[str] = None
    owner: Optional[dict] = None
    child_group: Optional[Path] = None
    removed: bool = False
    lifecycle_lock: asyncio.Lock = field(default_factory=asyncio.Lock,
                                         repr=False, compare=False)


class IsolatedPodmanPlatform:
    """One real physical host; many local addresses do not create more nodes.

    process_identity returns the existing preflight gpu_process_identity schema.
    A failed create is retained for exact-owned cleanup, never silently retried.
    Registration is atomic; container operations serialize only for the same
    worker. An unrelated retiring worker must not block create or address lookup.
    Native model init is still performed/awaited by the original ResourceManager.
    """
    def __init__(self, *, campaign: str, output: Path, parent: Path,
                 parent_invocation: str, parent_inode: int,
                 parent_limits: Mapping[str, str], cpu_ids: Sequence[int],
                 podman_base: Sequence[str], podman_environment: Mapping[str, str],
                 process_identity: Callable[[int], Optional[dict]],
                 node: str, gpu_uuids: Sequence[str], gpu_gb: float,
                 network_gbps: float, host_gb: float, storage_address: str,
                 recipe: ContainerRecipe, gate_script: Path,
                 address_prefix: str = '127.77.49.', first_address: int = 20,
                 lifecycle_observer=None):
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}', campaign):
            raise ValueError('invalid campaign identity')
        if not node or len(set(gpu_uuids)) != len(gpu_uuids):
            raise ValueError('physical inventory identity mismatch')
        if any(not re.fullmatch(r'GPU-[0-9a-fA-F-]+', u) for u in gpu_uuids):
            raise ValueError('actual GPU UUIDs required')
        if not 2 <= first_address <= 254:
            raise ValueError('address range exhausted')
        address_environment(storage_address)
        address_environment(address_prefix + str(first_address))
        if not re.fullmatch(r'[0-9a-f]{64}', recipe.image_id):
            raise ValueError('immutable qualified image ID required')
        if not recipe.native_images or '/hydra-entry.py' not in recipe.entrypoint:
            raise ValueError('explicit native image mapping/entrypoint required')
        if recipe.memory_bytes <= 0 or recipe.swap_bytes < 0:
            raise ValueError('finite container limits required')
        if not cpu_ids or not gate_script.is_file():
            raise ValueError('qualified entry gate and CPU set required')
        self.campaign, self.output = campaign, Path(output)
        self.output.mkdir(exist_ok=False)
        self.parent, self.parent_inode = Path(parent), parent_inode
        self.parent_invocation = parent_invocation
        self.parent_limits = dict(parent_limits)
        self.cpus = sorted(set(cpu_ids))
        self.base, self.cli_env = list(podman_base), dict(podman_environment)
        self.identity = process_identity
        self.node, self.uuids = node, tuple(gpu_uuids)
        self.resources = (len(gpu_uuids), gpu_gb, network_gbps, host_gb)
        self.storage_address, self.recipe = storage_address, recipe
        self.gate_script = Path(gate_script)
        self.prefix, self.next_address = address_prefix, first_address
        self.workers: Dict[str, OwnedWorker] = {}
        self.providers: Dict[str, OwnedWorker] = {}
        self.storage: Optional[OwnedWorker] = None
        self.serial = 0
        self.lock = asyncio.Lock()
        self.close_lock = asyncio.Lock()
        self.closing = False
        self.lifecycle_observer = lifecycle_observer
        self._check_parent()

    def _check_parent(self):
        if self.parent.stat().st_ino != self.parent_inode:
            raise RuntimeError('parent cgroup replaced')
        invocation = subprocess.check_output(['systemctl', '--user', 'show',
            self.parent.name, '-p', 'InvocationID', '--value'], text=True).strip()
        if invocation != self.parent_invocation:
            raise RuntimeError('parent invocation replaced')
        for name, value in self.parent_limits.items():
            if (self.parent / name).read_text().strip() != str(value):
                raise RuntimeError('parent limit drift: ' + name)

    def inventory(self):
        self._check_parent()
        return {self.node: self.resources}

    def storage_ip(self, node):
        if node != self.node:
            raise ValueError('unknown physical node')
        if self.storage is None or self.storage.removed:
            raise RuntimeError('storage not started')
        return self.storage_address

    def _save(self, name, value):
        with (self.output / name).open('x') as handle:
            json.dump(value, handle, indent=2)

    async def _call(self, args, *, allow_missing=False):
        self.serial += 1
        stem = f'{self.serial:06d}'
        def execute():
            with (self.output/(stem+'.stdout')).open('x') as out, \
                    (self.output/(stem+'.stderr')).open('x') as err:
                start = time.monotonic()
                proc = subprocess.run(self.base+list(args), env=self.cli_env,
                    stdout=out, stderr=err, timeout=60)
            self._save(stem+'.command.json', dict(args=list(args),
                returncode=proc.returncode, elapsed_s=time.monotonic()-start))
            if proc.returncode and not allow_missing:
                raise RuntimeError(f'Podman command {stem} failed: {proc.returncode}')
            return proc.returncode, (self.output/(stem+'.stdout')).read_text()
        # Cancellation must not race cleanup against a still-running `run` CLI.
        # Finish the bounded infrastructure command before releasing the lock;
        # the caller's finally then inspects/cleans its exact container identity.
        command = asyncio.create_task(asyncio.to_thread(execute))
        try:
            return await asyncio.shield(command)
        except asyncio.CancelledError:
            await command
            raise

    async def _inspect(self, worker):
        # Missing is established by `container exists`, not arbitrary inspect failure.
        code, _ = await self._call(['container', 'exists', worker.container_name], allow_missing=True)
        if code == 1:
            return None
        if code != 0:
            raise RuntimeError('cannot establish container existence')
        _, text = await self._call(['inspect', worker.container_name])
        state = json.loads(text)[0]
        if state['Config']['Labels'].get('org.primelora.campaign') != self.campaign:
            raise RuntimeError('container ownership mismatch')
        if worker.cid is not None and state['Id'] != worker.cid:
            raise RuntimeError('container identity replaced')
        return state

    async def _qualify(self, worker):
        deadline = time.monotonic()+10
        while not (worker.output/'entry.json').exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('entry gate did not acknowledge its process')
            await asyncio.sleep(.01)
        state = await self._inspect(worker)
        if not state or not state['State']['Running']:
            raise RuntimeError('worker exited before resource acknowledgement')
        owner = self.identity(state['State']['Pid'])
        if not owner or owner['uid'] != os.getuid() or owner['affinity'] != self.cpus:
            raise RuntimeError('actual worker PID/UID/affinity mismatch')
        child = Path(owner['cgroup'])
        if self.parent not in child.parents:
            raise RuntimeError('actual worker escaped common parent')
        limits = {k:(child/k).read_text().strip() for k in
                  ('memory.max','memory.swap.max','pids.max')}
        expected = {'memory.max':str(self.recipe.memory_bytes),
                    'memory.swap.max':str(self.recipe.swap_bytes),
                    'pids.max':str(self.recipe.pids_max)}
        if limits != expected:
            raise RuntimeError('actual worker limits differ')
        pid = owner['pid']
        entry = json.loads((worker.output/'entry.json').read_text())
        if (entry['start_ticks'] != owner['start_ticks'] or entry['affinity'] != self.cpus
                or entry['name'] != worker.container_name):
            raise RuntimeError('entry process differs from actual container owner')
        ns = {k:os.readlink(f'/proc/{pid}/ns/{k}') for k in ('ipc','net')}
        if ns['ipc'] == os.readlink('/proc/self/ns/ipc'):
            raise RuntimeError('host shared memory exposed')
        if ns['net'] != os.readlink('/proc/self/ns/net'):
            raise RuntimeError('unexpected network namespace')
        if worker is not self.storage:
            storage_state = await self._inspect(self.storage)
            if not storage_state or not storage_state['State']['Running']:
                raise RuntimeError('private storage owner exited')
            if ns['ipc'] != os.readlink(f"/proc/{storage_state['State']['Pid']}/ns/ipc"):
                raise RuntimeError('worker cannot share private storage IPC')
        if worker.uuid is not None:
            _, visible = await self._call(['exec',worker.cid,
                '/usr/bin/taskset','-c',','.join(map(str,self.cpus)),
                'nvidia-smi','--query-gpu=uuid','--format=csv,noheader'])
            if visible.splitlines() != [worker.uuid]:
                raise RuntimeError('actual container-visible GPU differs from selection')
        self._check_parent()
        worker.owner, worker.child_group = owner, child
        self._save(worker.container_name+'.ownership.json', dict(owner=owner,
            limits=limits, namespaces=ns, container_id=state['Id'], gpu_uuid=worker.uuid))
        # Atomic publication; the entrypoint has not imported torch or loaded a model.
        permit = worker.output/'permit.tmp'
        with permit.open('x') as handle:
            json.dump(dict(container_id=state['Id'], name=worker.container_name,
                           pid_start_ticks=owner['start_ticks']), handle)
        permit.rename(worker.output/'permit.json')

    async def _create(self, worker, environment, ipc, mounts):
        self._check_parent()
        recipe = self.recipe
        worker.output.mkdir(exist_ok=False)
        args = ['run','--detach','--name='+worker.container_name,
            '--pull=never','--cgroups=enabled',
            '--cgroup-parent='+str(self.parent).removeprefix('/sys/fs/cgroup'),
            '--memory='+str(recipe.memory_bytes),
            '--memory-swap='+str(recipe.memory_bytes+recipe.swap_bytes),
            '--pids-limit='+str(recipe.pids_max),'--network=host','--http-proxy=false',
            '--read-only','--cap-drop=ALL','--security-opt=no-new-privileges',
            '--security-opt=label=disable','--label=org.primelora.campaign='+self.campaign,
            '--tmpfs=/tmp:rw,size='+str(recipe.tmp_bytes),
            '--ipc='+ipc,'--volume='+str(worker.output)+':/hydra-output:rw',
            '--volume='+str(self.gate_script)+':/hydra-entry.py:ro']
        if ipc == 'private':
            args.append('--shm-size='+str(environment.pop('_PRIVATE_SHM_BYTES')))
        env = {**recipe.environment, **environment,
               'HYDRASERVE_CONTAINER_NAME':worker.container_name,
               'NVIDIA_VISIBLE_DEVICES':('nvidia.com/gpu='+worker.uuid if worker.uuid else 'void')}
        args += ['--env='+key+'='+str(value) for key,value in sorted(env.items())]
        args += ['--volume='+src+':'+dst+':ro' for src,dst in (*recipe.readonly_mounts, *mounts)]
        args += [recipe.image_id,'/usr/bin/taskset','-c',','.join(map(str,self.cpus)),
                 *recipe.entrypoint]
        _, text = await self._call(args)
        worker.cid = text.strip()
        if not re.fullmatch(r'[0-9a-f]{64}', worker.cid):
            raise RuntimeError('invalid actual container ID')
        await self._qualify(worker)

    @staticmethod
    def _check_address(address, ports):
        address_environment(address)
        # Detection only, not a claim of atomic reservation. Actual listener/init
        # failures must propagate; never treat this as service/model readiness.
        for port in ports:
            with socket.socket() as probe:
                probe.bind((address, port))

    async def start_storage(self, environment, *, shm_bytes, mounts=()):
        async with self.lock:
            self._require_open()
            if self.storage is not None:
                raise RuntimeError('storage creation already attempted')
            if not 0 < shm_bytes < self.recipe.memory_bytes:
                raise ValueError('private shm must fit bounded container')
            self._check_address(self.storage_address, (6666,8888))
            name = self.campaign+'-storage'
            self.storage = OwnedWorker('storage', name, self.storage_address,
                None, self.output/name)
            await self._create(self.storage, {**environment,
                **address_environment(self.storage_address),
                '_PRIVATE_SHM_BYTES':str(shm_bytes)}, 'private', mounts)

    async def create_worker(self, name, image, env, *, node, gpu_id, nas_path, gpu):
        async with self.lock:
            self._require_open()
            if node != self.node or not 0 <= gpu_id < len(self.uuids):
                raise ValueError('unknown physical GPU/node')
            if image not in self.recipe.native_images:
                raise ValueError('unqualified native image mapping')
            if nas_path != self.recipe.native_nas_path:
                raise ValueError('unqualified native model mount mapping')
            if name in self.workers or not re.fullmatch(r'[a-zA-Z0-9_.-]{1,128}', name):
                raise ValueError('duplicate/invalid native worker name')
            if self.storage is None or self.storage.cid is None or self.storage.removed:
                raise RuntimeError('qualified private storage required first')
            if self.next_address > 254:
                raise RuntimeError('project address range exhausted')
            address = self.prefix+str(self.next_address)
            self.next_address += 1
            self._check_address(address, (8080,8090,8091,8092,8093))
            cname = self.campaign+'-w'+str(len(self.workers))
            worker = OwnedWorker(name,cname,address,self.uuids[gpu_id],self.output/cname)
            self.workers[name] = worker
            self._save(cname+'.native-request.json', dict(name=name,image=image,
                env=env,node=node,gpu_id=gpu_id,gpu=gpu,nas_path=nas_path,
                native_gpu_request_is_not_a_physical_partition=True))
            # Acquire the new record's uncontended lock before publishing the
            # registration lock. close/delete must wait for this exact create,
            # including a cancelled but still-running bounded Podman CLI.
            await worker.lifecycle_lock.acquire()
        try:
            if self.lifecycle_observer is not None:
                self.lifecycle_observer.before_create(worker)
            await self._create(worker,{**env,**address_environment(address)},
                               'container:'+self.storage.cid, ())
            if self.lifecycle_observer is not None:
                self.lifecycle_observer.after_create(worker)
        finally:
            worker.lifecycle_lock.release()

    def _require_open(self):
        if self.closing:
            raise RuntimeError('platform is closing; new containers are not admitted')

    async def start_provider(self, name, address, environment):
        """Local CPU backbone provider, charged to the same service parent.

        This is not another physical compute node in the scheduler inventory.
        Native provider protocol/CPU packing executes in its own actual container.
        """
        async with self.lock:
            self._require_open()
            if name in self.providers or not re.fullmatch(r'[a-z0-9-]{1,40}',name):
                raise ValueError('duplicate/invalid provider identity')
            if self.storage is None or self.storage.cid is None or self.storage.removed:
                raise RuntimeError('private storage lifecycle required first')
            self._check_address(address,(8888,))
            cname=self.campaign+'-'+name
            worker=OwnedWorker(name,cname,address,None,self.output/cname)
            self.providers[name]=worker
            await self._create(worker,{**environment,**address_environment(address)},
                               'container:'+self.storage.cid,())

    async def worker_ip(self, name):
        worker = self.workers[name]
        async with worker.lifecycle_lock:
            if worker.removed:
                return None
            state = await self._inspect(worker)
            if not state or not state['State']['Running']:
                raise RuntimeError('native worker exited before /init')
            return worker.address  # NOT a readiness acknowledgement

    def _actual_gone(self, worker):
        if worker.owner:
            current = self.identity(worker.owner['pid'])
            if current and current['start_ticks'] == worker.owner['start_ticks']:
                return False
        if worker.child_group and worker.child_group.exists():
            for path in (worker.child_group/'cgroup.procs', *worker.child_group.rglob('cgroup.procs')):
                if path.exists() and path.read_text().split():
                    return False
        return True

    async def _delete(self, worker):
        if self.lifecycle_observer is not None and worker.uuid and not worker.removed:
            self.lifecycle_observer.before_delete(worker)
        state = await self._inspect(worker)
        if state is not None:
            await self._call(['logs',worker.cid or worker.container_name],allow_missing=True)
            if state['State']['Running']:
                await self._call(['stop','--time=10',worker.cid or worker.container_name])
            await self._call(['rm',worker.cid or worker.container_name])
        if not self._actual_gone(worker):
            raise RuntimeError('container acknowledged removal but owned processes remain')
        if self.lifecycle_observer is not None and worker.uuid and not worker.removed:
            self.lifecycle_observer.after_delete(worker)
        worker.removed = True

    async def delete_worker(self, name):
        worker = self.workers[name]
        async with worker.lifecycle_lock:
            await self._delete(worker)

    async def worker_deleted(self, name):
        worker = self.workers[name]
        async with worker.lifecycle_lock:
            return (worker.removed and await self._inspect(worker) is None
                    and self._actual_gone(worker))

    async def close(self):
        async with self.close_lock:
            # Stop registration and snapshot all owned records, including failed
            # or in-flight creates. Do not hold the registration lock while
            # awaiting container termination. Retain records for cleanup retry.
            async with self.lock:
                self.closing = True
                sequence = list(reversed(list(self.workers.values())))
                sequence += list(reversed(list(self.providers.values())))
                if self.storage:
                    sequence.append(self.storage)
            errors = []
            for worker in sequence:
                try:
                    async with worker.lifecycle_lock:
                        await self._delete(worker)
                except Exception as exc:
                    errors.append(f'{worker.container_name}: {type(exc).__name__}: {exc}')
            if errors:
                raise RuntimeError('owned cleanup incomplete: '+str(errors))
