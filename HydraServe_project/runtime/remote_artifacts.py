"""Request-only native adapter views over the shared published-artifact resolver.

The existing resolver owns HTTP, integrity, miss coalescing and cancellation.
This single-controller owner publishes only lossless tokenizer metadata views;
weights are symlinks to verified downloaded files. No prefetch or policy tuning.
"""
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import time
import uuid

from .tokenizer_compat import materialize_metadata_view


class NativeArtifactViews:
    def __init__(self, resolver, root):
        self.resolver = resolver
        self.root = Path(root).absolute()
        # A single controller owns fresh run-local views. Never trust leftovers.
        self.root.mkdir(mode=0o700)
        self.staging = self.root/'.staging'
        self.staging.mkdir()
        self.receipts = self.root/'.receipts'
        self.receipts.mkdir()
        self._locks = {}

    @staticmethod
    def _identity(view):
        if view.is_symlink() or not view.is_dir():
            raise ValueError('native adapter view is not an owned directory')
        result = {}
        for path in view.iterdir():
            stat = path.lstat()
            result[path.name] = dict(identity=[stat.st_dev,stat.st_ino,stat.st_size,
                stat.st_mtime_ns,stat.st_ctime_ns],
                link=os.readlink(path) if path.is_symlink() else None)
        return result

    def _publish(self, adapter_id, source, destination):
        receipt_path = self.receipts/(adapter_id+'.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (receipt['source'] != str(source) or receipt['destination'] != str(destination)
                    or receipt['view_identity'] != self._identity(destination)):
                raise ValueError('native adapter metadata view changed')
            return receipt
        if destination.exists() or destination.is_symlink():
            raise ValueError('native adapter view has no verified receipt')
        staging = self.staging/uuid.uuid4().hex
        staging.mkdir()
        staged = staging/adapter_id
        # A failed publication stays as evidence; never silently use an older view.
        receipt = materialize_metadata_view(source, staged)
        # Moving a directory between parents updates its '..' entry, requiring
        # owner write permission. Seal only after it reaches its final parent;
        # consumers cannot use the view until this operation returns a receipt.
        staged.chmod(0o755)
        staged.rename(destination)
        destination.chmod(0o555)
        staging.rmdir()
        receipt['destination'] = str(destination)
        receipt['view_identity'] = self._identity(destination)
        with receipt_path.open('x') as handle:
            json.dump(receipt, handle, sort_keys=True)
        return receipt

    async def prepare(self, contract):
        destination = self.root/contract.adapter_id
        if str(destination) != contract.adapter_path:
            raise ValueError('native view differs from the static adapter registry')
        source = self.resolver.root/'objects'/contract.adapter_id
        started = time.perf_counter()
        evidence = await self.resolver.resolve(contract.adapter_id, str(source), contract.request_id)
        if (evidence.get('tc_artifact_request_id') != contract.request_id
                or evidence.get('tc_artifact_id') != contract.adapter_id
                or evidence.get('tc_artifact_delivery_mode') != 'prepublished_gzip_v1'):
            raise ValueError('common remote resolver returned a mismatched artifact')
        lock = self._locks.setdefault(contract.adapter_id, asyncio.Lock())
        async with lock:
            task = asyncio.create_task(asyncio.to_thread(self._publish,
                contract.adapter_id, source, destination))
            try:
                view = await asyncio.shield(task)
            except asyncio.CancelledError:
                # Do not release the lock/tear down mounts while a writer runs.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not task.cancelled():
                    task.exception()
                raise
        return dict(evidence, native_adapter_path=str(destination),
            native_view_started_s=started, native_view_completed_s=time.perf_counter(),
            native_view_receipt_sha256=sha256((self.receipts/(contract.adapter_id+'.json')).read_bytes()).hexdigest(),
            native_tokenizer_schema_changed=view['schema_changed'],
            native_weight_bytes_copied=view['copied_weight_bytes'])
