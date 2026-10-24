"""Real loopback control-socket tests; no SHM/model/loader simulation claims."""
import ast
import asyncio
import json
import socket
import threading
import time
import types
import unittest
from unittest.mock import patch
from HydraServe_project.runtime.shm_control import NativeShmLease, render_app, render_storage_receive, APP, STORAGE
from HydraServe_project.runtime.native_delivery import render_overlays
from HydraServe_project.scripts.probe_layer_identity import NATIVE


class Peer:
    def __init__(self, response='valid'):
        self.listener=socket.socket(); self.listener.bind(('127.0.0.1',0)); self.listener.listen()
        self.listener.settimeout(3)
        self.port=self.listener.getsockname()[1]
        self.response=response; self.header=None; self.error=None
        self.received=threading.Event(); self.eof=threading.Event()
        self.thread=threading.Thread(target=self.serve,daemon=True); self.thread.start()
    def serve(self):
        try:
            with self.listener.accept()[0] as conn:
                conn.settimeout(3)
                def exact(n):
                    data=b''
                    while len(data)<n:
                        chunk=conn.recv(n-len(data))
                        if not chunk: raise EOFError('test header truncated')
                        data+=chunk
                    return data
                n=int.from_bytes(exact(8),'little'); self.header=exact(n); self.received.set()
                payload=(32).to_bytes(8,'little',signed=True)+(4096).to_bytes(8,'little',signed=True)
                if self.response=='truncated': conn.sendall(payload[:3]); return
                if self.response=='valid':
                    # Split a fixed-width field: client must not confuse TCP reads with frames.
                    conn.sendall(payload[:3]); time.sleep(.005); conn.sendall(payload[3:]+b'psm_test')
                if conn.recv(50)==b'': self.eof.set()
        except BaseException as exc: self.error=exc
        finally:self.listener.close()
    def done(self):
        self.thread.join(3)
        if self.thread.is_alive(): raise AssertionError('control peer remained alive')
        if self.error: raise self.error


class ControlTests(unittest.TestCase):
    def test_original_header_survives_startup_loop_and_releases_on_new_loop(self):
        peer=Peer(); lease=NativeShmLease(); header=b'{"model":"existing","pre_load":"no"}'
        self.assertEqual(asyncio.run(lease.begin('127.0.0.1',header,peer.port)),('psm_test',32,4096))
        self.assertFalse(peer.eof.is_set())
        async def end():self.assertTrue(lease.close())
        asyncio.run(end()); peer.done()
        self.assertTrue(peer.eof.is_set()); self.assertEqual(peer.header,header)
        self.assertFalse(lease.close())

    def test_cancel_closes_actual_connection(self):
        peer=Peer('wait'); lease=NativeShmLease()
        async def work():
            task=asyncio.create_task(lease.begin('127.0.0.1',b'header',peer.port))
            self.assertTrue(await asyncio.to_thread(peer.received.wait,1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
        asyncio.run(work()); peer.done()
        self.assertTrue(peer.eof.is_set()); self.assertFalse(lease.close())

    def test_truncated_descriptor_is_failure_not_ready(self):
        peer=Peer('truncated'); lease=NativeShmLease()
        with self.assertRaises(EOFError):asyncio.run(lease.begin('127.0.0.1',b'header',peer.port))
        peer.done(); self.assertFalse(lease.close())

    def test_concurrent_start_does_not_replace_or_close_original_owner(self):
        peer=Peer(); lease=NativeShmLease()
        asyncio.run(lease.begin('127.0.0.1',b'header',peer.port))
        with self.assertRaisesRegex(RuntimeError,'active'):
            asyncio.run(lease.begin('127.0.0.1',b'other',peer.port))
        self.assertFalse(peer.eof.is_set()); lease.close(); peer.done()

    def test_next_lease_after_release(self):
        lease=NativeShmLease()
        for header in (b'initial',b'destination'):
            peer=Peer(); asyncio.run(lease.begin('127.0.0.1',header,peer.port))
            lease.close(); peer.done(); self.assertEqual(peer.header,header)

    def test_app_nonconnection_policy_and_startup_overlap_unchanged(self):
        old=render_overlays(NATIVE,request_contract=True)[APP].decode()
        new=render_overlays(NATIVE,request_contract=True,shm_control=True)[APP].decode()
        def definitions(source):
            return {n.name:ast.dump(n) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))
                    and n.name not in ('start_load_model_shm','end_load_model_shm')}
        self.assertEqual(definitions(old),definitions(new))
        self.assertLess(new.index('asyncio.run(start_load_model_shm'),new.index('import vllm'))
        self.assertIn('await end_load_model_shm()',new)
        with self.assertRaisesRegex(ValueError,'already changed'):render_app(new,NATIVE)

    def test_actual_rendered_functions_release_when_loader_fails(self):
        peer=Peer(); lease=NativeShmLease(); calls=[]
        source=render_overlays(NATIVE,request_contract=True,shm_control=True)[APP].decode()
        tree=ast.parse(source)
        tree.body=[n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name in ('start_load_model_shm','end_load_model_shm')]
        class BoundPort:
            async def begin(self,address,header):return await lease.begin(address,header,peer.port)
            def close(self):return lease.close()
        def loader(*args):calls.append(args); raise RuntimeError('loader failure witness')
        ns=dict(time=time,json=json,os=types.SimpleNamespace(getenv=lambda _: '127.0.0.1'),
            shm_control_lease=BoundPort(),torch=types.SimpleNamespace(ops=types.SimpleNamespace(
                download_model=types.SimpleNamespace(start_load_shm=loader))))
        exec(compile(tree,APP,'exec'),ns)
        with self.assertRaisesRegex(RuntimeError,'loader failure witness'):
            asyncio.run(ns['start_load_model_shm']('existing',1,2,True))
        peer.done(); self.assertTrue(peer.eof.is_set())
        self.assertEqual(calls,[('psm_test',32,4096)])
        self.assertEqual(json.loads(peer.header),dict(model='existing',pp_rank=1,pp_size=2,is_dest=True,pre_load='no'))


class ReceiverTests(unittest.TestCase):
    def method(self, name, *, fixed, namespace):
        source = render_overlays(NATIVE, bounded_shm_receive=fixed)[STORAGE]
        owner = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'CachedStateDict')
        node = next(n for n in owner.body if getattr(n, 'name', None) == name)
        tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(tree, '<actual-native-receiver>', 'exec'), namespace)
        return namespace[name]

    def receive(self, *, fixed, payload, expected_size, tail=0, fragment=None):
        # Actual socket.recv_into with a small memoryview reproduces the same
        # byte-boundary error without allocating a multi-GiB shared mapping.
        left, right = socket.socketpair()
        self.addCleanup(left.close); self.addCleanup(right.close)
        left.settimeout(1); right.sendall(payload); right.shutdown(socket.SHUT_WR)
        storage = bytearray(8 + expected_size + tail)
        storage[8+expected_size:] = b'Z' * tail
        calls = []
        class Connection:
            def connect(self, addr): pass
            def sendall(self, header): pass
            def recv_into(self, buf, nbytes):
                calls.append((len(buf), nbytes))
                return left.recv_into(buf, min(nbytes, fragment) if fragment else nbytes)
            def close(self): left.close()
        shm = types.SimpleNamespace(buf=memoryview(storage), close=lambda: None)
        ns = dict(time=time, shared_memory=types.SimpleNamespace(SharedMemory=lambda **kw: shm),
            socket=types.SimpleNamespace(AF_INET=socket.AF_INET, SOCK_STREAM=socket.SOCK_STREAM,
                                        socket=Connection), CachedStateDict=types.SimpleNamespace(NUM_THREAD=2))
        # Connection factory has the same two positional arguments as socket().
        ns['socket'].socket = lambda *args: Connection()
        obj = types.SimpleNamespace(num_server=1, server_addr='unused', req_header={}, chunk_size=32,
                                    shm_addr=0, process_header=lambda _: b'header')
        self.method('recv_part_content', fixed=fixed, namespace=ns)(obj, 0, 'test', 8, 8+expected_size, expected_size)
        return storage, calls

    def test_original_receiver_reproduces_short_tail_error(self):
        with self.assertRaisesRegex(ValueError, 'buffer too small'):
            self.receive(fixed=False, payload=b'abcde', expected_size=5)

    def test_exact_tail_and_partial_reads_finish_without_overflow(self):
        for fragment in (None, 3):
            storage, calls = self.receive(fixed=True, payload=b'x'*37, expected_size=37,
                                          fragment=fragment)
            self.assertEqual(storage[8:], b'x'*37)
            self.assertEqual(int.from_bytes(storage[:8], 'little'), 45)
            self.assertTrue(all(n <= size for size, n in calls))
            self.assertEqual(calls[-1][1], 5 if fragment is None else 1)

    def test_neighboring_partition_is_not_overwritten(self):
        storage, calls = self.receive(fixed=True, payload=b'payload'+b'EXTRA', expected_size=7, tail=40)
        self.assertEqual(storage[8:15], b'payload')
        self.assertEqual(storage[15:], b'Z'*40)
        self.assertEqual(calls, [(7, 7)])

    def test_early_eof_is_failure(self):
        with self.assertRaisesRegex(EOFError, 'truncated'):
            self.receive(fixed=True, payload=b'abc', expected_size=7)

    def test_parent_never_publishes_complete_after_child_failure(self):
        async def run(exitcodes):
            storage = bytearray(128); processes = []; writer = types.SimpleNamespace(closed=False)
            async def noop(): pass
            async def read(n): return (64).to_bytes(8, 'little')
            def close(): writer.closed = True
            writer.write=lambda _: None; writer.drain=noop; writer.close=close; writer.wait_closed=noop
            async def connect(*args, **kwargs): return types.SimpleNamespace(read=read), writer
            async def allocate(*args): return 0
            class Process:
                def __init__(self, **kwargs):
                    self.exitcode=exitcodes[len(processes)]; self.joined=False; processes.append(self)
                def start(self): pass
                def is_alive(self): return False
                def join(self): self.joined=True
            ns = dict(time=time, asyncio=types.SimpleNamespace(open_connection=connect, sleep=asyncio.sleep),
                model_memory=types.SimpleNamespace(allocate_buffer=allocate,
                    shm=types.SimpleNamespace(buf=storage, name='test')),
                CachedStateDict=types.SimpleNamespace(NUM_THREAD=2),
                multiprocessing=types.SimpleNamespace(Process=Process))
            obj = types.SimpleNamespace(has_model=False, req_header={'model':'existing'},
                server_addr='unused', chunk_size=32, cache_id=0, buffer=False,
                process_header=lambda *a, **kw: b'header', recv_part_content=None)
            method = self.method('start_download_model', fixed=True, namespace=ns)
            if any(exitcodes):
                with self.assertRaisesRegex(RuntimeError, 'completion not published'): await method(obj)
                self.assertEqual(int.from_bytes(storage[:8], 'little'), 8)
            else:
                await method(obj)
                self.assertEqual(int.from_bytes(storage[:8], 'little'), 72)
            self.assertTrue(writer.closed)
            self.assertTrue(all(p.joined for p in processes))
        asyncio.run(run([0, 1])); asyncio.run(run([0, 0]))

    def test_opt_in_changes_only_receiver_and_completion(self):
        old = render_overlays(NATIVE)
        new = render_overlays(NATIVE, bounded_shm_receive=True)
        self.assertEqual(set(old), set(new))
        for key in old.keys()-{STORAGE}: self.assertEqual(old[key], new[key])
        def methods(source):
            owner=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='CachedStateDict')
            return {n.name: ast.dump(n) for n in owner.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        before, after = methods(old[STORAGE]), methods(new[STORAGE])
        self.assertEqual({key for key in before if before[key] != after[key]},
                         {'recv_part_content', 'start_download_model'})
        with self.assertRaisesRegex(ValueError, 'already changed'):
            render_storage_receive(new[STORAGE].decode(), NATIVE)


if __name__=='__main__':unittest.main()
