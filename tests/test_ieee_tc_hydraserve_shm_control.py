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
from HydraServe_project.runtime.shm_control import NativeShmLease, render_app, APP
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


if __name__=='__main__':unittest.main()
