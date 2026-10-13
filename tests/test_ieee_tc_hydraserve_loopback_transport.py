"""CPU TCP/AST witnesses, not native GPU or whole-service qualification."""
import ast
from contextlib import redirect_stdout
from hashlib import sha256
import io
import os
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from HydraServe_project.runtime.loopback_transport import (
    SOURCE_SHA256, address_environment, render_overlays)


NATIVE = Path(__file__).resolve().parents[1]/'vendor_new_baselines/HydraServe_main_20260520'
EXECUTOR = 'vllm/executor/gpu_executor.py'


class TransportTests(unittest.TestCase):
    def test_private_canonical_address_only(self):
        self.assertEqual(address_environment('127.77.49.11'), {
            'HYDRASERVE_BIND_IP':'127.77.49.11','VLLM_HOST_IP':'127.77.49.11',
            'HYDRASERVE_BIND_SOURCE_IP':'1'})
        for address in ('0.0.0.0','192.0.2.10','::1','127.000.0.1','localhost'):
            with self.assertRaises(ValueError):
                address_environment(address)

    def test_input_drift_rejected(self):
        read = Path.read_bytes
        with patch.object(Path,'read_bytes',lambda p:read(p)+b'\n'):
            with self.assertRaisesRegex(ValueError,'unqualified'):
                render_overlays(NATIVE)

    def test_only_optional_address_expressions_change(self):
        rendered = render_overlays(NATIVE)

        class UndoAddress(ast.NodeTransformer):
            def visit_If(self,node):
                if (isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Call)
                    and node.test.left.args and isinstance(node.test.left.args[0],ast.Constant)
                    and node.test.left.args[0].value == 'HYDRASERVE_BIND_SOURCE_IP'):
                    return None
                return self.generic_visit(node)

            def visit_Call(self,node):
                if (isinstance(node.func,ast.Attribute) and node.func.attr == 'getenv'
                    and node.args and isinstance(node.args[0],ast.Constant)
                    and node.args[0].value == 'HYDRASERVE_BIND_IP'):
                    return node.args[1]
                return self.generic_visit(node)

        for name, patched in rendered.items():
            native = (NATIVE/name).read_bytes()
            self.assertEqual(sha256(native).hexdigest(),SOURCE_SHA256[name])
            self.assertEqual(ast.dump(ast.parse(native),include_attributes=False),
                ast.dump(UndoAddress().visit(ast.parse(patched)),include_attributes=False),name)

    def test_kernel_default_source_is_not_the_replica_alias(self):
        # Actual socket witness for why unqualified host-network substitution
        # breaks native ip_list.index(address[0]); no native busy-loop failure.
        with socket.socket() as server, socket.socket() as client:
            server.bind(('127.77.49.10',0))
            server.listen(1)
            client.connect(server.getsockname())
            accepted, peer = server.accept()
            accepted.close()
            self.assertEqual(peer[0],'127.0.0.1')
            self.assertNotEqual(peer[0],'127.77.49.11')

    def test_native_two_stage_tcp_peer_identity_and_payload(self):
        raw = render_overlays(NATIVE)[EXECUTOR]
        tree = ast.parse(raw)
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'GPUExecutor')
        method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == 'init_connection')
        future = ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        selected = ast.fix_missing_locations(ast.Module(body=[future,method],type_ignores=[]))
        namespace = dict(os=os,socket=socket,time=time)
        exec(compile(selected,'pinned-native-init-connection','exec'),namespace)
        addresses = ['127.77.49.11','127.77.49.12']
        # Fail before threads if any required fixed native endpoint is occupied.
        for address,port in ((addresses[0],8090),(addresses[0],8092),(addresses[1],8092)):
            with socket.socket() as test:
                test.bind((address,port))
        mappings = [[],[]]
        workers = [NS(pp_rank=rank,pp_size=2,driver_worker=NS(model_runner=NS(
            model=NS(set_socket=lambda a,b,rank=rank:mappings[rank].append((a,b))))))
            for rank in range(2)]
        errors = []

        def connect(rank):
            try:
                namespace['init_connection'](workers[rank],addresses)
            except BaseException as exc:
                errors.append(exc)

        with patch.dict(os.environ,{'HYDRASERVE_BIND_SOURCE_IP':'1'}), redirect_stdout(io.StringIO()):
            threads = [threading.Thread(target=connect,args=(rank,),daemon=True) for rank in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
        self.assertFalse(any(thread.is_alive() for thread in threads),'native connection did not finish')
        self.assertEqual(errors,[])
        connections = {id(s):s for worker in workers
                       for pool in (worker.invoke_sockets,worker.comm_sockets) for s in pool if s}
        try:
            for rank,worker in enumerate(workers):
                peer = 1-rank
                for channel in ('invoke_sockets','comm_sockets'):
                    channel_socket = getattr(worker,channel)[peer]
                    self.assertEqual(channel_socket.getsockname()[0],addresses[rank])
                    self.assertEqual(channel_socket.getpeername()[0],addresses[peer])
                    channel_socket.settimeout(2)
                self.assertEqual(len(mappings[rank]),1)
            for channel in ('invoke_sockets','comm_sockets'):
                first, second = getattr(workers[0],channel)[1],getattr(workers[1],channel)[0]
                first.sendall(b'native-pipeline-address-witness')
                self.assertEqual(second.recv(128),b'native-pipeline-address-witness')
                second.sendall(b'ack')
                self.assertEqual(first.recv(3),b'ack')
        finally:
            for connection in connections.values():
                connection.close()


if __name__ == '__main__':
    unittest.main()
